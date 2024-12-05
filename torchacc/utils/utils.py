# This file is partially derived from
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/utils.py

import dataclasses
import pickle
from bisect import bisect_left
from typing import Mapping, OrderedDict

import numpy as np
import torch
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence


def call_to_str(base, *args, **kwargs):
    """Construct a string representation of a call.

    Args:
        base (str): name of the call.
        args (tuple, optional): args to ``base``.
        kwargs (dict, optional): kwargs supplied to ``base``.

    Returns:
        str: A string representation of ``base(*args, **kwargs)``.
    """
    name = f'{base}('
    if args:
        name += ', '.join(repr(arg) for arg in args)
        if kwargs:
            name += ', '
    if kwargs:
        name += ', '.join(f'{key}={repr(arg)}' for key, arg in kwargs.items())
    name += ')'
    return name


def partition_uniform(num_items, num_parts):
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    chunksize = num_items // num_parts
    residual = num_items - (chunksize * num_parts)

    parts = np.arange(0, (num_parts + 1) * chunksize, chunksize)

    for i in range(residual):
        parts[i + 1:] += 1
    parts = parts.tolist()

    return parts


def _lprobe(weights, num_parts, bottleneck):
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(
            weights, bsum, lo=step - chunksize, hi=min(step, num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight


def _rb_partition_balanced(weights, num_parts, eps):
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def prefix_sum_inc(weights):
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_


def partition_balanced(weights, num_parts, eps=1e-3):
    num_items = len(weights)
    # First check for the trivial edge case
    if num_items <= num_parts:
        return partition_uniform(num_items, num_parts)

    weights_ = prefix_sum_inc(weights)

    # Find the smallest bottleneck (weight of heaviest partition)
    bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

    # Now compute that partitioning
    parts, success = _lprobe(weights_, num_parts, bottleneck)
    assert success

    return parts


# from transformers
def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) > 0:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class
    return None


# from accelerate.utils
WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__',
                       '__annotations__')
WRAPPER_UPDATES = ('__dict__',)


def update_wrapper(wrapper,
                   wrapped,
                   assigned=WRAPPER_ASSIGNMENTS,
                   updated=WRAPPER_UPDATES):
    """Update a wrapper function to look like the wrapped function

       wrapper is the function to be updated
       wrapped is the original function
       assigned is a tuple naming the attributes assigned directly
       from the wrapped function to the wrapper function (defaults to
       functools.WRAPPER_ASSIGNMENTS)
       updated is a tuple naming the attributes of the wrapper that
       are updated with the corresponding attribute from the wrapped
       function (defaults to functools.WRAPPER_UPDATES)
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def is_namedtuple(data):
    """
    Checks if `x` is a `namedtuple` or not. Can have false positives, but only if a user is trying to mimic a
    `namedtuple` perfectly.
    """
    data_type = type(data)
    bases = data_type.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(data_type, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(isinstance(member, str) for member in fields)


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple, or namedtuple)
    """
    # Some objects may not be able to instantiate from a generator directly
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    else:
        return type(obj)(generator)


def recursively_apply(func,
                      data,
                      *args,
                      test_type=is_torch_tensor,
                      error_on_other_type=False,
                      **kwargs):
    """
    Recursively apply a function on a data structure that is a nested list/tuple/dictionary of a given base type.

    Args:
        func (`callable`):
            The function to recursively apply.
        data (nested list/tuple/dictionary of `main_type`):
            The data on which to apply `func`
        *args:
            Positional arguments that will be passed to `func` when applied on the unpacked data.
        main_type (`type`, *optional*, defaults to `torch.Tensor`):
            The base type of the objects to which apply `func`.
        error_on_other_type (`bool`, *optional*, defaults to `False`):
            Whether to return an error or not if after unpacking `data`, we get on an object that is not of type
            `main_type`. If `False`, the function will leave objects of types different than `main_type` unchanged.
        **kwargs:
            Keyword arguments that will be passed to `func` when applied on the unpacked data.

    Returns:
        The same data structure as `data` with `func` applied to every object of type `main_type`.
    """
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (recursively_apply(
                func,
                o,
                *args,
                test_type=test_type,
                error_on_other_type=error_on_other_type,
                **kwargs) for o in data),
        )
    elif isinstance(data, Mapping):
        return type(data)({
            k: recursively_apply(
                func,
                v,
                *args,
                test_type=test_type,
                error_on_other_type=error_on_other_type,
                **kwargs) for k, v in data.items()
        })
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Unsupported types ({type(data)}) passed to `{func.__name__}`. Only nested list/tuple/dicts of "
            f"objects that are valid for `{test_type.__name__}` should be passed."
        )
    return data


@torch.compiler.disable(recursive=False)
def convert_to_fp32(tensor):
    """
    Recursively converts the elements nested list/tuple/dictionary of tensors in FP16/BF16 precision to FP32.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to convert from FP16/BF16 to FP32.

    Returns:
        The same data structure as `tensor` with all tensors that were in FP16/BF16 precision converted to FP32.
    """

    def _convert_to_fp32(tensor):
        return tensor.float()

    def _is_fp16_bf16_tensor(tensor):
        return hasattr(
            tensor, "dtype") and tensor.dtype in (torch.float16, torch.bfloat16)

    return recursively_apply(
        _convert_to_fp32, tensor, test_type=_is_fp16_bf16_tensor)


class ConvertOutputsToFp32:
    """
    Decorator to apply to a function outputing tensors (like a model forward pass) that ensures the outputs in FP16
    precision will be convert back to FP32.

    Args:
        model_forward (`Callable`):
            The function which outputs we want to treat.

    Returns:
        The same function as `model_forward` but with converted outputs.
    """

    def __init__(self, model_forward):
        self.model_forward = model_forward
        update_wrapper(self, model_forward)

    def __call__(self, *args, **kwargs):
        return convert_to_fp32(self.model_forward(*args, **kwargs))

    def __getstate__(self):
        raise pickle.PicklingError(
            "Cannot pickle a prepared model with automatic mixed precision, please unwrap the model with `Accelerator.unwrap_model(model)` before pickling it."
        )


def convert_outputs_to_fp32(model_forward):
    model_forward = ConvertOutputsToFp32(model_forward)

    def forward(*args, **kwargs):
        return model_forward(*args, **kwargs)

    # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`
    forward.__wrapped__ = model_forward

    return forward


# Copied from torch.distributed.utils._apply_to_tensors
def apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        if isinstance(x, torch.Tensor):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            for f in dataclasses.fields(dc):
                name = f.name
                setattr(dc, name, apply(getattr(dc, name)))
            return dc
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)
