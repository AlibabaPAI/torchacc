import inspect
import torch


# Note: some kwargs are used by output_fn
# TODO: We may need to refer to the inspect.Signature.bind interface to improve this function.
def bind_args_to_kwargs(args, kwargs, sig: inspect.Signature):
    parameters = iter(sig.parameters.values())
    for arg in args:
        para_val = next(parameters)
        if para_val.kind in (inspect.Parameter.KEYWORD_ONLY,
                             inspect.Parameter.VAR_KEYWORD):
            raise ValueError(f"too many positional arguments")
        if para_val.name in kwargs:
            raise ValueError(f"multiple values for argument {para_val.name}")
        kwargs[para_val.name] = arg
    return kwargs


def split_args_into_chunks(args, chunks):
    # args_chunks: [num args, num chunks]
    args_chunks = []
    for arg in args:
        assert isinstance(arg, torch.Tensor), f"all args should be tensors"
        batch_size = arg.size(dim=0)
        assert batch_size % chunks == 0, f"batch_size {batch_size} cannot be evenly divided by the num_micro_batches {chunks}"
        args_chunks.append(arg.chunk(chunks))
    # transpose to: [num chunks, num args]
    args_chunks = [list(i) for i in zip(*args_chunks)]
    return args_chunks


def split_args_kwargs_into_chunks(args, kwargs, chunks):
    # args_chunks: [num chunks, num args]
    args_chunks = []
    # kwargs_chunks: { key1: [num chunks], key2: [num chunks], ...}
    kwargs_chunks = {}

    args_chunks = split_args_into_chunks(args, chunks)

    for key, value in kwargs.items():
        assert isinstance(value,
                          torch.Tensor), f"value in kwargs should be tensor"
        batch_size = value.size(dim=0)
        assert batch_size % chunks == 0, f"batch_size {batch_size} cannot be evenly divided by the num_micro_batches {chunks}"
        kwargs_chunks[key] = value.chunk(chunks)

    return args_chunks, kwargs_chunks
