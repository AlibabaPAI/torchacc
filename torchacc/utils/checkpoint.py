import functools
from typing import Any, Dict, Iterator, Optional, Set, Tuple

import torch
from torch.distributed.utils import _pack_kwargs, _replace_by_prefix, _unpack_kwargs
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_utils_checkpoint
import torch.fx as fx
from torch.fx.passes.split_module import split_module

from torchacc.utils.import_utils import is_torch_xla_available
from torchacc.utils.utils import get_module_class_from_name

if is_torch_xla_available():
    import torch_xla
    import torch_xla.distributed.fsdp.wrap as xla_wrap
    checkpoint_module = torch_xla.distributed.fsdp.checkpoint_module


_CHECKPOINT_WRAPPED_MODULE = "_checkpoint_wrapped_module"
_CHECKPOINT_PREFIX = _CHECKPOINT_WRAPPED_MODULE + "."


class ActivationWrapper(torch.nn.Module):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """

    def __init__(self, mod):
        super().__init__()
        self._checkpoint_wrapped_module = mod
        # state_dict post hook to remove prefix to allow loading into a
        # non-checkpoint wrapped module.
        self._register_state_dict_hook(self._post_state_dict_hook)
        # load_state_dict pre-hook to allow loading back into
        # checkpoint-wrapped module.
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def forward(self, *args, **kwargs):
        raise ValueError("Subclasses should implement forward().")

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._checkpoint_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self._checkpoint_wrapped_module.__getitem__(key)  # type: ignore[operator]

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
        for param_name, param in super().named_parameters(*args, **kwargs):
            yield param_name.replace(_CHECKPOINT_PREFIX, ""), param

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> Dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
        _replace_by_prefix(state_dict, f"{prefix}{_CHECKPOINT_PREFIX}", prefix)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """
        _replace_by_prefix(state_dict, prefix, prefix + f"{_CHECKPOINT_PREFIX}")


class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """

    def __init__(
        self,
        mod: torch.nn.Module,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = functools.partial(
                torch_utils_checkpoint,
                use_reentrant=False,
                **checkpoint_fn_kwargs,
            )
        else:
            # Construct user-specified checkpoint function.
            self.checkpoint_fn = functools.partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    @torch.compile(backend="openxla")
    def _fn(self, *args, **kwargs):
        return self._checkpoint_wrapped_module(*args, **kwargs)

    @torch.compiler.disable
    def forward(self, *args, **kwargs):
        flat_args, kwarg_keys = torch.distributed.algorithms._checkpoint.checkpoint_wrapper._pack_kwargs(*args, **kwargs)

        # @torch.compiler.disable
        def model_fn(*inputs):
            # torch._dynamo.graph_break()
            unpacked_args, unpacked_kwargs = torch.distributed.algorithms._checkpoint.checkpoint_wrapper._unpack_kwargs(inputs, kwarg_keys)
            with torch._dynamo.utils.disable_cache_limit():
                return self._checkpoint_wrapped_module(*unpacked_args, **unpacked_kwargs)
        return torch.utils.checkpoint.checkpoint(model_fn, *flat_args, use_reentrant=True)

        # torch._dynamo.graph_break()
        return self.checkpoint_fn(  # type: ignore[misc]
                self._fn, *args, **kwargs
            )


def fx_checkpoint(graph_model: fx.GraphModule,
                  layer_cls: Set[str],
                  model_name: Optional[str] = None,
                  qualname_map: Optional[Dict[str, str]] = None):
    curr_mod = None
    curr_idx = 0

    modules_to_gc = []

    def split_callback(n: torch.fx.node.Node):
        nonlocal curr_mod, curr_idx
        found = False
        if "nn_module_stack" in n.meta:
            for mod, t in n.meta["nn_module_stack"].items():
                if t[1].__name__ in layer_cls:
                    if mod != curr_mod:
                        curr_mod = mod
                        curr_idx += 1
                        modules_to_gc.append(f"submod_{curr_idx}")
                    found = True
                    break
        if not found and curr_mod is not None:
            curr_mod = None
            curr_idx += 1
        return curr_idx

    # Ask split_module to return mapping from new qualname to old qualname
    new_qualname_map: Dict[str, str] = {}
    split = split_module(graph_model, None, split_callback, new_qualname_map)

    # Update qualname_map
    # TODO: the names of the submodules of the model need to be restored.
    if qualname_map is not None:
        for k, v in new_qualname_map.items():
            v = f"{model_name}.{v}"
            if v in qualname_map and k != v:
                assert k not in qualname_map
                qualname_map[k] = qualname_map[v]
                del qualname_map[v]
            elif v not in qualname_map:
                assert k not in qualname_map
                qualname_map[k] = v

    for name in modules_to_gc:
        assert hasattr(split, name)
        mod = getattr(split, name)
        setattr(split, name, checkpoint_module(mod))
    return split


def gradient_checkpoint(model: torch.nn.Module, gc_cls: Set[str]):
    if isinstance(model, fx.GraphModule):
        return fx_checkpoint(model, gc_cls)
    cls = set()
    for name in gc_cls:
        c = get_module_class_from_name(model, name)
        assert c, f"class {name} in gc_cls not found in model"
        cls.add(c)

    import torchacc as ta
    if ta.is_lazy_device(next(model.parameters()).device):
        auto_wrapper_callable = lambda m, *args, **kwargs: checkpoint_module(m)
    else:
        auto_wrapper_callable = lambda m, *args, **kwargs: CheckpointWrapper(torch.compile(m, backend="openxla"))
        # auto_wrapper_callable = lambda m, *args, **kwargs: CheckpointWrapper(m)

    auto_wrap_policy = functools.partial(
        xla_wrap.transformer_auto_wrap_policy, transformer_layer_cls=cls)
    # auto_wrapper_callable = lambda m, *args, **kwargs: checkpoint_module(m)
    model, n_params = xla_wrap.recursive_wrap(model, auto_wrap_policy,
                                        auto_wrapper_callable, (), ())
    return model

    if ta.is_lazy_device(next(model.parameters()).device):
        auto_wrap_policy = functools.partial(
            xla_wrap.transformer_auto_wrap_policy, transformer_layer_cls=cls)
        auto_wrapper_callable = lambda m, *args, **kwargs: checkpoint_module(m)
        model, n_params = xla_wrap.recursive_wrap(model, auto_wrap_policy,
                                            auto_wrapper_callable, (), ())
    else:
        if hasattr(model, "module"):
            model.module.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()
        # cuda device
        # def checkpoint_fn(model, *args, **kwargs):
        #     flat_args, kwarg_keys = torch.distributed.algorithms._checkpoint.checkpoint_wrapper._pack_kwargs(*args, **kwargs)
        #     @torch.compile(backend="openxla")
        #     def model_fn(*inputs):
        #         unpacked_args, unpacked_kwargs = torch.distributed.algorithms._checkpoint.checkpoint_wrapper._unpack_kwargs(inputs, kwarg_keys)
        #         return model(*unpacked_args, **unpacked_kwargs)
        #     return torch.utils.checkpoint.checkpoint(model_fn, *flat_args)

        # non_reentrant_wrapper = functools.partial(
        #     checkpoint_wrapper,
        #     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        #     checkpoint_fn=checkpoint_fn,
        # )
        # cls = tuple(cls)
        # check_fn = lambda submodule: isinstance(submodule, cls)
        # apply_activation_checkpointing(
        #     model,
        #     checkpoint_wrapper_fn=non_reentrant_wrapper,
        #     check_fn=check_fn)
    return model
