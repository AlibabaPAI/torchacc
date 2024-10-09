import functools
from types import MethodType
from typing import Dict, Optional, Set

import torch
import torch.distributed.fsdp as torch_fsdp
import torch.fx as fx
from torch.fx.passes.split_module import split_module
import torch_xla.distributed.fsdp as xla_fsdp

from torchacc.config import Config
from torchacc.dist import ParallelModule
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.trace as trace
import torchacc.utils.utils as utils


def split_fsdp_wrap_modules(
        graph_model: fx.GraphModule,
        layer_cls: Set[str],
        model_name: Optional[str] = None,
        qualname_map: Optional[Dict[str, str]] = None) -> fx.GraphModule:
    curr_mod = None
    curr_idx = 0

    modules_types = {}

    def split_callback(n: torch.fx.node.Node):
        nonlocal curr_mod, curr_idx
        found = False
        if "nn_module_stack" in n.meta:
            for mod, t in n.meta["nn_module_stack"].items():
                type = t[1]
                if type.__name__ in layer_cls:
                    if mod != curr_mod:
                        curr_mod = mod
                        curr_idx += 1
                        modules_types[f"submod_{curr_idx}"] = type.__name__
                    found = True
                    break
        if not found and curr_mod is not None:
            curr_mod = None
            curr_idx += 1
        return curr_idx

    # Ask split_module to return mapping from new qualname to old qualname
    new_qualname_map: Dict[str, str] = {}
    split = split_module(graph_model, None, split_callback, new_qualname_map)
    # This is needed. FSDP will register a hook for the input tensor,
    # which will result in the recompilation of the computational graph.
    trace.move_single_param_to_callee(split, new_qualname_map)

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

    # Update the class name of the wrap modules
    for mod_name, mod_type in modules_types.items():
        assert hasattr(split, mod_name)
        mod = getattr(split, mod_name)
        mod.__class__.__name__ = mod_type
    return split


def fx_auto_wrap_policy(
    module: torch.nn.Module,
    recurse: bool,
    unwrapped_params: int,
    layer_cls: Set[str],
) -> bool:
    """A convenient auto wrap policy for fx models. If the submodule
    is an instance of layer_cls, the submodule will be wrapped
    as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
    by the outermost FSDP unit.
    Return if a module should be wrapped during FSDP auto wrapping.
    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module):
            The module to be considered in this decision.
        recurse (bool):
            Indicate if this is called to make a decision on whether we
            should recurse down a subgraph of the module structure.
            If False, it means this function is called to make a decision
            on whether we should wrap the said module.
        unwrapped_params (int):
            The number of parameters yet to be wrapped in this module.
        layer_cls (Set[str]):
            Submodules with one of the `layer_cls` names
            will be wrapped as separated FSDP units
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return module.__class__.__name__ in layer_cls


class FullyShardedDataParallel(ParallelModule):
    """Implementation of fully sharded data parallel.

    Args:
        model (torch.nn.Module): The model to enable fully sharded data parallel.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__(model, config)
        self.model = self.fsdp(model, config)

    def _get_underlay_model(self):
        return self.model

    def fsdp(self, model: torch.nn.Module, config: Config):
        if isinstance(model, fx.GraphModule):
            layer_cls = set()
            # Filter out some existing models, such as nn.Linear.
            for name in config.dist.fsdp.wrap_layer_cls:
                cls = utils.get_module_class_from_name(model, name)
                if cls is None:
                    layer_cls.add(name)
            model = split_fsdp_wrap_modules(model, layer_cls)
            auto_wrap_policy = functools.partial(
                fx_auto_wrap_policy,
                layer_cls=config.dist.fsdp.wrap_layer_cls,
            )
        else:
            layer_cls = set()
            for name in config.dist.fsdp.wrap_layer_cls:
                cls = utils.get_module_class_from_name(model, name)
                assert cls, f"class {name} in fsdp.wrap_layer_cls not found in model"
                layer_cls.add(cls)
            if config.is_eager_backend():
                auto_wrap_policy = torch_fsdp.wrap.ModuleWrapPolicy(layer_cls)
            else:
                auto_wrap_policy = functools.partial(
                    xla_fsdp.wrap.transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=layer_cls,
                )

        dtype = torch.float32
        if config.compute.fp16:
            dtype = torch.float16
        if config.compute.bf16:
            dtype = torch.bfloat16

        # (wenting.swt): When using fsdp, disable autocast for precision conversion
        # Instead, use low precision for all intermediate calculations
        # Only the output is float32. This is to align with Stanford Alpaca's fsdp implementation
        if config.compute.fp16 or config.compute.bf16:
            model._original_forward = model.forward
            model_forward_func = model.forward.__func__ if hasattr(
                model.forward, "__func__") else model.forward
            new_forward = torch.cuda.amp.autocast(dtype=dtype)(
                model_forward_func)
            model.forward = MethodType(new_forward, model)
            model.forward = MethodType(
                utils.convert_outputs_to_fp32(model.forward.__func__), model)

        auto_wrapper_callable = None
        if config.memory.gc and (config.memory.gc_cls
                                 == config.dist.fsdp.wrap_layer_cls):
            gc_cnt = config.memory.gc_cnt

            def auto_wrapper_callable(m, *args, **kwargs):
                nonlocal gc_cnt
                if gc_cnt is None:
                    m = checkpoint.checkpoint_module(m)
                elif gc_cnt > 0:
                    m = checkpoint.checkpoint_module(m)
                    gc_cnt -= 1
                return xla_fsdp.XlaFullyShardedDataParallel(m, *args, **kwargs)

        if config.is_eager_backend():
            if config.dist.dp.size > 1:
                process_group = (self.mesh.get_fsdp_proc_group(),
                                 self.mesh.get_dp_proc_group())
                sharding_strategy = torch_fsdp.ShardingStrategy.HYBRID_SHARD
            else:
                process_group = self.mesh.get_fsdp_proc_group()
                sharding_strategy = torch_fsdp.ShardingStrategy.FULL_SHARD
            mixed_precision = torch_fsdp.MixedPrecision(
                param_dtype=dtype,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
            model = torch_fsdp.FullyShardedDataParallel(
                model,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                device_id=torch.cuda.current_device(),
                sync_module_states=config.dist.fsdp.sync_module_states)
        else:
            model = xla_fsdp.XlaFullyShardedDataParallel(
                model,
                flatten_parameters=config.dist.fsdp.flatten_parameters,
                sync_module_states=config.dist.fsdp.sync_module_states,
                opt_flatten_overlap=True,
                pin_layout_in_collective_ops=False,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
                compute_dtype=dtype,
                buffer_dtype=dtype,
                sharding_groups=self.mesh.get_fsdp_rank_groups(),
                sharding_rank=self.mesh.get_fsdp_rank(),
                sharding_world_size=self.mesh.get_fsdp_num())
        return model

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self.model, "clip_grad_norm_"):
            self.model.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
