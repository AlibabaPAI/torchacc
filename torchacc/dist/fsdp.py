import functools
from types import MethodType
from typing import Any, Dict, Optional, Set
from enum import auto, Enum
import copy

import torch
import torch.distributed.fsdp as torch_fsdp
import torch.fx as fx
from torch.fx.passes.split_module import split_module
import torch_xla.distributed.fsdp as xla_fsdp

import torchacc as ta
from torchacc.config import Config
from torchacc.dist import ParallelModule
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.trace as trace
import torchacc.utils.utils as utils
import torchacc.dist.state_dict_utils as state_dict_utils


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

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
    ):
        """
        Return the optimizer state-dict in its sharded form.
        
        Args:
            model (torch.nn.Module): FSDP model(torchacc or xla) whose parameters were 
            passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for model's
                parameters.

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            fsdp model. Each rank get the sharded optim state added with shard_metadata.
        """
        if not isinstance(
                model, xla_fsdp.XlaFullyShardedDataParallel) and not isinstance(
                    model, FullyShardedDataParallel):
            raise NotImplementedError(
                "The model must be torchacc or xla FSDP model")
        assert isinstance(model,
                          xla_fsdp.XlaFullyShardedDataParallel) or isinstance(
                              model, FullyShardedDataParallel)

        if isinstance(model, FullyShardedDataParallel):
            model = model.model

        optimizer = {
            "optimizer": optim.state_dict(),
            "shard_metadata": model.get_shard_metadata(),
        }

        return optimizer

    @staticmethod
    def full_optim_state_dict(model: torch.nn.Module,
                              optim: torch.optim.Optimizer,
                              rank0_only: bool = True,
                              cpu_offload: bool = True) -> Dict[str, Any]:
        """Return the full optimizer state-dict.

        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in model are mapped back to their unflattened parameters.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications. However, if ``rank0_only=True``, then
            the state dict is only populated on rank 0, and all other ranks
            return an empty :class:`dict`.

        Args:
            model (torch.nn.Module): FSDP model(torchacc or xla FSDP) whose parameters were 
            passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for model 's
                parameters.
            rank0_only (bool): If ``True``, return the populated :class:`dict`
                only on rank 0; if ``False``, return it on all ranks. (Default:
                ``True``)
            cpu_offload(bool): If ``True``, offload the state-dict to cpu. (Default:
            ``True``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` 's original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an :class:`dict` with keys but empty value.
        """
        if not isinstance(
                model, xla_fsdp.XlaFullyShardedDataParallel) and not isinstance(
                    model, FullyShardedDataParallel):
            raise NotImplementedError(
                "The model must be torchacc or xla FSDP model")
        assert isinstance(model,
                          xla_fsdp.XlaFullyShardedDataParallel) or isinstance(
                              model, FullyShardedDataParallel)

        if isinstance(model, FullyShardedDataParallel):
            model = model.model

        shard_meta_data = model.get_shard_metadata()
        sharded_optim_state = optim.state_dict()['state']
        optim_state_param_groups = optim.state_dict()['param_groups']
        # unflattened and consolidated state_dict
        consolidate_optim_state_dict: Dict[str, Any] = {
            'state': {},
            'param_groups': {}
        }

        # param_names(2-dim list), param_shapes(2-dim list), param_numel(2-dim list)
        layer_name_lists, layer_size_lists, layer_numel_lists, _ = state_dict_utils.get_layer_full_info(
            shard_meta_data, model.state_dict())

        # transform 2-dim list name to 1-dim list name
        flatten_name_list = [
            fn for layer_fn in layer_name_lists for fn in layer_fn
        ]
        # (rank0_only and self.model.rank == 0) or (not rank0_only)
        if not rank0_only or model.rank == 0:
            consolidate_optim_state_dict['param_groups'] = copy.deepcopy(
                optim_state_param_groups)
            consolidate_optim_state_dict['param_groups'][0]['params'].clear()
            for fn in flatten_name_list:
                consolidate_optim_state_dict['param_groups'][0][
                    'params'].append(fn)

        unflat_state_dict = {fn: {} for fn in flatten_name_list}

        for idx, layer_state in enumerate(sharded_optim_state.values()):
            layer_names = layer_name_lists[idx]
            layer_shapes = layer_size_lists[idx]
            layer_numels = layer_numel_lists[idx]
            for state_name, state_params in layer_state.items():
                tensor_buffer = state_dict_utils.all_gather_state(
                    state_params, model.sharding_groups, model.all_gather_op)
                tensor_buffer = state_dict_utils.unpad(
                    tensor_buffer, layer_numels,
                    model.world_size * model._shard_size_multiple)
                orig_params = state_dict_utils.unflatten_params(
                    tensor_buffer, layer_names, layer_shapes, layer_numels)

                if not rank0_only or model.rank == 0:
                    for fn, fp in zip(layer_names, orig_params):
                        if cpu_offload:
                            ta.mark_step()  # tensor evaluation
                            unflat_state_dict[fn][state_name] = fp.cpu()
                        else:
                            unflat_state_dict[fn][state_name] = fp
                ta.mark_step()
        consolidate_optim_state_dict['state'] = unflat_state_dict

        return consolidate_optim_state_dict

    @staticmethod
    def load_optim_state_dict(model: torch.nn.Module,
                              optim_state_dict: Dict[str, Any],
                              rank0_only: bool = True) -> Dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the
        optimizer associated with the FSDP model.
        We check whether the optim_state_dict is sharded automatically.
        For shard optim_state_dict, we must set rank0_only to false.
                
        Args:
            model (torch.nn.Module): FSDP model(torchacc or xla) whose parameters were 
            passed into the optimizer whose state_dict is ``optim_state_dict``.
            optim_state_dict (Dict[str, Any]): The optimizer states to be loaded.
            rank0_only: (bool): control whether load state_dict only from
                rank0 at the begining.(Default: ``True``). If set to True,
                nonzero ranks should pass None in.
        
        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            model which is sharded.
        """
        if not isinstance(
                model, xla_fsdp.XlaFullyShardedDataParallel) and not isinstance(
                    model, FullyShardedDataParallel):
            raise NotImplementedError(
                "The model must be torchacc or xla FSDP model")
        assert isinstance(model,
                          xla_fsdp.XlaFullyShardedDataParallel) or isinstance(
                              model, FullyShardedDataParallel)

        if isinstance(model, FullyShardedDataParallel):
            model = model.model

        shard_meta_data = model.get_shard_metadata()

        if optim_state_dict is None:
            if not rank0_only or model.rank == 0:
                raise ValueError('optim_state_dict cannot be None')
            assert rank0_only and model.rank != 0

        # for shard optim_state_dict, we return directly
        if optim_state_dict is not None and 'shard_metadata' in optim_state_dict.keys(
        ):
            if rank0_only is True:
                raise NotImplementedError(
                    "we only support rank0_only = False for loading shard optim_state_dict."
                )
            assert rank0_only is False

            # the world size should not change
            if shard_meta_data['world_size'] != optim_state_dict[
                    'shard_metadata']['world_size']:
                raise NotImplementedError(
                    "the sharded_optim_state_dict is loaded with world_size: "
                    f"{shard_meta_data['world_size']} but stored with: "
                    f"{optim_state_dict['shard_metadata']['world_size']}!")
            assert shard_meta_data['world_size'] == optim_state_dict[
                'shard_metadata']['world_size']
            return optim_state_dict['optimizer']

        unflat_optim_state = optim_state_dict
        flat_optim_state: Dict[str, Any] = {'state': {}, 'param_groups': {}}

        layer_name_lists, layer_size_lists, layer_numel_lists, _ = state_dict_utils.get_layer_full_info(
            shard_meta_data, model.state_dict())

        if rank0_only:
            unflat_optim_state = state_dict_utils.broadcast_processed_state(
                unflat_optim_state, model.rank, model.sharding_groups)
        unflat_state = unflat_optim_state['state']

        flat_optim_state['param_groups'] = copy.deepcopy(
            unflat_optim_state['param_groups'])

        for idx, layer_names in enumerate(layer_name_lists):
            flat_value: Dict[str, Any] = {}
            # broadcast tensor to other ranks per layer per state
            for state_name in unflat_state[layer_names[0]].keys():
                tensor_buffer_list = []
                # we need the params of a whole layer state to be flatten and shard
                for name in layer_names:
                    state_params = unflat_state[name][state_name]
                    # all ranks have same scalar tensor(step) which has been broadcasted in
                    # broadcast_processed_state above
                    if isinstance(state_params,
                                  torch.Tensor) and state_params.dim() == 0:
                        flat_value[state_name] = state_params
                        break
                    tensor_buffer = unflat_state[name][state_name]
                    if rank0_only:
                        tensor_buffer = state_dict_utils.broadcast_state(
                            state_params, model.xla_device, model.rank,
                            model.sharding_groups,
                            model.collective_broadcast_op)
                    tensor_buffer_list.append(tensor_buffer)

                flat_tensor = state_dict_utils.flatten_tensor_list(
                    tensor_buffer_list)

                if len(flat_tensor):
                    flat_value[state_name] = model._get_shard(flat_tensor)
                ta.mark_step()

            flat_optim_state['state'][idx] = flat_value

        flat_optim_state['param_groups'][0]['params'] = [
            i for i in range(0, len(flat_optim_state['state'].keys()))
        ]

        return flat_optim_state
