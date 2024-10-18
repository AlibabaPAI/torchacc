import torch
import torch.distributed as dist
from typing import Any, Dict, NamedTuple, Optional
from torch.utils._pytree import tree_map_only
import torch_xla.core.xla_model as xm


def _numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel


def get_layer_full_info(shard_metadata, model_state_dict):
    """
    Get full name, shape and numel info of unflatten and unshard optimizer's state_dict according 
    to shard_metadata and model's state_dict;
    Args:
        shard_metadata (dict):
            ``model.get_shard_metadata()`` from an FSDP model of any rank
        model_state_dict(dict):
            The state_dict from an FSDP model.

    Returns:
        For all ranks, we get the same shard_metadata and model_state_dict, and the return value is
        same:
        layer_name_list(list): 2-dimension list([[layer_name_group1], [layer_name_group2], ...]), contains the full name information.
        if parameters if flatten, each layer may have mutiple orig name and parameter.
        layer_size_list(list): 2-dimension list([[layer_size_group1], [layer_size_group2], ...]), contains the unflatten and unshard shape information of 
        each layer.
        layer_numel_list(list): 2-dimension list([[layer_numel_group1], [layer_numel_group2], ...]), contains the unflatten and unshard numel information of 
        each layer. 
    """
    layer_name_list = []
    layer_size_list = []
    layer_numel_list = []
    buffer_info = shard_metadata.get("buffer_info", {})

    # consolidate the sharded parameters
    for name, param in model_state_dict.items():
        if name in buffer_info:  # cast buffer back to its original dtype
            p = p.to(buffer_info[name]["_orig_dtype"])

        is_sharded = False
        name_splits = name.split(".")
        model_num = 0
        # if start with 'model', we just skip the model
        for name in name_splits:
            if name != 'model':
                break
            else:
                model_num = model_num + 1
        name_splits = name_splits[model_num:]
        name = ".".join(name_splits)

        for idx, sep in enumerate(name_splits):
            if sep.startswith("_fsdp_shard"):
                is_sharded = True
                prefix = ".".join(name_splits[:idx])
                suffix = ".".join(name_splits[idx:])
                break

        if is_sharded:
            p_info = shard_metadata["shard_info"][prefix][suffix]
            orig_name = p_info["_orig_name"]
            orig_size = p_info["_orig_size"]
            full_name = orig_name
            if prefix != "":
                full_name = prefix + "." + orig_name
            layer_name_list.append(full_name)
            layer_size_list.append(orig_size)
            layer_numel_list.append(_numel(orig_size))

        else:
            # unsharded buffers, we don't need the info in shard_metadata
            layer_name_list.append(name)
            layer_size_list.append(param.shape)
            layer_numel_list.append(_numel(param.shape))

    # flatten_parameters = True
    flatten_info = shard_metadata["flatten_info"]
    if flatten_info != {}:
        layer_name_list_ = []
        layer_size_list_ = []
        layer_numel_list_ = []
        for name in layer_name_list:
            if "_fsdp_wrapped_module.flat_param_" in name:
                metadata = flatten_info[name]
                prefix = ".".join(name.split(".")[:-1])
                param_names, param_shapes, param_numel = metadata
                full_names = param_names

                if prefix != "":
                    full_names = [prefix + "." + n for n in full_names]

                full_names = [
                    fn.replace("_fsdp_wrapped_module.",
                               "").replace("_fpw_module.", "")
                    for fn in full_names
                ]

                layer_name_list_.append(full_names)
                layer_size_list_.append(param_shapes)
                layer_numel_list_.append(param_numel)

        return (layer_name_list_, layer_size_list_, layer_numel_list_)

    # return with lists
    layer_name_list = [[
        fn.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")
    ] for fn in layer_name_list]
    layer_size_list = [[s] for s in layer_size_list]
    layer_numel_list = [[n] for n in layer_numel_list]

    return (layer_name_list, layer_size_list, layer_numel_list)


def unpad(tensor_buffer, layer_numels, world_size):
    if tensor_buffer.dim() == 0:
        return tensor_buffer
    numel = 0
    for layer_numel in layer_numels:
        numel += layer_numel
    if numel % world_size != 0:
        pad_size = world_size - numel % world_size
        tensor_buffer = tensor_buffer[:-pad_size]
    return tensor_buffer


def unflatten_optim_params(params, param_names, param_shapes, param_numels):
    if params.dim() == 0:
        full_params = [params for _ in range(len(param_names))]
    else:
        full_params = [
            t.view(s)
            for (t, s) in zip(params.split(param_numels), param_shapes)
        ]

    return full_params


class _PosDimTensorInfo(NamedTuple):
    """
    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """

    shape: torch.Size
    dtype: torch.dtype


def _setup_gloo_distributed(group):
    if not torch.distributed.is_initialized():
        dist.init_process_group()
    pg = dist.new_group(ranks=group, backend="gloo")
    return pg


def _cleanup_gloo_distributed(pg):
    dist.destroy_process_group(pg)


def broadcast_processed_state(optim_state: dict[str, Any], rank,
                              sharding_groups):
    objects: list[Any] = [None]
    if rank == 0:
        objects[0] = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(
                v.shape, v.dtype),  # type: ignore[union-attr]
            optim_state,
        )

    ordinal = xm.get_ordinal()
    world_size = xm.xrt_world_size()
    # global group
    new_group = list(range(int(world_size)))

    # sharding_groups may be None if we use xla fsdp directly
    if sharding_groups is not None:
        # broadcast within each sharding_group
        for group in sharding_groups:
            if ordinal in group:
                new_group = group
                break

    pg_group = _setup_gloo_distributed(new_group)
    # the src is the global rank of each sharding group's rank0
    dist.broadcast_object_list(
        objects, src=dist.get_global_rank(pg_group, 0), group=pg_group)
    _cleanup_gloo_distributed(pg_group)

    if rank == 0:
        return optim_state
    else:
        return objects[0]


def broadcast_state(state_params, device, rank, sharding_groups,
                    collective_broadcast_op):
    if rank == 0 and isinstance(state_params, torch.Tensor):
        tensor_buffer = state_params.to(device)
    else:
        tensor_buffer = torch.zeros(
            state_params.shape, dtype=state_params.dtype, device=device)

    # Since broadcast employs all-reduce, here we only need to ensure that root_ordinal
    # is different from xm.get_ordinal() on the non-root nodes
    root_ordinal = xm.get_ordinal() if rank == 0 else -1

    collective_broadcast_op([tensor_buffer],
                            root_ordinal=root_ordinal,
                            groups=sharding_groups)

    return tensor_buffer


def all_gather_state(state_params, sharding_groups, all_gather_op):
    if state_params.dim() == 0:
        return state_params

    tensor_buffer = all_gather_op(state_params, groups=sharding_groups)

    return tensor_buffer


def flatten_optim_state(param_list):
    if len(param_list) == 0:
        return param_list

    flat_tensors = [torch.flatten(param) for param in param_list]

    return torch.cat(flat_tensors, dim=0)
