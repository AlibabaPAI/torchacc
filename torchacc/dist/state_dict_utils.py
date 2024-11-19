import copy
import os
import pickle
import threading
import re
from collections import OrderedDict
from glob import glob
from typing import Dict, Any, NamedTuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils._pytree import tree_map_only
import torch_xla.core.xla_model as xm

def _numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel


def unflatten_params(params, param_names, param_shapes, param_numels):
    if params.dim() == 0:
        full_params = [params for _ in range(len(param_names))]
    else:
        full_params = [
            t.view(s)
            for (t, s) in zip(params.split(param_numels), param_shapes)
        ]

    return full_params


def unpad(params, layer_numel, world_size):
    if params.dim() == 0:
        return params
    numel = 0
    for layer_numel in layer_numel:
        numel += layer_numel
    if numel % world_size != 0:
        pad_size = world_size - numel % world_size
        params = params[:-pad_size]
    return params


def get_layer_full_info(shard_metadata, model_state_dict):
    """
    Get full name, shape and numel info of unflatten and unshard model's state_dict according 
    to shard_metadata and shard model's state_dict;
    Args:
        shard_metadata (dict):
            ``model.get_shard_metadata()`` from an FSDP model of any rank
        model_state_dict(dict):
            The FSDP model's state_dict.

    Returns:
        layer_name_list(list): 2-dimension list([[layer_name_group1], [layer_name_group2], ...]), contains the full name information.
        if parameters if flatten, each layer may have mutiple orig name and parameter.
        layer_size_list(list): 2-dimension list([[layer_size_group1], [layer_size_group2], ...]), contains the unflatten and unshard shape information of 
        each layer.
        layer_numel_list(list): 2-dimension list([[layer_numel_group1], [layer_numel_group2], ...]), contains the unflatten and unshard numel information of 
        each layer. 
        sharded_list(list): 1-dimension list, contains whether the layer params is sharded.
    """
    layer_name_list = []
    layer_size_list = []
    layer_numel_list = []
    sharded_list = []
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

        sharded_list.append(is_sharded)
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

        return (layer_name_list_, layer_size_list_, layer_numel_list_,
                sharded_list)

    # return with lists
    layer_name_list = [[
        fn.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")
    ] for fn in layer_name_list]
    layer_size_list = [[s] for s in layer_size_list]
    layer_numel_list = [[n] for n in layer_numel_list]

    return (layer_name_list, layer_size_list, layer_numel_list, sharded_list)

def all_gather_state(state_params, sharding_groups, all_gather_op):
    if state_params.dim() == 0:
        return state_params

    tensor_buffer = all_gather_op(state_params, groups=sharding_groups)

    return tensor_buffer

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


def load_checkpoints(ckpt_dir, ckpt_name):
    """
    Load checkpoints that match the pattern of `ckpt_dir + ckpt_name`.
    We use multiple thread to accelerate the loading progress, each thread
    load one shard checkpoint.
    """
    ckpt_path_pattern = os.path.join(ckpt_dir, ckpt_name)
    ckpt_paths = glob(ckpt_path_pattern)
    checkpoints_and_paths = [[] for _ in range(len(ckpt_paths))]

    def load_ckpt(path, idx):
        ckpt = torch.load(path, map_location="cpu")
        checkpoints_and_paths[idx] = (ckpt, path)

    threads = []

    for idx, path in enumerate(ckpt_paths):
        thread = threading.Thread(target=load_ckpt, args=(path, idx))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    checkpoints_and_paths.sort(key=lambda c: c[0]["shard_metadata"]["rank"])
    checkpoints = [c[0] for c in checkpoints_and_paths]
    for rank, (ckpt, path) in enumerate(checkpoints_and_paths):
        assert ckpt["shard_metadata"]["world_size"] == len(checkpoints), (
            f'Expecting {ckpt["shard_metadata"]["world_size"]} files '
            f"(based on metadata in {path}) but got {len(checkpoints)} files. "
            f"Please check if you have missing or unexpected files in {ckpt_path_pattern}."
        )
        assert ckpt["shard_metadata"]["rank"] == rank, (
            f'Expecting rank {ckpt["shard_metadata"]["rank"]} for {path} but it is '
            f"ranked {rank} (out of {len(checkpoints)} files). "
            f"Please check if you have missing or unexpected files in {ckpt_path_pattern}."
        )

        return checkpoints


def save_checkpoints(state_dict_list, shard_metadata_list, save_paths,
                     save_type):
    """
    Save checkpoints to save_paths.
    We use multiple thread to accelerate the saving progress, each thread
    save one shard checkpoint.
    """
    # save consolidate checkpoint
    if not isinstance(state_dict_list, list):
        torch.save(state_dict_list, save_paths)
        return

    def save_checkpoint(state_dict, shard_metadata, save_path, save_type):
        model = {
            f"{save_type}": state_dict,
            "shard_metadata": shard_metadata,
        }

        torch.save(model, save_path)

    threads = []
    # save reshard checkpoints
    for state_dict, shard_metadata, save_path in zip(state_dict_list,
                                                     shard_metadata_list,
                                                     save_paths):
        thread = threading.Thread(
            target=save_checkpoint,
            args=(state_dict, shard_metadata, save_path, save_type))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def consolidate_sharded_model_checkpoints(ckpt_dir, checkpoints):
    """
    Consolidate the sharded FSDP checkpoints into a single model checkpoint.
    Release the tensor in sharded FSDP checkpoints immediately to save memory.
    """
    state_dict_list = [ckpt["model"] for ckpt in checkpoints]
    shard_metadata = checkpoints[0]["shard_metadata"]
    layer_name_list, layer_size_list, layer_numel_list, sharded_list = get_layer_full_info(
        shard_metadata, state_dict_list[0])

    file_path = os.path.join(ckpt_dir, "layer_info.pickle")
    with open(file_path, 'wb') as f:
        pickle.dump(
            [layer_name_list, layer_size_list, layer_numel_list, sharded_list],
            f)

    full_state_dict = OrderedDict()

    # consolidate and unflatten per layer
    for idx, (state_name,
              state_params) in enumerate(state_dict_list[0].items()):
        layer_name = layer_name_list[idx]
        layer_size = layer_size_list[idx]
        layer_numel = layer_numel_list[idx]
        is_sharded = sharded_list[idx]

        consolidate_params = state_params
        if is_sharded:
            p_shard_list = []
            for state_dict in state_dict_list:
                p_shard_list.append(state_dict[state_name])
                state_dict[state_name] = None

            consolidate_params = torch.cat(p_shard_list, dim=0)
        consolidate_params = unpad(consolidate_params, layer_numel,
                                   shard_metadata['world_size'] *
                                   128)  # world_size * _shard_size_multiple

        orig_params = unflatten_params(consolidate_params, layer_name,
                                       layer_size, layer_numel)

        for fn, fp in zip(layer_name, orig_params):
            full_state_dict[fn] = fp

    return full_state_dict


def consolidate_sharded_optimizer_checkpoints(ckpt_dir, checkpoints,
                                              layer_info):
    """
    Consolidate the sharded FSDP checkpoints into a single optimizer checkpoint.
    Release the tensor in sharded FSDP checkpoints immediately to save memory.
    """
    optim_state_dict_list = [ckpt['optimizer'] for ckpt in checkpoints]
    shard_metadata = checkpoints[0]["shard_metadata"]

    layer_name_list, layer_size_list, layer_numel_list, sharded_list = layer_info
    flatten_name_list = [fn for layer_fn in layer_name_list for fn in layer_fn]

    full_optim_state_dict: Dict[str, Any] = {'state': {}, 'param_groups': {}}

    full_optim_state_dict['param_groups'] = copy.deepcopy(
        optim_state_dict_list[0]['param_groups'])
    full_optim_state_dict['param_groups'][0]['params'].clear()

    for fn in flatten_name_list:
        full_optim_state_dict['param_groups'][0]['params'].append(fn)

    unflat_state_dict = {fn: {} for fn in flatten_name_list}

    # consolidate and unflatten per layer per state
    for idx, layer_state in enumerate(
            optim_state_dict_list[0]['state'].values()):
        layer_name = layer_name_list[idx]
        layer_size = layer_size_list[idx]
        layer_numel = layer_numel_list[idx]
        is_sharded = sharded_list[idx]

        for state_name, state_param in layer_state.items():
            consolidate_params = state_param
            if is_sharded and state_param.dim() != 0:
                p_shard_list = []
                for optim_state_dict in optim_state_dict_list:
                    p_shard_list.append(
                        optim_state_dict['state'][idx][state_name])
                    optim_state_dict['state'][idx][state_name] = None

                consolidate_params = torch.cat(p_shard_list, dim=0)
            consolidate_params = unpad(consolidate_params, layer_numel,
                                       shard_metadata['world_size'] *
                                       128)  # world_size * _shard_size_multiple
            orig_params = unflatten_params(consolidate_params, layer_name,
                                           layer_size, layer_numel)

            for fn, fp in zip(layer_name, orig_params):
                unflat_state_dict[fn][state_name] = fp
    full_optim_state_dict['state'] = unflat_state_dict

    return full_optim_state_dict


def _get_shard(tensor, shard_num):
    """
    Return the shard tensor list of a full flatten tensor.
    """
    if tensor.numel() % (shard_num *
                         128) != 0:  # world_size * _shard_size_multiple
        pad_size = (shard_num * 128) - tensor.numel() % (shard_num * 128)
        tensor = F.pad(tensor, [0, pad_size])

    local_size = tensor.size(0) // shard_num
    tensor_list = []
    for i in range(shard_num):
        begin = i * local_size
        end = (i + 1) * local_size
        tensor_list.append(tensor[begin:end].clone())

    return tensor_list


def flatten_tensor_list(param_list):
    if len(param_list) == 0:
        return param_list

    flat_tensors = [torch.flatten(param) for param in param_list]

    return torch.cat(flat_tensors, dim=0)


def reshard_model_dict(consolidate_model_dict, shard_model, layer_name_lists,
                       reshard_num):
    """
    reshard the consolidate model into shard_model_state_dict_list according to the reshard_num.
    Release tensor in consolidate_model_dict immediately to save tensor.
    Return the shard_model_state_dict_list and shard_metadata_list.
    """
    shard_model_state_dict: Dict[str, Any] = {}
    shard_model_state_dict_list = [
        copy.deepcopy(shard_model_state_dict) for _ in range(reshard_num)
    ]

    # flatten and shard tensor per layer
    for (shard_model_name, layer_names) in zip(shard_model['model'].keys(),
                                               layer_name_lists):
        tensor_buffer_list = []
        for name in layer_names:
            tensor_buffer = consolidate_model_dict[name]
            tensor_buffer_list.append(tensor_buffer)
            consolidate_model_dict[name] = None
        flat_tensor = flatten_tensor_list(tensor_buffer_list)
        shard_tensor_list = _get_shard(flat_tensor, reshard_num)

        for shard_tensor, shard_model_dict in zip(shard_tensor_list,
                                                  shard_model_state_dict_list):
            shard_model_dict[shard_model_name] = shard_tensor

    # get shardmeta_list
    shard_metadata_list = []
    for idx in range(reshard_num):
        shard_meta_data = copy.deepcopy(shard_model["shard_metadata"])
        shard_meta_data['world_size'] = reshard_num
        shard_meta_data['rank'] = idx
        shard_metadata_list.append(shard_meta_data)

    return shard_model_state_dict_list, shard_metadata_list


def reshard_optim_dict(consolidate_optim_dict, shard_optim, layer_name_lists,
                       reshard_num):
    """
    reshard the consolidate optim into shard_optim_state_dict_list according to the reshard_num.
    Release tensor in consolidate_optim_dict immediately to save tensor.
    Return the shard_optim_state_dict_list and shard_metadata_list.
    """
    consolidate_optim_state = consolidate_optim_dict['state']

    shard_optim_state_dict: Dict[str, Any] = {'state': {}, 'param_groups': {}}
    shard_optim_state_dict_list = [
        copy.deepcopy(shard_optim_state_dict) for _ in range(reshard_num)
    ]

    # flatten and shard tensor per layer per state_name
    for idx, layer_names in enumerate(layer_name_lists):
        shard_value: Dict[str, Any] = {}
        shard_value_list = [
            copy.deepcopy(shard_value) for _ in range(reshard_num)
        ]
        for state_name in consolidate_optim_state[layer_names[0]].keys():
            tensor_buffer_list = []
            # we need the params of a whole layer state to be flatten and shard
            for name in layer_names:
                state_params = consolidate_optim_state[name][state_name]
                consolidate_optim_state[name][state_name] = None

                # state name 'step'
                if isinstance(state_params,
                              torch.Tensor) and state_params.dim() == 0:
                    for shard_value in shard_value_list:
                        shard_value[state_name] = state_params
                    break

                tensor_buffer_list.append(state_params)

            flat_tensor = flatten_tensor_list(tensor_buffer_list)

            if state_params.dim() != 0:
                shard_tensor_list = _get_shard(flat_tensor, reshard_num)
                for (shard_value, shard_tensor) in zip(shard_value_list,
                                                       shard_tensor_list):
                    shard_value[state_name] = shard_tensor

        for (shard_value,
             shard_optim_state_dict) in zip(shard_value_list,
                                            shard_optim_state_dict_list):
            shard_optim_state_dict['state'][idx] = shard_value

    shard_metadata_list = []

    # get the param_group of optim_state_dict and shard_meta_lists
    for (idx, shard_optim_state_dict) in enumerate(shard_optim_state_dict_list):
        shard_optim_state_dict['param_groups'] = shard_optim['optimizer'][
            'param_groups']

        shard_meta_data = copy.deepcopy(shard_optim["shard_metadata"])
        shard_meta_data['world_size'] = reshard_num
        shard_meta_data['rank'] = idx
        shard_metadata_list.append(shard_meta_data)

    return shard_optim_state_dict_list, shard_metadata_list


def consolidate_and_reshard_fsdp_model_dict(ckpt_dir,
                                            model_ckpt_name_pattern,
                                            save_dir="",
                                            model_save_name_pattern="",
                                            reshard_num=1,
                                            save_model=True):
    """
    Consolidate the sharded FSDP checkpoints into a single model checkpoint. Then
    reshard the FSDP model according to the reshard_num.

    Args:
        ckpt_dir (str):
            The dir to all FSDP shard model checkpoint files.
        model_ckpt_name_pattern (str):
            The name_pattern to all FSDP shard model checkpoint files. Files matching the
            pattern ``ckpt_dir + ckpt_name`` will be loaded. Each
            checkpoint file is assumed to be a dict with a "model" key
            containing the FSDP model's ``model.state_dict()`` and a
            "shard_metadata" key containing the FSDP model's
            ``model.get_shard_metadata()``.
        save_dir (str):
            The save dir for consolidate or reshard model checkpoints.
        model_save_name_pattern (str, Optional):
            The name_pattern for consolidate or reshard model checkpoints.
            For reshard checkpoints name pattern: ``rank*-of-*-model.pth``
            The final save_path is save_dir + save_name.
        reshard_num (int, Optional):
            Reshard the fsdp model by reshard_num. If set to 1, we don't need to do
            resharding.
        save_model (str, Optional):
            if ``True``, the model checkpoint will be saved to ``save_dir + save_name``.

    Returns:
        model_state_dict: the consolidated model state dict or reshard model state dict list.
        shard_meta_list: the reshard metadatalist. For consolidated model, return None.
    """
    checkpoints = load_checkpoints(ckpt_dir, model_ckpt_name_pattern)
    full_state_dict = consolidate_sharded_model_checkpoints(
        ckpt_dir, checkpoints)

    if reshard_num == 1:
        if save_model:
            if not save_dir or not model_save_name_pattern:
                raise ValueError(
                    "save_dir and model_save_name_pattern should not be None!")
            actual_save_path = os.path.join(save_dir, model_save_name_pattern)

            save_checkpoints(full_state_dict, checkpoints[0]['shard_metadata'],
                             actual_save_path, 'model')

        return full_state_dict, None

    # load layer_info
    file_path = os.path.join(ckpt_dir, "layer_info.pickle")
    layer_info = []
    try:
        with open(file_path, 'rb') as f:
            layer_info = pickle.load(f)
    except FileNotFoundError:
        raise NotImplementedError("please consolidate model first!")

    model_state_dict_list, shard_metadata_list = reshard_model_dict(
        full_state_dict, checkpoints[0], layer_info[0], reshard_num)

    if save_model:
        if not save_dir or not model_save_name_pattern:
            raise ValueError(
                "save_dir and save_name_pattern should not be None!")

        actual_save_path = []
        for idx in range(reshard_num):
            # replace the two '*'
            save_name_temp = model_save_name_pattern.replace('*', str(idx), 1)
            save_name_temp = save_name_temp.replace('*', str(reshard_num), 1)
            actual_save_path.append(os.path.join(save_dir, save_name_temp))

        save_checkpoints(model_state_dict_list, shard_metadata_list,
                         actual_save_path, 'model')

    return model_state_dict_list, shard_metadata_list


def consolidate_and_reshard_fsdp_optim_dict(ckpt_dir,
                                            optimizer_ckpt_name_pattern,
                                            save_dir="",
                                            optimizer_save_name_pattern="",
                                            reshard_num=1,
                                            save_optimizer=True):
    """
    Consolidate the sharded FSDP checkpoints into a single optimizer checkpoint. Then
    reshard the FSDP optimizer according to the reshard_num.
    Args:
        ckpt_dir (str):
            The dir to all FSDP shard optimizer checkpoint files.
        optimizer_ckpt_name_pattern (str):
            The name_pattern to all FSDP shard optimizer checkpoint files. Files matching the
            pattern ``ckpt_dir + ckpt_name`` will be loaded. Each
            checkpoint file is assumed to be a dict with a "optimizer" key
            containing the FSDP optimizer's ``optimizer.state_dict()`` and a
            "shard_metadata" key containing the FSDP model's
            ``model.get_shard_metadata()``.
        save_dir (str, Optional):
            The save dir for consolidate or reshard optimizer checkpoints.
        optimizer_save_name_pattern (str, Optional):
            The name_pattern for consolidate or reshard optimizer checkpoints.
            For reshard checkpoints name pattern:: `rank*-of-*-optimizer.pth`
            The final save_path is save_dir + save_name.
        reshard_num (int, Optional):
            Reshard the fsdp optimizer by reshard_num. If set to 1, we don't need to do
            resharding.
        save_model (str, Optional):
            if ``True``, the model checkpoint will be saved to ``save_dir + save_name``.
    Returns:
        optim_state_dict: the consolidated optim state dict or reshard optim state dict list
        shard_meta_list: the reshard metadatalist. For consolidated optim, return None.
    """
    # load checkpoints
    checkpoints = load_checkpoints(ckpt_dir, optimizer_ckpt_name_pattern)

    # load layer_info
    file_path = os.path.join(ckpt_dir, "layer_info.pickle")
    layer_info = []
    try:
        with open(file_path, 'rb') as f:
            layer_info = pickle.load(f)
    except FileNotFoundError:
        print(f"please consolidate model first!")

    full_optim_state_dict = consolidate_sharded_optimizer_checkpoints(
        ckpt_dir, checkpoints, layer_info)

    if reshard_num == 1:
        if save_optimizer:
            if not save_dir or not optimizer_save_name_pattern:
                raise ValueError(
                    "save_dir and optimizer_save_name_pattern should not be None!"
                )
            actual_save_path = os.path.join(save_dir,
                                            optimizer_save_name_pattern)

            save_checkpoints(full_optim_state_dict,
                             checkpoints[0]['shard_metadata'], actual_save_path,
                             'optimizer')

        return full_optim_state_dict, None

    optim_state_dict_list, shard_metadata_list = reshard_optim_dict(
        full_optim_state_dict, checkpoints[0], layer_info[0], reshard_num)

    if save_optimizer:
        if not save_dir or not optimizer_save_name_pattern:
            raise ValueError("save_dir and save_name should not be None!")

        actual_save_path = []
        for idx in range(reshard_num):
            # replace the two '*'
            save_name_temp = optimizer_save_name_pattern.replace(
                '*', str(idx), 1)
            save_name_temp = save_name_temp.replace('*', str(reshard_num), 1)
            actual_save_path.append(os.path.join(save_dir, save_name_temp))

        save_checkpoints(optim_state_dict_list, shard_metadata_list,
                         actual_save_path, 'optimizer')

    return optim_state_dict_list, shard_metadata_list


def consolidate_and_reshard_fsdp_checkpoint(ckpt_dir,
                                            model_ckpt_name_pattern,
                                            optimizer_ckpt_name_pattern,
                                            save_dir="",
                                            model_save_name_pattern="",
                                            optimizer_save_name_pattern="",
                                            reshard_num=1,
                                            save_checkpoint=True):
    """
    Consolidate the sharded FSDP model and optimizer checkpoints into a single checkpoint. Then
    reshard the FSDP checkpoint according to the reshard_num.
    """
    consolidate_and_reshard_fsdp_model_dict(ckpt_dir, model_ckpt_name_pattern,
                                            save_dir, model_save_name_pattern,
                                            reshard_num, save_checkpoint)
    consolidate_and_reshard_fsdp_optim_dict(ckpt_dir,
                                            optimizer_ckpt_name_pattern,
                                            save_dir,
                                            optimizer_save_name_pattern,
                                            reshard_num, save_checkpoint)
