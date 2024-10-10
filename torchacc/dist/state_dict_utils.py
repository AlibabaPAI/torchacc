from collections import OrderedDict
from glob import glob
from typing import Dict
import pickle

import torch

def _numel(shape):
    numel = 1
    for d in shape:
        numel *= d
    return numel

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


def unflatten_params(params, param_names, param_shapes, param_numels):
    if params.dim() == 0:
        full_params = [params for _ in range(len(param_names))]
    else:
        full_params = [
            t.view(s)
            for (t, s) in zip(params.split(param_numels), param_shapes)
        ]

    return full_params


def get_layer_full_info(shard_metadata, model_state_dict):
    """
    Get full name, shape and numel info of unflatten and unshard state_dict according 
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

        return (layer_name_list_, layer_size_list_, layer_numel_list_, sharded_list)

    # return with lists
    layer_name_list = [[
        fn.replace("_fsdp_wrapped_module.", "").replace("_fpw_module.", "")
    ] for fn in layer_name_list]
    layer_size_list = [[s] for s in layer_size_list]
    layer_numel_list = [[n] for n in layer_numel_list]

    return (layer_name_list, layer_size_list, layer_numel_list, sharded_list)

def load_checkpoints(ckpt_prefix, ckpt_suffix="*.pth"):
  ckpt_path_pattern = ckpt_prefix + ckpt_suffix
  ckpt_paths = glob(ckpt_path_pattern)

  checkpoints_and_paths = []
  for path in ckpt_paths:
    ckpt = torch.load(path, map_location="cpu")
    checkpoints_and_paths.append((ckpt, path))

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

def consolidate_sharded_model_checkpoints(ckpt_prefix,
                                          ckpt_suffix="*.pth",
                                          save_path="",
                                          save_model=True):
  """
  Consolidate the sharded FSDP checkpoints into a single model checkpoint.

  Args:
      ckpt_prefix (str):
          prefix to FSDP checkpoint files from all ranks
      ckpt_suffix (str, Optional):
          suffix to FSDP checkpoint files from all ranks. Files matching the
          pattern ``ckpt_prefix + ckpt_suffix`` will be loaded. The each
          checkpoint file is assumed to be a dict with a "model" key
          containing the FSDP model's ``model.state_dict()`` and a
          "shard_metadata" key containing the FSDP model's
          ``model.get_shard_metadata()``.
      save_path (str, Optional):
          the save path to the consolidated model checkpoint file (if
          ``save_model`` is ``True``). The checkpoint file is a dict with a
          "model" key containing the consolidated model state dict.
      save_model (str, Optional):
          if ``True``, the consolidated model checkpoint will be saved to
          ``save_path`` (or ``ckpt_prefix + "_consolidated.pth"`` if
          ``save_path`` is empty).

  Returns:
      full_state_dict: the consolidated model state dict
      actual_save_path: the path to the consolidated model checkpoint file
          (``None`` if ``save_model`` is ``False``)
  """
  checkpoints = load_checkpoints(ckpt_prefix, ckpt_suffix)
  state_dict_list = [ckpt["model"] for ckpt in checkpoints]
  shard_metadata = checkpoints[0]["shard_metadata"]
  layer_name_list, layer_size_list, layer_numel_list, sharded_list = get_layer_full_info(shard_metadata, state_dict_list[0])
  file_path = ckpt_prefix + "layer_info.pickle"

  with open(file_path, 'wb') as f:
    pickle.dump([layer_name_list, layer_size_list, layer_numel_list, sharded_list], f)
    
  full_state_dict = OrderedDict()
  
  # consolidate and unflatten
  for idx, (state_name, state_params) in enumerate(state_dict_list[0].items()):
    layer_name = layer_name_list[idx]
    layer_size = layer_size_list[idx]
    layer_numel = layer_numel_list[idx]
    is_sharded = sharded_list[idx]
      
    consolidate_params = state_params
    if is_sharded:
        p_shard_list = []
        for state_dict in state_dict_list:
            p_shard_list.append(state_dict[state_name])
        consolidate_params = torch.cat(p_shard_list, dim=0)
    orig_params = unflatten_params(consolidate_params, layer_name, layer_size, layer_numel)

    for fn, fp in zip(layer_name, orig_params):
        full_state_dict[fn] = fp    
    
  actual_save_path = None
  if save_model:
    actual_save_path = save_path if save_path else ckpt_prefix + "_consolidated.pth"
    torch.save({"model": full_state_dict}, actual_save_path)
    print(f"saved consolidated model to {actual_save_path}")

  return full_state_dict, actual_save_path


def consolidate_sharded_optimizer_checkpoints(ckpt_prefix,
                                              ckpt_suffix="*.pth",
                                              save_path="",
                                              save_model=True):
    '''
        Consolidate the sharded FSDP checkpoints into a single optimizer checkpoint.
        we need first consolidate model checkpoint to reuse the layer_info
    '''
    checkpoints = load_checkpoints(ckpt_prefix, ckpt_suffix)
    optim_state_dict_list = [ckpt['optimizer'] for ckpt in checkpoints]
    shard_metadata = checkpoints[0]["shard_metadata"]
    file_path = ckpt_prefix + "layer_info.pickle"
    layer_info = []
    try:
        with open(file_path, 'rb') as f:
            layer_info = pickle.load(f)
    except FileNotFoundError:
        print(f"please consolidate model first!")

    layer_name_list, layer_size_list, layer_numel_list, sharded_list = layer_info
    flatten_name_list = [
            fn for layer_fn in layer_name_list for fn in layer_fn
        ]

    full_optim_state_dict: Dict[str, Any] = {
            'state': {},
            'param_groups': {}
        }
    
    full_optim_state_dict['param_groups'] = optim_state_dict_list[0]['param_groups']
    
    full_optim_state_dict['param_groups'][0]['params'].clear()
    for fn in flatten_name_list:
        full_optim_state_dict['param_groups'][0][
            'params'].append(fn)
    
    unflat_state_dict = {fn: {} for fn in flatten_name_list}

    for idx, layer_state in enumerate(optim_state_dict_list[0]['state'].values()):
        layer_name = layer_name_list[idx]
        layer_size = layer_size_list[idx]
        layer_numel = layer_numel_list[idx]
        is_sharded = sharded_list[idx]
        
        for state_name, state_param in layer_state.items():
            consolidate_params = state_param
            if is_sharded and state_param.dim() != 0:
                p_shard_list = []
                for optim_state_dict in optim_state_dict_list:
                    p_shard_list.append(optim_state_dict['state'][idx][state_name])
                   
                consolidate_params = torch.cat(p_shard_list, dim=0)
            orig_params = unflatten_params(consolidate_params, layer_name, layer_size, layer_numel)
            
            for fn, fp in zip(layer_name, orig_params):
                unflat_state_dict[fn][state_name] = fp    
    full_optim_state_dict['state'] = unflat_state_dict
    
    actual_save_path = None
    if save_model:
        actual_save_path = save_path if save_path else ckpt_prefix + "_consolidated.pth"
        torch.save({"optimizer": full_state_dict}, actual_save_path)
        print(f"saved consolidated optimizer to {actual_save_path}")

    return full_optim_state_dict, actual_save_path
