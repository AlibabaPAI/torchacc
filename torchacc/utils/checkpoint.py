import functools
from typing import Dict, Optional, Set

import torch
import torch.fx as fx

from torch.fx.passes.split_module import split_module

from torchacc.utils.utils import get_module_class_from_name

from torchacc.utils.import_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla
    import torch_xla.distributed.fsdp.wrap as xla_wrap
    checkpoint_module = torch_xla.distributed.fsdp.checkpoint_module


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

    auto_wrap_policy = functools.partial(
        xla_wrap.transformer_auto_wrap_policy, transformer_layer_cls=cls)
    auto_wrapper_callable = lambda m, *args, **kwargs: checkpoint_module(m)
    model, n_params = xla_wrap.recursive_wrap(model, auto_wrap_policy,
                                              auto_wrapper_callable, (), ())
    return model
