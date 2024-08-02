import inspect
import operator
from typing import Dict, List

import torch
import torch.fx as fx

_pipeline_tracer = None


def is_hf_model(model):
    try:
        import transformers
    except ImportError:
        return False
    if isinstance(model, transformers.PreTrainedModel):
        return True
    return False


def get_concrete_args(model: torch.nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(
            input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}")

    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names and p.default != p.empty
    }

    return concrete_args


def _hf_fx_trace(model, concrete_args):
    import transformers.utils.fx as hf_fx
    tracer = hf_fx.HFTracer()

    global _pipeline_tracer
    _pipeline_tracer = tracer

    # Tracing.
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = fx.GraphModule(model, traced_graph)

    # From transformers.utils.fx.symbolic_trace
    traced.config = model.config
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced


def _torch_fx_trace(root, concrete_args=None):
    tracer = fx.Tracer()

    global _pipeline_tracer
    _pipeline_tracer = tracer

    graph = tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__
        if isinstance(root, torch.nn.Module) else root.__name__)
    return fx.GraphModule(tracer.root, graph, name)


def trace(model, input_names):
    input_names = list(input_names) if input_names else []
    if is_hf_model(model):
        concrete_args = get_concrete_args(model, input_names)
        return _hf_fx_trace(model, concrete_args=concrete_args)
    else:
        concrete_args = get_concrete_args(model, input_names)
        return _torch_fx_trace(model, concrete_args=concrete_args)


def is_getitem(node):
    return (node.op, node.target) == ("call_function", operator.getitem)


def is_output(node):
    return node.op == "output"


def is_call_module(node):
    return node.op == "call_module"


def move_single_param_to_callee(split, qualname_map):
    # lift single-use parameter fetches into the modules that use them
    def delete_user_reference(node, user, delete_node=True):
        assert len(user.kwargs) == 0
        use_idxs = [i for i, arg in enumerate(user.args) if arg == node]
        assert len(use_idxs) == 1
        args_copy = list(user.args)
        args_copy.pop(use_idxs[0])
        user.args = tuple(args_copy)
        if delete_node:
            node.graph.erase_node(node)

        return use_idxs[0]

    def move_param_to_callee(root, callee_name, param_val, use_idx, is_buffer):
        assert isinstance(param_val, torch.Tensor), (
            f"Expected '{node.target}' to be {torch.Tensor} but got {type(param_val)}."
        )
        callee = root.get_submodule(callee_name)
        new_param_name = f"moved_{node.target.replace('.', '_')}"
        assert not hasattr(
            callee, new_param_name
        ), f"Module {callee_name} already has a parameter named {new_param_name}"
        if is_buffer:
            callee.register_buffer(new_param_name, param_val)
        else:
            setattr(callee, new_param_name, param_val)

        # Update qualname mapping
        # New qualname will have submodule prefix
        new_qualname = f"{callee_name}.{new_param_name}"
        if node.target in qualname_map:
            # Just in case the target name is already in the qualname_map
            # returned by split_module() -- we update the mapping using the
            # new name as a new key
            qualname_map[new_qualname] = qualname_map.pop(node.target)
        else:
            qualname_map[new_qualname] = node.target

        ph_counter = 0
        for sn in callee.graph.nodes:
            if sn.op == "placeholder":
                if ph_counter == use_idx:
                    with callee.graph.inserting_before(sn):
                        get_attr = callee.graph.get_attr(new_param_name)
                        sn.replace_all_uses_with(get_attr)
                        callee.graph.erase_node(sn)
                ph_counter += 1
        callee.graph.lint()
        callee.recompile()

        return get_attr

    to_delete = list()  # a list of nodes for deferral deletion

    for node in split.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 1:
            user = list(node.users)[0]
            assert user.op == "call_module"
            use_idx = delete_user_reference(node, user)

            # Move parameter into submodule and replace PH with a get_attr
            atoms = node.target.split(".")
            mod_itr = split
            for atom in atoms[:-1]:
                mod_itr = getattr(mod_itr, atom)
            param_val = getattr(mod_itr, atoms[-1])
            is_buffer = atoms[-1] in mod_itr._buffers

            move_param_to_callee(split, user.target, param_val, use_idx,
                                 is_buffer)

            to_delete.append((mod_itr, atoms))

    # deferral deletion
    for mod_itr, atoms in to_delete:
        delattr(mod_itr, atoms[-1])

    split.delete_all_unused_submodules()
    split.graph.lint()
    split.recompile()
