from enum import Enum
from types import MethodType
from typing import Dict

import torch
from torch.fx.passes.split_module import split_module

import torchacc.utils.trace as trace


# From PiPPy IR.py
def pipe_split():
    if trace._pipeline_tracer is not None and hasattr(trace._pipeline_tracer,
                                                      "graph"):
        trace._pipeline_tracer.graph.call_function(pipe_split, (), {})


class SplitPoint(Enum):
    BEGINNING = 1
    END = 2


class PipeSplitWrapper(torch.nn.Module):

    def __init__(
        self,
        mod: torch.nn.Module,
        split_point: SplitPoint = SplitPoint.BEGINNING,
    ):
        super().__init__()
        self.mod = mod
        self.split_point = split_point

    def forward(self, *args, **kwargs):
        try:
            if self.split_point == SplitPoint.BEGINNING:
                pipe_split()

            return self.mod(*args, **kwargs)
        finally:
            if self.split_point == SplitPoint.END:
                pipe_split()


def add_pipe_split(module, split_type):

    def _forward_with_pipe_split(m, *args, **kwargs):
        try:
            if split_type == SplitPoint.BEGINNING:
                pipe_split()

            return module._torchacc_pp_forward_original(*args, **kwargs)
        finally:
            if split_type == SplitPoint.END:
                pipe_split()

    module._torchacc_pp_forward_original = module.forward
    module.forward = MethodType(_forward_with_pipe_split, module)


def annotate_split_points(mod: torch.nn.Module, spec: Dict[str, SplitPoint]):
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        mod_to_wrap = mod
        predecessor_module = None
        for i, atom in enumerate(atoms):
            try:
                predecessor_module = mod_to_wrap
                mod_to_wrap = getattr(mod_to_wrap, atom)
            except AttributeError as e:
                raise AttributeError(
                    f'Specified target {qualname} referenced nonexistent module {".".join(atoms[:i+1])}'
                )

        # add_pipe_split(mod_to_wrap, split_type)
        wrapped_mod = PipeSplitWrapper(mod_to_wrap, split_type)
        setattr(predecessor_module, atoms[-1], wrapped_mod)


def is_split_node(node):
    return (node.op, node.target) == ("call_function", pipe_split)


# Sequentially propagate the outputs of the submodule that have multiple users.
def _propagate_output(split):
    model_idx = {}
    model_nodes = []
    idx = 0
    for node in split.graph.nodes:
        if trace.is_call_module(node):
            model_idx[node.target] = idx
            model_nodes.append(node)
            idx += 1

    def add_input(mod, node):
        placeholder = mod.graph.placeholder(
            "added_" + node.name,
            type_expr=node.type,
        )
        placeholder.meta = node.meta.copy()
        # update target to prevent any duplication of arguments
        placeholder.target = placeholder.name
        return placeholder

    # add output which is input_idx'th placeholder or add_output
    def add_output(call_mod, mod, input_idx, add_output):
        output_vals = None
        old_output = None
        input_cnt = 0

        for n in mod.graph.nodes:
            if n.op == "placeholder":
                if input_cnt == input_idx:
                    add_output = n
                input_cnt += 1
            elif n.op == "output":
                assert add_output
                if isinstance(n.args[0], tuple):
                    output_vals = n.args[0] + (add_output,)
                else:
                    output_vals = n.args + (add_output,)
                old_output = n
        mod.graph.erase_node(old_output)
        mod.graph.output(output_vals)

        proxy = torch.fx.proxy.Proxy(call_mod)
        # previous output of this call_mod is 1 tensor, so the use of this output:
        #   out = call_mod(args)
        #   out2 = other_mod(out, ...)
        # need change to:
        #   out = call_mod(args)
        #   out_0 = out[0]
        #   out2 = other_mod(out_0, ...)
        if len(output_vals) == 2:
            with split.graph.inserting_after(call_mod):
                prev_output = proxy[0].node
            prev_users = list(call_mod.users)
            for u in prev_users:
                if u != prev_output:
                    u.replace_input_with(call_mod, prev_output)
        return output_vals

    def get_module(name):
        return split.graph.owning_module.get_submodule(name)

    def get_output_node(mod, idx):
        for n in mod.graph.nodes:
            if trace.is_output(n):
                if isinstance(n.args[0], tuple):
                    return n.args[0][idx]
                else:
                    assert idx == 0
                    return n.args[0]
        assert False

    def propagate_single_user(node):
        assert len(node.users) == 1
        user = list(node.users)[0]
        if trace.is_output(user):
            use_idx = len(model_idx)
        elif trace.is_call_module(user):
            use_idx = model_idx[user.target]
        else:
            raise ValueError(f"Unknown user: {user}")
        def_mod = None
        out_idx = 0
        if trace.is_call_module(node):
            def_mod = node.target
            def_idx = model_idx[def_mod]
            out_idx = 0
        elif trace.is_getitem(node):
            def_mod = node.args[0].target
            def_idx = model_idx[def_mod]
            out_idx = node.args[1]
        else:
            raise ValueError(f"Unknown node: {node}")
        if use_idx == def_idx + 1:
            return
        mod = get_module(def_mod)
        out_node = get_output_node(mod, out_idx)
        new_inp = node
        for idx in range(def_idx + 1, use_idx):
            call_mod = model_nodes[idx]
            mod = get_module(call_mod.target)

            placeholder = add_input(mod, out_node)
            output_vals = add_output(call_mod, mod, -1, placeholder)

            # add args
            proxy = torch.fx.proxy.Proxy(call_mod)
            call_mod.args = call_mod.args + (new_inp,)
            with split.graph.inserting_after(list(call_mod.users)[-1]):
                new_inp = proxy[len(output_vals) - 1].node
        user.replace_input_with(node, new_inp)

    def propagate_multiple_users(node):
        users = list(node.users)
        users.sort(key=lambda x: model_idx[x.target])
        new_inp = node
        for use_idx, call_mod in enumerate(users[:-1]):
            mod = get_module(call_mod.target)
            input_idx = call_mod.args.index(new_inp)
            # add output which is input_idx'th placeholder
            output_vals = add_output(call_mod, mod, input_idx, None)

            proxy = torch.fx.proxy.Proxy(call_mod)
            next_user = users[use_idx + 1]
            with split.graph.inserting_after(list(call_mod.users)[-1]):
                new_inp = proxy[len(output_vals) - 1].node
            next_user.replace_input_with(node, new_inp)

    for node in split.graph.nodes:
        if trace.is_getitem(node) and len(node.users) != 1:
            propagate_multiple_users(node)
        elif trace.is_call_module(node) and len(node.users) != 1:
            mod = get_module(node.target)
            output_num = 0
            for n in mod.graph.nodes:
                if trace.is_output(n):
                    if isinstance(n.args[0], tuple):
                        output_num = len(n.args[0])
                    else:
                        output_num = 1
            if output_num == 1:
                propagate_multiple_users(node)

    for node in split.graph.nodes:
        if trace.is_getitem(node):
            assert len(node.users) == 1
            propagate_single_user(node)
        elif trace.is_call_module(node) and len(node.users) == 1:
            propagate_single_user(node)

    for mod in split.children():
        mod.graph.lint()
        mod.recompile()
    split.graph.lint()
    split.recompile()
    return split


def split(traced_model, orig_model, split_points, num_stages):
    split_mod = None
    curr_idx = 0

    def split_callback(n: torch.fx.node.Node):
        nonlocal split_mod, curr_idx
        if "nn_module_stack" in n.meta:
            for mod in n.meta["nn_module_stack"]:
                # split in the beginning
                if mod in split_points:
                    if mod != split_mod:
                        split_mod = mod
                        curr_idx += 1
                    break
        return curr_idx

    # Ask split_module to return mapping from new qualname to old qualname
    qualname_map: Dict[str, str] = {}
    split = split_module(traced_model, orig_model, split_callback, qualname_map)
    # a (custom) tracer can produce dead code like orphan get_attr nodes
    split.graph.eliminate_dead_code()
    for submodule in split.modules():
        if isinstance(submodule, torch.fx.GraphModule):
            for node in submodule.graph.nodes:
                if is_split_node(node):
                    submodule.graph.erase_node(node)
            submodule.recompile()

    trace.move_single_param_to_callee(split, qualname_map)

    # Sequentially propagate the outputs of the submodule that have multiple users.
    _propagate_output(split)
    return split, qualname_map


def create_post_process(m, last_name, last_model):
    graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    attrs: Dict[str, torch.fx.graph_module.GraphModule] = {}
    model_inputs = []
    for node in last_model.graph.nodes:
        if node.op == "output":
            outputs = node.args
            if isinstance(node.args[0], (tuple, list)):
                outputs = node.args[0]
            for i, n in enumerate(outputs):
                name = f"{last_name}_{i}"
                placeholder = graph.placeholder(name, type_expr=n.type)
                placeholder.meta = n.meta.copy()
                model_inputs.append(placeholder)

    def get_node(n):
        if trace.is_getitem(n):
            mod = n.args[0]
            assert mod.op == "call_module" and mod.target == last_name, \
                "The output of the model can only be the output of the last stage."
            return model_inputs[n.args[1]]
        elif trace.is_call_module(n):
            return model_inputs[0]
        else:
            raise ValueError(f"Unknown node: {n}")

    for node in m.graph.nodes:
        if node.op == "output":
            graph.output(torch.fx.graph.map_arg(node.args[0], get_node))
    return torch.fx.graph_module.GraphModule(attrs, graph)


def get_model_input_names(model):
    input_names = []
    for node in model.graph.nodes:
        if node.op == "placeholder":
            input_names.append(node.target)
    return input_names


def get_model_output_names(model):
    output_names = []
    for node in model.graph.nodes:
        if node.op == "output":
            for arg in node.args:
                if not isinstance(arg, (list, tuple)):
                    arg = [arg]
                for a in arg:
                    output_names.append(a.name)
    return output_names


# Note [PP input_tensor_attr]
# The input tensor can be obtained from the dataset, which is represented as a string.
# It can also be represented by the index(int) in the corresponding output tensor list
# from the previous stage.
def get_input_tensor_attr(m, stage_id):
    name, model = list(m.named_children())[stage_id]
    inputs = []
    input_attr = []
    for node in m.graph.nodes:
        if node.op == "call_module" and node.target == name:
            inputs = node.args
    for inp in inputs:
        if inp.op == "placeholder":
            input_attr.append(inp.target)
        elif trace.is_getitem(inp):
            # getitem idx
            input_attr.append(inp.args[1])
        elif trace.is_call_module(inp):
            input_attr.append(0)
        else:
            raise ValueError(f"Unknown input {inp}")
    return input_attr
