import torch
from torch._decomp import global_decomposition_table, register_decomposition
from torch._ops import OpOverload
from torch.library import _impls

# please donot move the instantiation
# the torch.library.Library will delete all the registrations when destructor called
_meta_lib_dont_use_me_use_register_meta = torch.library.Library(
    "aten", "IMPL", "Meta")

if torch._C._has_mkldnn:
    _meta_lib_dont_use_me_use_register_meta_for_mkldnn = torch.library.Library(
        "mkldnn", "IMPL", "Meta")
    if torch._C.has_mkl:
        _meta_lib_dont_use_me_use_register_meta_for_mkl = torch.library.Library(
            "mkl", "IMPL", "Meta")
    _meta_lib_dont_use_me_use_register_meta_for_onednn = torch.library.Library(
        "onednn", "IMPL", "Meta")
    _meta_lib_dont_use_me_use_register_meta_for_quantized = torch.library.Library(
        "quantized", "IMPL", "Meta")


def replace_decompose():
    # get registered inplace decompose op
    inplace_set = []
    for type in ["meta", "post_autograd", "pre_autograd"]:
        for keys in global_decomposition_table[type]:
            op_name = keys.__name__.split('.')[0]
            if op_name[-1] == '_' and keys not in inplace_set:
                inplace_set.append(keys)

    # clear decomposition table
    for key, table in global_decomposition_table.items():
        table.clear()

    # build empty func for decompose inplace op
    for op in inplace_set:

        @register_decomposition(op)
        def empty(*args, **kwargs):
            return args[0]

    # the function come from torch/_meta_registrations.py
    # we add the removal of op in the function in case we can't register it repeatedly
    def activate_meta():
        activate_meta_table = {}

        # For a given op, we pick the most specific decomp function from
        # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
        for type in ["meta", "post_autograd", "pre_autograd"]:
            registry = global_decomposition_table[type]

            for opo in registry:
                if opo not in activate_meta_table:
                    activate_meta_table[opo] = registry[opo]
        for op_overload, fn in activate_meta_table.items():
            # Don't register meta for HigherOrderOp's decomp.
            # We can reconsider this in the future, but in general,
            # the way you do a meta for a HigherOrderOp is different from
            # OpOverload.
            if isinstance(op_overload, torch._ops.HigherOrderOperator):
                continue
            assert isinstance(op_overload, OpOverload)
            # remove from py kernels
            if torch._C.DispatchKey.Meta in op_overload.py_kernels:
                del op_overload.py_kernels[torch._C.DispatchKey.Meta]

            op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

            if torch._C._dispatch_has_kernel_for_dispatch_key(
                    op_overload.name(), "CompositeImplicitAutograd"):
                # Internally, we shouldn't be registering meta kernels for any operators that
                # have CompositeImplicitAutograd kernels.
                # Instead, we should be letting those decompositions run, and writing meta kernels
                # only for the base operators.
                if op_overload in global_decomposition_table["meta"]:
                    raise RuntimeError(
                        f"{op_overload} is a CompositeImplicitAutograd op, we shouldn't "
                        "register meta function for it. Instead, we should let the decomposition run and write "
                        "meta kernels for the base operators.")
                pass
            elif op_overload.is_view:
                # Attempting to register a python meta kernel for a view operator.
                # We shouldn't do this, because the output will report as not having aliased storages.
                # All view ops have meta kernels in C++ today, so we should use those instead.
                pass
            elif op_overload.name() in {
                    "aten::empty_strided",  # causing infinite recursion, test_meta.py
                    "aten::clone",  # causing infinite recursion
                    "aten::_to_copy",  # causing infinite recursion, test_serialization.py -k test_tensor_subclass_getstate_overwrite  # noqa: B950
                    "aten::copy_",  # Exception not raised, test_torch.py -k test_storage_meta_errors_cpu_int64  # noqa: B950
                    "aten::constant_pad_nd",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_amp_istft_cuda_float32  # noqa: B950
                    "aten::rot90",  # requires_grad mismatch! test_ops.py -k test_fake_crossref_backward_amp_rot90_cuda_float32  # noqa: B950
                    "aten::as_strided_scatter",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_no_amp_as_strided_scatter_cuda_float32  # noqa: B950
            }:
                pass
            else:
                # remove from the dispatcher
                if isinstance(op_overload, str):
                    name = op_overload
                elif isinstance(op_overload, OpOverload):
                    name = op_overload._schema.name
                    overload_name = op_overload._schema.overload_name
                    if overload_name != '':
                        name = name + '.' + overload_name

                key = "aten" + "/" + name.split("::")[-1] + "/" + "Meta"
                if key in _impls:
                    _impls.remove(key)

                if "mkldnn::" in op_overload.name():
                    _meta_lib_dont_use_me_use_register_meta_for_mkldnn.impl(
                        op_overload, fn)
                elif "mkl::" in op_overload.name():
                    _meta_lib_dont_use_me_use_register_meta_for_mkl.impl(
                        op_overload, fn)
                elif "onednn::" in op_overload.name():
                    _meta_lib_dont_use_me_use_register_meta_for_onednn.impl(
                        op_overload, fn)
                elif "quantized::" in op_overload.name():
                    _meta_lib_dont_use_me_use_register_meta_for_quantized.impl(
                        op_overload, fn)
                else:
                    # replace decompose op would only go here
                    _meta_lib_dont_use_me_use_register_meta.impl(
                        op_overload, fn)

    activate_meta()
