import contextlib
from unittest.mock import patch

import torch
from functorch.compile import make_boxed_func
from torch._dynamo import disable, register_backend
from torch._dynamo.utils import counters
from torch._functorch.aot_autograd import aot_module_simplified

import torchacc as ta


def aot_autograd(**kwargs):

    def compiler_fn(gm: torch.fx.GraphModule, example_inputs):
        # Hack to get around circular import problems with aot_eager_decomp_partition
        if callable(kwargs.get("decompositions")):
            kwargs["decompositions"] = kwargs["decompositions"]()

        # NB: dont delete counter increment
        counters["aot_autograd"]["total"] += 1
        use_fallback = False

        if use_fallback:
            ta.utils.logger.debug(
                "Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return gm

        # OK attempt to compile

        def _wrapped_bw_compiler(*args, **kwargs):
            # stop TorchDynamo from trying to compile our generated backwards pass
            return disable(disable(bw_compiler)(*args, **kwargs))

        bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
        if bw_compiler.__name__ == "_wrapped_bw_compiler":
            kwargs["bw_compiler"] = bw_compiler
        else:
            kwargs["bw_compiler"] = _wrapped_bw_compiler
        kwargs["inference_compiler"] = (
            kwargs.get("inference_compiler") or kwargs["fw_compiler"])

        from functorch.compile import nop
        from torch._inductor.debug import enable_aot_logging

        # debug asserts slow down compile time noticeably,
        # So only default them on when the aot_eager backend is used.
        if kwargs.get("fw_compiler", None) == nop:
            patch_config = patch("functorch.compile.config.debug_assert", True)
        else:
            patch_config = contextlib.nullcontext()

        try:
            # NB: NOT cloned!
            # Enable inplace ops in fx graph
            keep_inference_input_mutations = True
            with enable_aot_logging(), patch_config:
                cg = aot_module_simplified(
                    gm,
                    example_inputs,
                    keep_inference_input_mutations=keep_inference_input_mutations,
                    **kwargs)
                counters["aot_autograd"]["ok"] += 1
                return disable(cg)
        except Exception:
            counters["aot_autograd"]["not_ok"] += 1
            raise

    return compiler_fn


def hybridtrace_boxed(model, fake_tensor_inputs):
    return xla_backend_helper(model, fake_tensor_inputs, boxed=True)


def xla_backend_helper(model, fake_tensor_inputs, boxed=False):
    try:
        import torch_xla._dynamo.config as config
        import torch_xla._dynamo.dynamo_bridge as dynamo_bridge
    except ImportError as e:
        raise ImportError(
            "Please follow the instruction in https://torchacc.readthedocs.io/en/stable/install.html to install torch_xla"
        ) from e

    config.use_call_computation = True
    config.skip_input_data_check = True
    config.outside_on_cuda = False
    config.mark_step_after_layer_if_early_sync = True

    compiled_graph = None

    def fwd(*args):
        nonlocal model
        nonlocal compiled_graph
        if compiled_graph is None:
            compiled_graph = dynamo_bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return make_boxed_func(fwd) if boxed else fwd


hybridtrace = aot_autograd(fw_compiler=hybridtrace_boxed,)

register_backend(name="hybridtrace", compiler_fn=hybridtrace)
