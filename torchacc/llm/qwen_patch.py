import importlib
import inspect
import os
import re
import textwrap
import types


def rewrite_forward(instance, func_name):
    """Remove dtype and cuda check in Qwen transformer.h.attn.forward and
    transformer.h.attn.core_attention_flash.forward"""
    # get path of modeling_qwen.py
    module_path = "transformers_modules/Qwen-72B-Chat/modeling_qwen"
    module_path = module_path.replace(os.path.sep, ".")
    module = importlib.import_module(module_path)

    source = inspect.getsource(getattr(instance, func_name))
    # Remove checks in FlashSelfAttention forward that are
    # incompatible with torchacc.
    # remove dtype assert, cause there could be float32 type of weight when we
    # use AMP training.
    new_source = re.sub(
        r"assert all\(\(i\.dtype in \[torch\.float16, torch\.bfloat16\] for i in \(q, k, v\)\)\)",
        r"pass", source)
    # remove cuda device check because it is xla not cuda when we use TorchAcc.
    new_source = re.sub(r"(assert all\(\(i.is_cuda for i in \(q, k, v\)\)\))",
                        r"pass", new_source)
    # remove batch_size > 1 check to avoid introducing dynamic shape.
    new_source = re.sub(r"batch_size > 1", r"False", new_source)

    # Remove dtype and device check in QWenAttention forward.
    new_source = re.sub(r"self\.is_fp32", r"False", new_source)
    new_source = re.sub(r"query\.is_cuda", r"True", new_source)
    # replace flash attention
    new_source = re.sub(r"flash_attn_unpadded_func",
                        r"torchacc.ops.flash_attn_varlen_xla", new_source)
    new_source = textwrap.dedent(new_source)

    # import torchacc
    torchacc = importlib.import_module('torchacc')
    module.__dict__['torchacc'] = torchacc

    # replace forward with modified one.
    exec(new_source, module.__dict__)
    new_function = types.MethodType(module.__dict__[func_name], instance)
    setattr(instance, func_name, new_function)


def patch_qwen_model(model):
    for layer in model.transformer.h:
        rewrite_forward(layer.attn, 'forward')
        rewrite_forward(layer.attn.core_attention_flash, 'forward')
    return model
