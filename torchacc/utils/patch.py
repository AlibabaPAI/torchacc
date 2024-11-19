import inspect
from functools import wraps

import torch
from torch_xla.amp import syncfree

import torchacc.ops as ops
from torchacc.core import amp


def _patch_functions(fn, newfn):
    xfingerprint = inspect.signature(fn)
    fingerprint = inspect.signature(newfn)
    if xfingerprint != fingerprint:
        raise RuntimeError(
            'Unable to patch {}, signature mismatch: {} vs {}'.format(
                fn, xfingerprint, fingerprint))
    newfn._orig = fn
    return newfn


def _choose_functions(fn, new_fn):
    func = _patch_functions(fn, new_fn)

    @wraps(func)
    def wrapper(*args, **kwargs):

        def check_tensors(arg):
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                return True
            elif isinstance(arg, (list, tuple)):
                return any(check_tensors(item) for item in arg)
            return False

        has_cuda_tensor = any(check_tensors(arg) for arg in args) or \
                            any(check_tensors(kwarg) for kwarg in kwargs.values())

        if has_cuda_tensor:
            return func._orig(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


def patch_amp():
    '''
    replace the optimizer and grad scaler with the syncfree optimizer and the torchacc grad scaler.
    '''
    torch.optim.SGD = _patch_functions(torch.optim.SGD, syncfree.SGD)
    torch.optim.Adam = _patch_functions(torch.optim.Adam, syncfree.Adam)
    torch.optim.AdamW = _patch_functions(torch.optim.AdamW, syncfree.AdamW)
    torch.cuda.amp.GradScaler = amp.GradScaler


def patch_fa():
    '''
    Replace `flash_attn.flash_attn_func`, `flash_attn.flash_attn_varlen_func` with
    `torchacc.ops.flash_attn_xla` and `torchacc.ops.flash_attn_varlen_xla`,
    and dynamically determine which one to use at runtime based on the input device.
    '''
    from .logger import logger
    try:
        import flash_attn
        if hasattr(flash_attn.flash_attn_func, '__orig'):
            return
        flash_attn.flash_attn_func = _choose_functions(
            flash_attn.flash_attn_func, ops.flash_attn_xla)
        flash_attn.flash_attn_varlen_func = _choose_functions(
            flash_attn.flash_attn_varlen_func, ops.flash_attn_varlen_xla)
    except ImportError:
        logger.warning(f"Patch flash_attn failed.")


def patch_llama(use_flash_attn):
    '''
    Replace `transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask` with
    `return None` and replace flash_attn with the interface in torchacc.
    This requires transformers>=4.41.0.
    '''
    if use_flash_attn:
        from transformers.cache_utils import Cache
        from transformers.models.llama.modeling_llama import LlamaModel

        def update_causal_mask(
            self,
            attention_mask: torch.Tensor,
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool,
        ):
            return None

        LlamaModel._update_causal_mask = update_causal_mask

def patch_qwen():
    '''
    Modify the calculation of `rotary_seq_len` in `Qwen2FlashAttention2.forward` to avoid xla graph be executed
    '''
    import inspect
    import transformers
    from .logger import logger
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("4.37.0"):
        try:
            import transformers.models.qwen2.modeling_qwen2 as qwen2
            import re
            
            src = inspect.getsource(qwen2.Qwen2FlashAttention2)

            pattern1 = r"rotary_seq_len\s*=\s*\(\s*max\(kv_seq_len,\s*position_ids\[:,\s*-1\]\.max\(\)\.item\(\)\s*\+\s*1\)\s*if\s*position_ids\s*is\s*not\s*None\s*else\s*kv_seq_len\s*\)"
            pattern2 = r"rotary_seq_len\s*=\s*max\(kv_seq_len,\s*position_ids\[:,\s*-1\]\.max\(\)\.item\(\)\)\s*\+\s*1"
            replacement = "rotary_seq_len = kv_seq_len"

            src = re.sub(pattern1, replacement, src)
            src = re.sub(pattern2, replacement, src)
            dict_src = \
            """
    QWEN2_ATTENTION_CLASSES = {
        "eager": Qwen2Attention,
        "flash_attention_2": Qwen2FlashAttention2,
        "sdpa": Qwen2SdpaAttention,
    }
            """
            src = src + dict_src
            exec(src, qwen2.__dict__) 
        except Exception as e:
            logger.warning(f"patch qwen2 failed due to: {e}")