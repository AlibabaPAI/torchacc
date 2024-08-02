import inspect
from functools import wraps

import torch
from torch_xla.amp import syncfree

import torchacc.ops as ops
from torchacc.core import amp
from torchacc.utils.logger import logger


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
    Replace `transformers.modeling_flash_attention_utils._flash_attention_forward` with
    `torchacc.ops.flash_attn_xla` and `torchacc.ops.flash_attn_varlen_xla`
    '''
    try:
        import transformers
        version = transformers.__version__
        if version >= "4.44.2":
            import transformers.modeling_flash_attention_utils as modeling_flash_attention_utils
            from typing import Optional

            def _flash_attention_forward(
                query_states: torch.Tensor,
                key_states: torch.Tensor,
                value_states: torch.Tensor,
                attention_mask: torch.Tensor,
                query_length: int,
                is_causal: bool,
                dropout: float = 0.0,
                position_ids: Optional[torch.Tensor] = None,
                softmax_scale: Optional[float] = None,
                sliding_window: Optional[int] = None,
                use_top_left_mask: bool = False,
                softcap: Optional[float] = None,
                deterministic: bool = None,
            ):
                if attention_mask is not None:
                    return ops.flash_attn_varlen_xla(
                        query_states.contiguous(),
                        key_states.contiguous(),
                        value_states.contiguous(),
                        attention_mask=attention_mask.contiguous(),
                        dropout_p=dropout,
                        softmax_scale=softmax_scale)
                else:
                    return ops.flash_attn_xla(
                        query_states,
                        key_states,
                        value_states,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale)

            modeling_flash_attention_utils._flash_attention_forward = _flash_attention_forward
        else:
            logger.warn(
                f'FlashAttention is not successfully patched with transformers version={version},'
                ' try to patch flash_attn.flash_attn_func')
            try:
                import flash_attn
                if hasattr(flash_attn.flash_attn_func, '__orig'):
                    return
                flash_attn.flash_attn_func = _choose_functions(
                    flash_attn.flash_attn_func, ops.flash_attn_xla)
            except ImportError:
                logger.warn(f"Patch flash_attn.flash_attn_func failed.")
    except:
        logger.warn('torchacc will not patch any FlashAttention function.')


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
            if attention_mask is not None:
                return attention_mask
            return None

        LlamaModel._update_causal_mask = update_causal_mask
