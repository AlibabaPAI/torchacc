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
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        if attention_mask is not None:
            return ops.flash_attn_varlen_xla(query_states.contiguous(), key_states.contiguous(), value_states.contiguous(), attention_mask=attention_mask.contiguous(), dropout_p=dropout, softmax_scale=softmax_scale)
        else:
            return ops.flash_attn_xla(query_states, key_states, value_states, dropout_p=dropout, softmax_scale=softmax_scale)

    LlamaFlashAttention2._flash_attention_forward = _flash_attention_forward


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
