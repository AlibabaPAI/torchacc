import inspect
import os
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Union

import torch

import torchacc.ops as ops
from torchacc.core import amp
from torchacc.utils.import_utils import is_torch_xla_available
from torchacc.utils.logger import logger

if is_torch_xla_available():
    from torch_xla.amp import syncfree


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


def patch_optim_step(backend):
    '''
    replace the optimizer step with the hybridtrace enabled step.
    '''
    if hasattr(torch.optim.Optimizer.__init__, "_orig"):
        return
    orign_init_fn = torch.optim.Optimizer.__init__

    def _init_fn(self, params: Union[Iterable[torch.Tensor],
                                     Iterable[Dict[str, Any]]],
                 defaults: Dict[str, Any]) -> None:
        if "lr" in defaults and isinstance(defaults["lr"], (float, int)):
            defaults["lr"] = torch.tensor(defaults["lr"])
        orign_init_fn(self, params, defaults)

    torch.optim.Optimizer.__init__ = _patch_functions(
        torch.optim.Optimizer.__init__, _init_fn)
    torch.optim.SGD.step = torch.compile(torch.optim.SGD.step, backend=backend)
    torch.optim.Adam.step = torch.compile(
        torch.optim.Adam.step, backend=backend)
    torch.optim.AdamW.step = torch.compile(
        torch.optim.AdamW.step, backend=backend)


def patch_transformers_fa():
    '''
    Replace `transformers.modeling_flash_attention_utils._flash_attention_forward` with
    `torchacc.ops.flash_attn_xla` and `torchacc.ops.flash_attn_varlen_xla`
    '''
    if os.getenv('TORCHACC_PATCH_FA', '1') not in ['1', 'true', 'True']:
        return

    try:
        import transformers
        from packaging import version
        version_ts = transformers.__version__
        if version.parse(version_ts) >= version.parse(
                "4.43.0") and version.parse(version_ts) <= version.parse(
                    "4.46.3"):
            from typing import Optional

            import transformers.modeling_flash_attention_utils as modeling_flash_attention_utils

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
                use_sliding_windows = (
                    sliding_window is not None and
                    key_states.shape[1] > sliding_window)
                window_size = (sliding_window,
                               sliding_window) if use_sliding_windows else (-1,
                                                                            -1)
                if query_states.shape[0] == 1 and position_ids is not None:
                    return ops.flash_attn_varlen_position_ids_xla(
                        query_states,
                        key_states,
                        value_states,
                        position_ids,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        window_size=(sliding_window, sliding_window)
                        if sliding_window is not None else (-1, -1),
                        deterministic=deterministic,
                        causal=is_causal)
                elif attention_mask is not None:
                    return ops.flash_attn_varlen_xla(
                        query_states,
                        key_states,
                        value_states,
                        attention_mask=attention_mask.contiguous(),
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=is_causal,
                        window_size=window_size,
                        deterministic=deterministic)
                else:
                    return ops.flash_attn_xla(
                        query_states,
                        key_states,
                        value_states,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=is_causal,
                        window_size=window_size,
                        deterministic=deterministic)

            modeling_flash_attention_utils._flash_attention_forward = _choose_functions(
                modeling_flash_attention_utils._flash_attention_forward,
                _flash_attention_forward)
        elif version.parse(version_ts) > version.parse("4.46.3"):
            from typing import Optional

            import transformers.modeling_flash_attention_utils as modeling_flash_attention_utils

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
                cu_seq_lens_q: Optional[torch.LongTensor] = None,
                cu_seq_lens_k: Optional[torch.LongTensor] = None,
                max_length_q: Optional[int] = None,
                max_length_k: Optional[int] = None,
                target_dtype: Optional[torch.dtype] = None,
            ):
                use_sliding_windows = (
                    sliding_window is not None and
                    key_states.shape[1] > sliding_window)
                window_size = (sliding_window,
                               sliding_window) if use_sliding_windows else (-1,
                                                                            -1)
                if query_states.shape[0] == 1 and position_ids is not None:
                    return ops.flash_attn_varlen_position_ids_xla(
                        query_states,
                        key_states,
                        value_states,
                        position_ids,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        window_size=(sliding_window, sliding_window)
                        if sliding_window is not None else (-1, -1),
                        deterministic=deterministic,
                        causal=is_causal)
                elif attention_mask is not None:
                    return ops.flash_attn_varlen_xla(
                        query_states.contiguous(),
                        key_states.contiguous(),
                        value_states.contiguous(),
                        attention_mask=attention_mask.contiguous(),
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        window_size=(sliding_window, sliding_window)
                        if sliding_window is not None else (-1, -1),
                        deterministic=deterministic,
                        causal=is_causal)
                else:
                    return ops.flash_attn_xla(
                        query_states,
                        key_states,
                        value_states,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        window_size=(sliding_window, sliding_window)
                        if sliding_window is not None else (-1, -1),
                        deterministic=deterministic,
                        causal=is_causal)

            modeling_flash_attention_utils._flash_attention_forward = _choose_functions(
                modeling_flash_attention_utils._flash_attention_forward,
                _flash_attention_forward)
        else:
            logger.warn(
                f'FlashAttention is not successfully patched with transformers version={version_ts}.'
            )
    except Exception as e:
        logger.warn(
            f'torchacc will not patch any FlashAttention function due to {e}.')


def patch_ta_fa():
    try:
        import flash_attn
        if hasattr(flash_attn.flash_attn_func, '_orig'):
            return
        flash_attn.flash_attn_func = _choose_functions(
            flash_attn.flash_attn_func, ops.flash_attn_xla)
    except ImportError:
        logger.warn(f"Patch flash_attn.flash_attn_func failed.")


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


def patch_qwen(use_flash_attn):
    '''
    Modify the calculation of `rotary_seq_len` in `Qwen2FlashAttention2.forward` to avoid xla graph be executed.
    Replace `transformers.models.qwen.modeling_qwen2.Qwen2Model._update_causal_mask` with `return None`
    and replace flash_attn with the interface in torchacc. This requires transformers>=4.41.0.
    '''
    import inspect

    import transformers
    from packaging import version

    if use_flash_attn:
        from transformers.cache_utils import Cache
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model

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

        Qwen2Model._update_causal_mask = update_causal_mask

    if version.parse(transformers.__version__) >= version.parse("4.37.0"):
        try:
            import re

            import transformers.models.qwen2.modeling_qwen2 as qwen2

            src = inspect.getsource(qwen2.Qwen2FlashAttention2)

            pattern1 = r"rotary_seq_len\s*=\s*\(\s*max\(kv_seq_len,\s*position_ids\[:,\s*-1\]\.max\(\)\.item\(\)\s*\+\s*1\)\s*if\s*position_ids\s*is\s*not\s*None\s*else\s*kv_seq_len\s*\)"
            pattern2 = r"rotary_seq_len\s*=\s*max\(kv_seq_len,\s*position_ids\[:,\s*-1\]\.max\(\)\.item\(\)\)\s*\+\s*1"
            replacement = "rotary_seq_len = kv_seq_len"

            src = re.sub(pattern1, replacement, src)
            src = re.sub(pattern2, replacement, src)
            dict_src = \
            """\nQWEN2_ATTENTION_CLASSES = {
            "eager": Qwen2Attention,
            "flash_attention_2": Qwen2FlashAttention2,
            "sdpa": Qwen2SdpaAttention,
            }
            """
            src = src + dict_src
            exec(src, qwen2.__dict__)
        except Exception as e:
            logger.warning(f"patch qwen2 failed due to: {e}")


def patch_autocast(target_device='cuda'):
    if os.getenv('TORCHACC_PATCH_TORCH_AUTOCAST', '1') in [
            '1', 'true', 'True'
    ] and not hasattr(torch.autocast.__init__, '_orig'):
        original_init = torch.autocast.__init__

        def patched_init(self,
                         device_type: str,
                         dtype: Optional[torch.dtype] = None,
                         enabled: bool = True,
                         cache_enabled: Optional[bool] = None):
            if target_device == 'cuda' and device_type == 'xla':
                device_type = 'cuda'
            elif target_device == 'xla' and device_type == 'cuda':
                device_type = 'xla'
            original_init(
                self,
                device_type,
                dtype=dtype,
                enabled=enabled,
                cache_enabled=cache_enabled)

        torch.autocast.__init__ = _patch_functions(torch.autocast.__init__,
                                                   patched_init)
