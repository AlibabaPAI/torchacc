import einops
import torch

from .flash_attn import flash_attn_varlen_xla


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=None) -> torch.Tensor:
    # TODO: shape check and scale support
    bsz = query.shape[0]
    q_len = query.shape[-2]
    query = einops.rearrange(query, "b h s ... -> (b s) h ...")
    key = einops.rearrange(key, "b h s ... -> (b s) h ...")
    value = einops.rearrange(value, "b h s ... -> (b s) h ...")
    cu_lens = torch.arange(
        0, (bsz + 1) * q_len,
        step=q_len,
        dtype=torch.int32,
        device=query.device)
    max_s = q_len
    is_causal = attn_mask is not None or is_causal
    output = flash_attn_varlen_xla(
        query,
        key,
        value,
        cu_lens,
        cu_lens,
        max_s,
        max_s,
        dropout_p,
        softmax_scale=None,
        causal=is_causal)
    output = einops.rearrange(output, "(b s) h ... -> b h s ...", b=bsz)
    return output
