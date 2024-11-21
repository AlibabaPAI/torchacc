import einops
import torch

from .flash_attn import flash_attn_xla


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attn_mask=None,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=None) -> torch.Tensor:
    # TODO: shape check and scale support
    bsz = query.shape[0]

    is_causal = attn_mask is not None or is_causal
    output = flash_attn_xla(
        query, key, value, dropout_p, softmax_scale=None, causal=is_causal)
    return output
