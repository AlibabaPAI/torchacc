from typing import Optional

import torch
import torch.distributed as dist

from .utils import diff_all_to_all, flash_attention


def ulysses(
    q: torch.Tensor,  # [bsz, seqlen, numhead, headdim]
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    alibi_slopes: Optional[tuple] = None,
    deterministic: bool = False,
    process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Implementation of DeepSpeed-Ulysses.

    Args:
        q (torch.Tensor): [batch_size, seqlen, nheads, headdim]
        k (torch.Tensor): [batch_size, seqlen, nheads_k, headdim]
        v (torch.Tensor): [batch_size, seqlen, nheads_k, headdim]
        q_lens (torch.Tensor, optional): [batch_size]. The original sequence length of q
            without padding and division of context parallel.
        k_lens (torch.Tensor, optional): [batch_size]. The original sequence length of k and v
            without padding and division of context parallel.
        dropout_p (float): Dropout probability.
        softmax_scale (float, optional): The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal (bool): Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size (tuple): (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes (tuple, optional): (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic (bool): Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
           (they might not have the right scaling).
        process_group (torch.distributed.ProcessGroup, optional): The process group used
            to communicate by splitting the nheads dimension.
    Returns:
        out (torch.Tensor): [batch_size, seqlen, nheads, headdim].
    """
    if q.shape[2] % dist.get_world_size(process_group) != 0:
        raise ValueError(f"The nheads {q.shape[2]} needs to be divisible by the size" \
                         f" of process group {dist.get_world_size(process_group)}.")

    q = diff_all_to_all(q, scatter_dim=2, gather_dim=1, group=process_group)
    k = diff_all_to_all(k, scatter_dim=2, gather_dim=1, group=process_group)
    v = diff_all_to_all(v, scatter_dim=2, gather_dim=1, group=process_group)

    out = flash_attention(
        q,
        k,
        v,
        q_lens,
        k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic)

    out = diff_all_to_all(out, scatter_dim=1, gather_dim=2, group=process_group)

    return out
