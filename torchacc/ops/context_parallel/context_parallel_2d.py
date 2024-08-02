from typing import Optional

import torch
import torch.distributed as dist

from .ring_attn import ring_attention
from .ulysses import ulysses
from .utils import diff_all_to_all


def context_parallel_2d(
    q: torch.Tensor,  # [batch_size, seqlen, nheads, headdim]
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
    inter_process_group: Optional[dist.ProcessGroup] = None,
    intra_process_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Implementation of 2D context parallel.

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
            **(-alibi_slope * |i + seqlen_k - seqlen_q - j|)**
            is added to the attention score of query i and key j.
        deterministic (bool): Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
           (they might not have the right scaling).
        inter_process_group (torch.distributed.ProcessGroup, optional): The process group used
            to communicate by splitting the seqlen dimension. It is recommended to use inter-node
            communication with low bandwidth.
        intra_process_group (torch.distributed.ProcessGroup, optional): The process group used
            to communicate by splitting the nheads dimension. It is recommended to use intra-node
            communication with high bandwidth. Note that inter_process_group and intra_process_group
            should not both be None. The group size of inter_process_group and intra_process_group
            should not both be 1.

    Returns:
        out (torch.Tensor): [batch_size, seqlen, nheads, headdim].
    """
    if inter_process_group is None and intra_process_group is None:
        raise ValueError(
            "inter_process_group and intra_process_group should not both be None."
        )

    if dist.get_world_size(inter_process_group) == 1 and dist.get_world_size(
            intra_process_group) == 1:
        raise ValueError(
            "The group size of inter_process_group and intra_process_group should not both be 1."
        )

    if dist.get_world_size(inter_process_group) == 1:
        inter_process_group = None
    if dist.get_world_size(intra_process_group) == 1:
        intra_process_group = None

    if intra_process_group is not None and inter_process_group is not None:
        q = diff_all_to_all(
            q, scatter_dim=2, gather_dim=1, group=intra_process_group)
        k = diff_all_to_all(
            k, scatter_dim=2, gather_dim=1, group=intra_process_group)
        v = diff_all_to_all(
            v, scatter_dim=2, gather_dim=1, group=intra_process_group)

        out = ring_attention(
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
            deterministic=deterministic,
            process_group=inter_process_group)

        out = diff_all_to_all(
            out, scatter_dim=1, gather_dim=2, group=intra_process_group)
    elif intra_process_group is not None:
        out = ulysses(
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
            deterministic=deterministic,
            process_group=intra_process_group)
    else:
        out = ring_attention(
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
            deterministic=deterministic,
            process_group=inter_process_group)
    return out
