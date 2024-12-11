# This file is largely inspired by and partially follows the structure of
# https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/ring_flash_attn_varlen.py

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (_flash_attn_varlen_backward,
                                             _flash_attn_varlen_forward)

import torchacc as ta
from torchacc.utils.import_utils import is_torch_xla_available

from .utils import (RingComm, flatten_varlen_lse, unflatten_varlen_lse,
                    update_out_and_lse)

if is_torch_xla_available():
    import torch_xla


def ring_flash_attn_varlen_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        true_k_lens,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    comm = RingComm(process_group)

    out = None
    lse = None
    kv = torch.stack([k, v], dim=0)
    next_kv = None

    rng_states = [None] * comm.world_size

    for step in range(comm.world_size):
        cur_kv_rank = (comm.world_size + comm.rank - step) % comm.world_size
        cur_k_len = true_k_lens[cur_kv_rank]

        if step + 1 != comm.world_size:
            next_kv: torch.Tensor = comm.send_recv(kv)
            comm.commit()

        if (not causal or step <= comm.rank) and torch.any(cur_k_len > 0):
            k = kv[0].contiguous()
            v = kv[1].contiguous()

            nheads, headdim = k.shape[2], k.shape[3]
            unpadded_k = torch.cat([
                u[:v] if v > 0 else u.new_zeros((1, nheads, headdim))
                for u, v in zip(k, cur_k_len)
            ])
            unpadded_v = torch.cat([
                u[:v] if v > 0 else u.new_zeros((1, nheads, headdim))
                for u, v in zip(v, cur_k_len)
            ])

            # if there is seq_len=0 in current batch, pad it with zero to make seq_len=1.
            cur_k_len = cur_k_len.clamp(min=1)
            cu_seqlens_k = torch.cat([cur_k_len.new_zeros([1]),
                                      cur_k_len]).cumsum(
                                          0, dtype=torch.int32).to(
                                              k.device, non_blocking=True)

            if ta.is_lazy_tensor(q):
                block_lse, block_out, rng_states[
                    cur_kv_rank] = torch_xla._XLAC._flash_attention_forward(
                        q,
                        unpadded_k,
                        unpadded_v,
                        cu_seqlens_q.to(q.device, non_blocking=True),
                        cu_seqlens_k,
                        alibi_slopes,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p,
                        softmax_scale,
                        False,
                        causal,
                        window_size[0],
                        window_size[1],
                        False,
                        None,
                    )
            else:
                block_out, _, _, _, _, block_lse, _, rng_states[
                    cur_kv_rank] = _flash_attn_varlen_forward(
                        q,
                        unpadded_k,
                        unpadded_v,
                        cu_seqlens_q.to(q.device, non_blocking=True),
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        dropout_p,
                        softmax_scale,
                        causal=causal and step == 0,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        return_softmax=True and dropout_p > 0,
                    )
            block_lse = flatten_varlen_lse(
                block_lse,
                cu_seqlens=cu_seqlens_q,
            )
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            kv = next_kv

    out = out.to(q.dtype)
    # Note: The clone here is important, which avoids the correctness issues associated with
    # inplace operators.
    lse = unflatten_varlen_lse(lse, cu_seqlens_q, max_seqlen_q).clone()
    return out, lse, next_kv[0], next_kv[1], rng_states


def ring_flash_attn_varlen_backward(
        process_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        true_k_lens,
        rng_states,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    kv_comm = RingComm(process_group, reverse=True)
    dq = None

    if not ta.is_lazy_tensor(q):
        block_dq_buffer = torch.zeros_like(q)
        block_dk_buffer = torch.zeros_like(k)
        block_dv_buffer = torch.zeros_like(v)
    dk = torch.zeros_like(k).flatten(0, 1)
    dv = torch.zeros_like(v).flatten(0, 1)

    next_dk, next_dv = None, None
    next_k, next_v = None, None

    for step in range(kv_comm.world_size):
        cur_kv_rank = (kv_comm.rank + step + 1) % kv_comm.world_size
        cur_k_len = true_k_lens[cur_kv_rank]
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
        if step != 0:
            next_dk = kv_comm.send_recv(dk)
            next_dv = kv_comm.send_recv(dv)
        kv_comm.commit()

        if (step <= kv_comm.rank or not causal) and torch.any(cur_k_len > 0):
            bwd_causal = causal and step == 0

            nheads, headdim = k.shape[2], k.shape[3]
            unpadded_k = torch.cat([
                u[:v] if v > 0 else u.new_zeros((1, nheads, headdim))
                for u, v in zip(k, cur_k_len)
            ])
            unpadded_v = torch.cat([
                u[:v] if v > 0 else u.new_zeros((1, nheads, headdim))
                for u, v in zip(v, cur_k_len)
            ])

            # if there is seq_len=0 in current batch, pad it with zero to make seq_len=1.
            cur_k_len = cur_k_len.clamp(min=1)
            cu_seqlens_k = torch.cat([cur_k_len.new_zeros([1]),
                                      cur_k_len]).cumsum(
                                          0, dtype=torch.int32)

            if ta.is_lazy_tensor(q):
                block_dq_buffer, unpadded_dk, unpadded_dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
                    dout,
                    q,
                    unpadded_k,
                    unpadded_v,
                    out,
                    softmax_lse,
                    cu_seqlens_q.to(q.device, non_blocking=True),
                    cu_seqlens_k.to(k.device, non_blocking=True),
                    alibi_slopes,
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p,
                    softmax_scale,
                    False,
                    bwd_causal,
                    window_size[0],
                    window_size[1],
                    deterministic,
                    None,
                    rng_states[cur_kv_rank],
                )
            else:
                unpadded_dk = torch.cat(
                    [u[:v] for u, v in zip(block_dk_buffer, cur_k_len)])
                unpadded_dv = torch.cat(
                    [u[:v] for u, v in zip(block_dv_buffer, cur_k_len)])
                _flash_attn_varlen_backward(
                    dout,
                    q,
                    unpadded_k,
                    unpadded_v,
                    out,
                    softmax_lse,
                    block_dq_buffer,
                    unpadded_dk,
                    unpadded_dv,
                    cu_seqlens_q.to(q.device, non_blocking=True),
                    cu_seqlens_k.to(k.device, non_blocking=True),
                    max_seqlen_q,
                    max_seqlen_k,
                    dropout_p,
                    softmax_scale,
                    bwd_causal,
                    window_size,
                    alibi_slopes,
                    deterministic,
                    rng_states[cur_kv_rank],
                )

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk[:cu_seqlens_k[-1], :, :] = unpadded_dk.to(torch.float32)
                dv[:cu_seqlens_k[-1], :, :] = unpadded_dv.to(torch.float32)
            else:
                padded_dk = F.pad(
                    unpadded_dk,
                    (0, 0, 0, 0, 0, dk.shape[0] - cu_seqlens_k[-1]))
                padded_dv = F.pad(
                    unpadded_dv,
                    (0, 0, 0, 0, 0, dv.shape[0] - cu_seqlens_k[-1]))
                dq += block_dq_buffer
                kv_comm.wait()
                dk = padded_dk + next_dk
                dv = padded_dv + next_dv
        elif step != 0:
            kv_comm.wait()
            dk = next_dk
            dv = next_dv

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k = next_k
            v = next_v

    dk = dk.unflatten(0, (-1, max_seqlen_k))
    dv = dv.unflatten(0, (-1, max_seqlen_k))
    return dq.to(torch.bfloat16), dk.to(q.dtype), dv.to(q.dtype)


class RingFlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        true_k_lens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse, k, v, rng_states = ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            true_k_lens,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q,
                              *rng_states)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.true_k_lens = true_k_lens
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        # Here, the saved tensors are extracted in advance to prevent
        # errors caused by repeated unpacking during offloading.
        saved_tensors = ctx.saved_tensors
        q, k, v, out, softmax_lse, cu_seqlens_q = saved_tensors[:6]
        rng_states = saved_tensors[6:]
        dq, dk, dv = ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.true_k_lens,
            rng_states=rng_states,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def ring_flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens_q,
    max_seqlen_q,
    max_seqlen_k,
    true_k_lens,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnVarlenFunc.apply(
        qkv[:, 0],
        qkv[:, 1],
        qkv[:, 2],
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        true_k_lens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    max_seqlen_q,
    max_seqlen_k,
    true_k_lens,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return RingFlashAttnVarlenFunc.apply(
        q,
        kv[:, 0],
        kv[:, 1],
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        true_k_lens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def ring_attention(
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
    process_group: Optional[dist.ProcessGroup] = None,
):
    """Implementation of ring attention.

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
            to communicate by splitting the seqlen dimension.

    Returns:
        out (torch.Tensor): [batch_size, seqlen, nheads, headdim].
    """
    assert isinstance(window_size, tuple) and len(window_size) == 2
    world_size = dist.get_world_size(process_group)
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    if q_lens is not None:
        assert torch.all(
            q_lens == lq * world_size
        ), "Currently, only the non-varlen version of q is supported."
        q_lens = q_lens // world_size
    else:
        q_lens = torch.tensor([lq] * b, dtype=torch.int32)
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32)
    q = q.flatten(0, 1)

    if k_lens is None:
        true_k_lens = [torch.tensor([lk] * b, dtype=torch.int32)] * world_size
    else:
        # true k_lens in each rank.
        true_k_lens = [k_lens - i * lk for i in range(world_size)]
        true_k_lens = [u.clamp(max=lk, min=0) for u in true_k_lens]
    return RingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        lq,
        lk,
        true_k_lens,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        False,
        process_group,
    ).unflatten(0, (b, lq))
