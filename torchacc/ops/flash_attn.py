import einops
import torch
import torch_xla
import torch_xla.distributed.spmd as xs


class FlashAttnVarlenQKVPackedXla(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        qkv,
        attention_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1]**(-0.5)

        assert isinstance(window_size, tuple) and len(window_size) == 2
        assert attention_mask is not None

        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        softmax_lse, out, rng_state, cu_seqlens_q, cu_seqlens_k = torch_xla._XLAC._flash_attention_forward(
            q, k, v, attention_mask, alibi_slopes, dropout_p, softmax_scale,
            False, causal, window_size[0], window_size[1], return_softmax, None)
        out = out.to(qkv.dtype)

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q,
                              cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.dropout_p, ctx.softmax_scale, False,
            ctx.causal, ctx.window_size[0], ctx.window_size[1],
            ctx.deterministic, None, rng_state)

        dqkv = torch.stack([dq, dk, dv], dim=1)
        return dqkv, None, None, None, None, None, None, None, None, None


class SPMDFlashAttnVarlenXla(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                attention_mask,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                return_softmax,
                mesh=None,
                partition_spec=None):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)
        assert isinstance(window_size, tuple) and len(window_size) == 2

        ctx.partition_spec = partition_spec
        ctx.mesh = mesh
        ctx.q_full_shape = None
        ctx.k_full_shape = None  # for GQA

        full_q = q
        full_k = k
        full_v = v
        if partition_spec is not None:
            ctx.q_full_shape = q.shape
            ctx.k_full_shape = k.shape
            q = xs.enable_manual_sharding(
                q, partition_spec, mesh=mesh).global_tensor
            k = xs.enable_manual_sharding(
                k, partition_spec, mesh=mesh).global_tensor
            v = xs.enable_manual_sharding(
                v, partition_spec, mesh=mesh).global_tensor

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        with torch.no_grad():
            softmax_lse, out, rng_state, cu_seqlens_q, cu_seqlens_k = torch_xla._XLAC._flash_attention_forward(
                q, k, v, attention_mask, alibi_slopes, dropout_p, softmax_scale,
                False, causal, window_size[0], window_size[1], return_softmax,
                None)

        if partition_spec is not None:
            out = xs.disable_manual_sharding(
                out, partition_spec, ctx.q_full_shape, mesh=mesh).global_tensor

        out = out.to(q.dtype)

        ctx.save_for_backward(full_q, full_k, full_v, out, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors

        partition_spec = ctx.partition_spec
        mesh = ctx.mesh

        if partition_spec is not None:
            q = xs.enable_manual_sharding(
                q, partition_spec, mesh=mesh).global_tensor
            k = xs.enable_manual_sharding(
                k, partition_spec, mesh=mesh).global_tensor
            v = xs.enable_manual_sharding(
                v, partition_spec, mesh=mesh).global_tensor
            dout = xs.enable_manual_sharding(
                dout, partition_spec, mesh=mesh).global_tensor
            out = xs.enable_manual_sharding(
                out, partition_spec, mesh=mesh).global_tensor

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.dropout_p, ctx.softmax_scale, False,
            ctx.causal, ctx.window_size[0], ctx.window_size[1],
            ctx.deterministic, None, rng_state)

        if partition_spec is not None:
            dq = xs.disable_manual_sharding(
                dq, partition_spec, ctx.q_full_shape, mesh=mesh).global_tensor
            dk = xs.disable_manual_sharding(
                dk, partition_spec, ctx.k_full_shape, mesh=mesh).global_tensor
            dv = xs.disable_manual_sharding(
                dv, partition_spec, ctx.k_full_shape, mesh=mesh).global_tensor

        dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenXla(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, attention_mask, dropout_p, softmax_scale, causal,
                window_size, alibi_slopes, deterministic, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)
        assert isinstance(window_size, tuple) and len(window_size) == 2

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        softmax_lse, out, rng_state, cu_seqlens_q, cu_seqlens_k = torch_xla._XLAC._flash_attention_forward(
            q, k, v, attention_mask, alibi_slopes, dropout_p, softmax_scale,
            False, causal, window_size[0], window_size[1], return_softmax, None)
        out = out.to(q.dtype)

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q,
                              cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.dropout_p, ctx.softmax_scale, False,
            ctx.causal, ctx.window_size[0], ctx.window_size[1],
            ctx.deterministic, None, rng_state)

        dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnXla(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal, window_size,
                alibi_slopes, deterministic, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)
        assert isinstance(window_size, tuple) and len(window_size) == 2

        bsz, q_len, head_size, _ = q.size()

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        softmax_lse, out, rng_state = torch_xla._XLAC._flash_attention_forward(
            q, k, v, None, alibi_slopes, dropout_p, softmax_scale, False,
            causal, window_size[0], window_size[1], return_softmax, None)
        out = out.to(q.dtype)

        ctx.save_for_backward(q, k, v, out, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bsz = bsz
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors

        maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
        dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]

        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, None, None, ctx.alibi_slopes,
            ctx.dropout_p, ctx.softmax_scale, False, ctx.causal,
            ctx.window_size[0], ctx.window_size[1], ctx.deterministic, None,
            rng_state)

        dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_qkvpacked_xla(
    qkv,
    attention_mask,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    assert qkv.dtype in [torch.bfloat16, torch.float16
                        ], 'flash attention only supports fp16/bf16'
    return FlashAttnVarlenQKVPackedXla.apply(
        qkv,
        attention_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def spmd_flash_attn_varlen_xla(
    q,
    k,
    v,
    attention_mask,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    mesh=None,
    partition_spec=None,
):
    assert q.dtype in [torch.bfloat16,
                       torch.float16], 'flash attention only supports fp16/bf16'
    return SPMDFlashAttnVarlenXla.apply(
        q,
        k,
        v,
        attention_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        mesh,
        partition_spec,
    )


def flash_attn_varlen_xla(
    q,
    k,
    v,
    attention_mask,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim), where each query sequence is paded to seqlen_q.
        k: (batch_size, seqlen_k, nheads_k, headdim), where each key sequence is paded to seqlen_k.
        v: (batch_size, seqlen_k, nheads_k, headdim), where each value sequence is paded to seqlen_k.
        attention_mask: (batch_size, seqlen_k), each position is either 0 or 1, with every row starting
            with 1s followed by 0s. The count of 1s represents the real sequence length.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert q.dtype in [torch.bfloat16,
                       torch.float16], 'flash attention only supports fp16/bf16'
    if attention_mask.dtype != torch.int32:
        attention_mask.to(torch.in32)
    return FlashAttnVarlenXla.apply(
        q,
        k,
        v,
        attention_mask,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash_attn_xla(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    assert q.dtype in [torch.bfloat16,
                       torch.float16], 'flash attention only supports fp16/bf16'
    return FlashAttnXla.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )
