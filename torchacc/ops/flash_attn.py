import einops
import torch
import torch_xla
import torch_xla.distributed.spmd as xs


class FlashAttnVarlenQKVPackedXla(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
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

        softmax_lse, out, rng_state = torch_xla._XLAC._flash_attention_forward(
            qkv[:, 0], qkv[:, 1], qkv[:, 2], cu_seqlens, cu_seqlens,
            alibi_slopes, max_seqlen, max_seqlen, dropout_p, softmax_scale,
            False, causal, window_size[0], window_size[1], return_softmax, None)
        out = out.to(qkv.dtype)

        ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen
        ctx.max_seqlen_k = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        qkv, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, qkv[:, 0], qkv[:, 1], qkv[:, 2], out, softmax_lse, cu_seqlens,
            cu_seqlens, ctx.alibi_slopes, ctx.max_seqlen_q, ctx.max_seqlen_k,
            ctx.dropout_p, ctx.softmax_scale, False, ctx.causal,
            ctx.window_size[0], ctx.window_size[1], ctx.deterministic, None,
            rng_state)

        dqkv = torch.stack([dq, dk, dv], dim=1)
        return dqkv, None, None, None, None, None, None, None, None, None


class SPMDFlashAttnVarlenXla(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
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

        with torch.no_grad():
            softmax_lse, out, rng_state = torch_xla._XLAC._flash_attention_forward(
                q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, max_seqlen_q,
                max_seqlen_k, dropout_p, softmax_scale, False, causal,
                window_size[0], window_size[1], return_softmax, None)

        if partition_spec is not None:
            out = xs.disable_manual_sharding(
                out, partition_spec, ctx.q_full_shape, mesh=mesh).global_tensor

        out = out.to(q.dtype)

        ctx.save_for_backward(full_q, full_k, full_v, out, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
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

        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p,
            ctx.softmax_scale, False, ctx.causal, ctx.window_size[0],
            ctx.window_size[1], ctx.deterministic, None, rng_state)

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
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                max_seqlen_k, dropout_p, softmax_scale, causal, window_size,
                alibi_slopes, deterministic, return_softmax):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)
        assert isinstance(window_size, tuple) and len(window_size) == 2

        softmax_lse, out, rng_state = torch_xla._XLAC._flash_attention_forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, max_seqlen_q,
            max_seqlen_k, dropout_p, softmax_scale, False, causal,
            window_size[0], window_size[1], return_softmax, None)
        out = out.to(q.dtype)

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q,
                              cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors

        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k,
            ctx.alibi_slopes, ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p,
            ctx.softmax_scale, False, ctx.causal, ctx.window_size[0],
            ctx.window_size[1], ctx.deterministic, None, rng_state)

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
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=q.device)
        q = einops.rearrange(q, "b s ... -> (b s) ...")
        k = einops.rearrange(k, "b s ... -> (b s) ...")
        v = einops.rearrange(v, "b s ... -> (b s) ...")

        softmax_lse, out, rng_state = torch_xla._XLAC._flash_attention_forward(
            q, k, v, cu_q_lens, cu_q_lens, alibi_slopes, q_len, q_len,
            dropout_p, softmax_scale, False, causal, window_size[0],
            window_size[1], return_softmax, None)
        out = out.to(q.dtype)

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_q_lens, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = q_len
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.bsz = bsz
        out = einops.rearrange(out, "(b s) ... -> b s ...", b=bsz)
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_q_lens, rng_state = ctx.saved_tensors

        dout = einops.rearrange(dout, "b s ... -> (b s) ...", b=ctx.bsz)
        dq, dk, dv, softmax_d = torch_xla._XLAC._flash_attention_backward(
            dout, q, k, v, out, softmax_lse, cu_q_lens, cu_q_lens,
            ctx.alibi_slopes, ctx.max_seqlen, ctx.max_seqlen, ctx.dropout_p,
            ctx.softmax_scale, False, ctx.causal, ctx.window_size[0],
            ctx.window_size[1], ctx.deterministic, None, rng_state)

        dq = einops.rearrange(dq, "(b s) ... -> b s ...", b=ctx.bsz)
        dk = einops.rearrange(dk, "(b s) ... -> b s ...", b=ctx.bsz)
        dv = einops.rearrange(dv, "(b s) ... -> b s ...", b=ctx.bsz)

        dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., :dout.shape[-1]]
        dv = dv[..., :dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None


def flash_attn_varlen_qkvpacked_xla(
    qkv,
    cu_seqlens,
    max_seqlen,
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
        cu_seqlens,
        max_seqlen,
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
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
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
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
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
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    assert q.dtype in [torch.bfloat16,
                       torch.float16], 'flash attention only supports fp16/bf16'
    return FlashAttnVarlenXla.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
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
