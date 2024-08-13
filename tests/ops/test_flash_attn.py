import os

import pytest
import torch
import torchacc as ta
from flash_attn import flash_attn_func


@pytest.fixture(autouse=True, scope="module")
def setup_env():
    orign_env = os.getenv('PJRT_ALLOCATOR_FRACTION')
    os.environ['PJRT_ALLOCATOR_FRACTION'] = '0.5'
    yield
    if orign_env is None:
        os.environ.pop('PJRT_ALLOCATOR_FRACTION', None)
    else:
        os.environ['PJRT_ALLOCATOR_FRACTION'] = orign_env


# FIXME:
# seqlen_q = 113, seqlen_k = 203, d = 32, dropout_p = 0.0, causal = False, local = True, alibi = False, deterministic = False, mha_type = 'mha', dtype = torch.float16
# seqlen_q = 113, seqlen_k = 203, d = 32, dropout_p = 0.0, causal = True, local = False, alibi = False, deterministic = False, mha_type = 'mha', dtype = torch.float16
# seqlen_q = 113, seqlen_k = 203, d = 256, dropout_p = 0.0, causal = False, local = False, alibi = True, deterministic = False, mha_type = 'mqa', dtype = torch.bfloat16
# seqlen_q = 512, seqlen_k = 256, d = 32, dropout_p = 0.0, causal = False, local = False, alibi = True, deterministic = False, mha_type = 'gqa', dtype = torch.bfloat16
# seqlen_q = 113, seqlen_k = 203, d = 32, dropout_p = 0.17, causal = False, local = False, alibi = False, deterministic = False, mha_type = 'mha', dtype = torch.float16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("alibi", [False, True])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32, 96, 111, 224, 256])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
def test_flash_attn_varlen_output(seqlen_q, seqlen_k, d, dropout_p, causal,
                                  local, alibi, deterministic, mha_type, dtype):
    # TODO(to wenting.swt): maybe we need support this
    if d % 8 != 0:
        pytest.skip(reason="Expected head_size_og % 8 == 0 to be true")
    # TODO(to wenting.swt): fix the correctness issue, refer to FIXME
    if local or causal or alibi or (dropout_p > 0):
        pytest.skip(reason="Correctness issue")

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else tuple(
        torch.randint(0, seqlen_k, (2,)).tolist())
    q = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True)
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True)
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device=device,
        dtype=dtype,
        requires_grad=True)

    if alibi:
        alibi_slopes = torch.rand(
            batch_size, nheads, device=device, dtype=torch.float32) * 0.3
    else:
        alibi_slopes = None

    out_fa = flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=False,
    )
    g = torch.randn_like(out_fa)
    (
        dq_fa,
        dk_fa,
        dv_fa,
    ) = torch.autograd.grad(out_fa, (q, k, v), g)
    dq_fa = dq_fa.cpu().detach()
    dk_fa = dk_fa.cpu().detach()
    dv_fa = dv_fa.cpu().detach()
    out_fa = out_fa.cpu().detach()
    q = q.cpu().detach()
    k = k.cpu().detach()
    v = v.cpu().detach()
    g = g.cpu().detach()
    if alibi:
        alibi_slopes = alibi_slopes.cpu()
    torch.cuda.synchronize()

    # xla
    device = ta.lazy_device()
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seqlen_q,
        step=seqlen_q,
        dtype=torch.int32,
        device=q.device)
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seqlen_k,
        step=seqlen_k,
        dtype=torch.int32,
        device=q.device)
    if alibi:
        alibi_slopes = alibi_slopes.cpu().to(device)
    out_xla = ta.ops.flash_attn_varlen_xla(
        q.flatten(0, 1),
        k.flatten(0, 1),
        v.flatten(0, 1),
        cu_seqlens_q,
        cu_seqlens_k,
        seqlen_q,
        seqlen_k,
        dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
    ).unflatten(0, (-1, seqlen_q))
    g = g.to(device)
    (
        dq_xla,
        dk_xla,
        dv_xla,
    ) = torch.autograd.grad(out_xla, (q, k, v), g)
    ta.mark_step(wait=True)
    dq_xla = dq_xla.cpu().detach()
    dk_xla = dk_xla.cpu().detach()
    dv_xla = dv_xla.cpu().detach()
    out_xla = out_xla.cpu().detach()

    # TODO(to wenting.swt): The rtol and atol here are a bit high.
    assert torch.allclose(out_xla, out_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
    assert torch.allclose(dq_xla, dq_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
    assert torch.allclose(dk_xla, dk_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
    assert torch.allclose(dv_xla, dv_fa, rtol=1e-2, atol=1e-2, equal_nan=True)
