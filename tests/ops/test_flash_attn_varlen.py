from typing import Optional, Tuple
import torch.nn.functional as F
import pytest
import torch
import torchacc as ta
from torch import nn

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class FlashAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def _flash_attention_forward(self,
                                 query_states,
                                 key_states,
                                 value_states,
                                 attention_mask,
                                 query_length,
                                 dropout=0.0,
                                 softmax_scale=None,
                                 causal=False,
                                 position_ids=None):

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states,
                attention_mask.contiguous(), query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size,
                                    query_length)  # re fill the masked with 0.f
        elif position_ids is not None:  # now only support same seq for q and k
            total = position_ids.shape[-1]
            cumsum = torch.nonzero(position_ids == 0).squeeze(1)
            cu_seqlens_q = torch.cat([
                cumsum,
                torch.tensor(
                    [total], dtype=torch.int32, device=position_ids.device)
            ],
                                     dim=0).to(torch.int32)
            max_seqlen_in_batch_q = (cu_seqlens_q[1:] -
                                     cu_seqlens_q[:-1]).max().item()
            cu_seqlens_k = cu_seqlens_q
            max_seqlen_in_batch_k = max_seqlen_in_batch_q
            indices_q = torch.arange(
                total, dtype=torch.int64, device=query_states.device)

            attn_output = flash_attn_varlen_func(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                causal=causal).unsqueeze(0)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask,
                    query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape  # b, s, h, d

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                              head_dim),
            indices_k  # filter out the key with unmask query
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads,
                                head_dim), indices_k)
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads,
                                    head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None, causal: bool = False, position_ids:torch.Tensor=None) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            bsz, q_len, _, _ = query_states.size()
        else:
            q_len, _, _ = query_states.size()

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            causal=causal,
            position_ids=position_ids.squeeze(0)
            if position_ids is not None else None)

        return attn_output


class FlashAttentionXla(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def _flash_attention_forward(self,
                                 query_states,
                                 key_states,
                                 value_states,
                                 attention_mask,
                                 query_length,
                                 dropout=0.0,
                                 softmax_scale=None,
                                 causal=False,
                                 position_ids=None):

        # Contains at least one padding token in the sequence
        if attention_mask is None and position_ids is None:
            attn_output = ta.ops.flash_attn_xla(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                causal=causal)  # re fill the masked with 0.f
        elif attention_mask is not None:
            attn_output = ta.ops.flash_attn_varlen_xla(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout_p=dropout,
                causal=causal)
        else:
            attn_output = ta.ops.flash_attn_varlen_position_ids_xla(
                query_states,
                key_states,
                value_states,
                position_ids=position_ids,
                dropout_p=dropout,
                causal=causal)
        return attn_output
    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None, causal: bool = False, position_ids: torch.Tensor = None) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _, _ = query_states.size()

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            causal=causal,
            position_ids=position_ids)

        return attn_output


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("seqlen", [128, 1024])
def test_flash_attn_varlen(seqlen, d, dtype, mha_type, causal):

    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen, nheads_k, d, device=device, dtype=dtype)
    g = torch.randn_like(q)
    attention_mask = torch.zeros(
        batch_size, seqlen, dtype=torch.int32).to(device)

    k_lengths = torch.randint(low=2, high=seqlen, size=(batch_size,))

    for i in range(batch_size):
        k_len = k_lengths[i].item()
        attention_mask[i, :k_len] = 1

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    model = FlashAttention(d * nheads, nheads, nheads_k).to(device)
    model.train()
    with torch.cuda.amp.autocast(dtype=dtype):
        ret = model(q, k, v, attention_mask=attention_mask, causal=causal)

    (
        dq,
        dk,
        dv,
    ) = torch.autograd.grad(ret, (q, k, v), g)

    torch.cuda.synchronize()

    q = q.cpu().detach()
    k = k.cpu().detach()
    v = v.cpu().detach()
    g = g.cpu().detach()
    attention_mask = attention_mask.cpu().detach()

    torch.manual_seed(0)
    device = ta.lazy_device()
    q_xla = q.to(device)
    k_xla = k.to(device)
    v_xla = v.to(device)
    g_xla = g.to(device)
    q_xla.requires_grad = True
    k_xla.requires_grad = True
    v_xla.requires_grad = True
    attention_mask_xla = attention_mask.to(device)
    model_xla = FlashAttentionXla(d * nheads, nheads, nheads_k).to(device)
    model_xla.train()

    with torch.cuda.amp.autocast(dtype=dtype):
        ret_xla = model_xla(
            q_xla,
            k_xla,
            v_xla,
            attention_mask=attention_mask_xla,
            causal=causal)

    (
        dq_xla,
        dk_xla,
        dv_xla,
    ) = torch.autograd.grad(ret_xla, (q_xla, k_xla, v_xla), g_xla)
    ta.sync()

    assert torch.allclose(
        ret_xla.cpu().detach(),
        ret.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dq_xla.cpu().detach(),
        dq.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dk_xla.cpu().detach(),
        dk.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dv_xla.cpu().detach(),
        dv.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("seqlen", [128, 1024])
def test_flash_attn_varlen_position_ids(seqlen, d, dtype, mha_type, causal):

    batch_size = 4
    nheads = 9
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)

    torch.manual_seed(0)
    device = "cuda"

    def generate_qkv_and_position_ids(batch_size, max_seqlen_q, max_seqlen_k,
                                      dtype, n_heads_q, n_heads_k, device,
                                      head_dim, use_same_seqlen):
        '''
        generate varlen qkv and postion_ids and pad total seqlen to 8
        '''
        seq_len_q = torch.randint(1, max_seqlen_q + 1, (batch_size,))
        if use_same_seqlen:
            seq_len_k = seq_len_q.clone()
        else:
            seq_len_k = torch.randint(1, max_seqlen_k + 1, (batch_size,))
        total_q = seq_len_q.sum().item()
        total_k = seq_len_k.sum().item()

        padd_q = 0 if total_q % 8 == 0 else 8 - total_q % 8
        padd_k = 0 if total_k % 8 == 0 else 8 - total_k % 8

        # padding to last q and k
        if padd_q:
            seq_len_q[-1] += padd_q
            total_q += padd_q
        assert total_q % 8 == 0
        if padd_k:
            seq_len_k[-1] += padd_k
            total_k += padd_k
        assert total_k % 8 == 0

        q = torch.randn((1, total_q, n_heads_q, head_dim),
                        dtype=dtype,
                        device=device)
        k = torch.randn((1, total_k, n_heads_k, head_dim),
                        dtype=dtype,
                        device=device)
        v = torch.randn((1, total_k, n_heads_k, head_dim),
                        dtype=dtype,
                        device=device)

        assert torch.all(seq_len_q > 0)
        assert torch.all(seq_len_k > 0)

        position_ids_q = torch.cat([
            torch.arange(0, seq_len, dtype=torch.int32, device=device)
            for seq_len in seq_len_q
        ],
                                   dim=0).unsqueeze(0)
        position_ids_k = torch.cat([
            torch.arange(0, seq_len, dtype=torch.int32, device=device)
            for seq_len in seq_len_k
        ],
                                   dim=0).unsqueeze(0)
        assert position_ids_q.shape[1] % 8 == 0
        assert position_ids_k.shape[1] % 8 == 0

        return q, k, v, position_ids_q, position_ids_k

    q, k, v, position_ids_q, position_ids_k = generate_qkv_and_position_ids(
        batch_size,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        dtype=dtype,
        n_heads_q=nheads,
        n_heads_k=nheads_k,
        device=device,
        head_dim=d,
        use_same_seqlen=True)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    g = torch.randn_like(q)

    model = FlashAttention(d * nheads, nheads, nheads_k).to(device)
    model.train()
    with torch.cuda.amp.autocast(dtype=dtype):
        ret = model(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            position_ids=position_ids_q,
            causal=causal)

    (
        dq,
        dk,
        dv,
    ) = torch.autograd.grad(ret, (q, k, v), g)

    torch.cuda.synchronize()

    q = q.cpu().detach()
    k = k.cpu().detach()
    v = v.cpu().detach()
    g = g.cpu().detach()

    torch.manual_seed(0)
    device = ta.lazy_device()
    q_xla = q.to(device)
    k_xla = k.to(device)
    v_xla = v.to(device)
    g_xla = g.to(device)
    position_ids_xla = position_ids_k.to(device)
    q_xla.requires_grad = True
    k_xla.requires_grad = True
    v_xla.requires_grad = True

    model_xla = FlashAttentionXla(d * nheads, nheads, nheads_k).to(device)
    model_xla.train()

    with torch.cuda.amp.autocast(dtype=dtype):
        ret_xla = model_xla(
            q_xla, k_xla, v_xla, position_ids=position_ids_xla, causal=causal)

    (
        dq_xla,
        dk_xla,
        dv_xla,
    ) = torch.autograd.grad(ret_xla, (q_xla, k_xla, v_xla), g_xla)
    ta.sync()

    assert torch.allclose(
        ret_xla.cpu().detach(),
        ret.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dq_xla.cpu().detach(),
        dq.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dk_xla.cpu().detach(),
        dk.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
    assert torch.allclose(
        dv_xla.cpu().detach(),
        dv.cpu().detach(),
        rtol=1e-2,
        atol=1e-2,
        equal_nan=True)
