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
                                 causal=False):

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
                      attention_mask: Optional[torch.Tensor] = None, causal: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _, _ = query_states.size()

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            causal=causal)

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
                                 causal=False):

        # Contains at least one padding token in the sequence
        if attention_mask is None:
            attn_output = ta.ops.flash_attn_xla(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                causal=causal)  # re fill the masked with 0.f
        else:
            attn_output = ta.ops.flash_attn_varlen_xla(
                query_states,
                key_states,
                value_states,
                attention_mask=attention_mask,
                dropout_p=dropout,
                causal=causal)
        return attn_output
    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None, causal: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _, _ = query_states.size()

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=0.0,
            causal=causal)

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
        ret = model(q, k, v, attention_mask=attention_mask)

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
            q_xla, k_xla, v_xla, attention_mask=attention_mask_xla)

    (
        dq_xla,
        dk_xla,
        dv_xla,
    ) = torch.autograd.grad(ret_xla, (q_xla, k_xla, v_xla), g_xla)
    ta.sync()

    assert torch.allclose(
        ret_xla.cpu().detach(),
        ret.cpu().detach(),
        rtol=1e-1,
        atol=1e-1,
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
