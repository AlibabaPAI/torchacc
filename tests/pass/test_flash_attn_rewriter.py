import math
import os
from typing import Optional, Tuple

import einops
import pytest
import torch
import torchacc as ta
from torch import nn

BATCH_SIZE = 64
SEQ_LEN = 256
DIMS = 16
N_HEADS = 8
hidden_size = DIMS * N_HEADS
num_attention_heads = N_HEADS
num_key_value_heads = N_HEADS


@pytest.fixture(autouse=True, scope="module")
def setup_env():
    xla_flags = os.getenv("XLA_FLAGS", "")
    if 'xla_gpu_enable_flash_attention' not in xla_flags:
        os.environ[
            'XLA_FLAGS'] = xla_flags + ' --xla_gpu_enable_flash_attention=true'
    yield
    if len(xla_flags) == 0:
        os.environ.pop('XLA_FLAGS', None)
    else:
        os.environ['XLA_FLAGS'] = xla_flags


class FlashAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from torchacc.ops import flash_attn_varlen_qkvpacked_xla

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))

        qkv = torch.stack([query_states, key_states, value_states], dim=2)
        qkv = qkv.transpose(1, 3)

        qkv = einops.rearrange(qkv, "b s ... -> (b s) ...")
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=qkv.device)
        output = flash_attn_varlen_qkvpacked_xla(
            qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
        output = einops.rearrange(output, "(b s) ... -> b s ...", b=bsz)

        return self.o_proj(einops.rearrange(output, "b s h d -> b s (h d)"))


class LlamaAttention(nn.Module):

    def __init__(self, tp=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False)
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.tp = tp

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        if not self.tp:
            key_states = key_states.transpose(2, 3)

            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(
                self.head_dim)
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1,
                dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        else:
            attn_weights = torch.einsum('abij,abjk->abik', query_states,
                                        key_states.transpose(2, 3)) / math.sqrt(
                                            self.head_dim)
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1,
                dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.einsum('abij,abjk->abik', attn_weights,
                                       value_states)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


@pytest.mark.skip(reason="Currently, flash attn rewriter is not yet supported.")
@pytest.mark.parametrize("enable_tp_attention", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_rewriter(enable_tp_attention, dtype):
    torch.manual_seed(101)
    device = ta.lazy_device()
    model = LlamaAttention(enable_tp_attention).to(device)
    model.train()
    hidden_states = torch.linspace(
        -0.5, 0.5, BATCH_SIZE * SEQ_LEN * DIMS * N_HEADS,
        device=device).reshape(BATCH_SIZE, SEQ_LEN, DIMS * N_HEADS).to(dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    rets = []
    for i in range(3):
        with torch.cuda.amp.autocast(dtype=dtype):
            ret = model(hidden_states)
            rets.append(ret)
            loss = ret.flatten().sum()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        ta.sync()

    torch.manual_seed(101)
    device = ta.lazy_device()
    model_fa = FlashAttention().to(device)
    model_fa.train()
    hidden_states_fa = torch.linspace(
        -0.5, 0.5, BATCH_SIZE * SEQ_LEN * DIMS * N_HEADS,
        device=device).reshape(BATCH_SIZE, SEQ_LEN, DIMS * N_HEADS).to(dtype)
    opt_fa = torch.optim.SGD(model_fa.parameters(), lr=0.01)
    rets_fa = []
    for i in range(3):
        with torch.cuda.amp.autocast(dtype=dtype):
            ret_fa = model_fa(hidden_states_fa)
            rets_fa.append(ret_fa)
            loss_fa = ret_fa.flatten().sum()
        loss_fa.backward()
        opt_fa.step()
        torch.cuda.synchronize()
        ta.sync()

    for idx, (ret, ret_fa) in enumerate(zip(rets, rets_fa)):
        allclose_fwd = torch.allclose(
            ret.cpu().detach(), ret_fa.cpu().detach(), rtol=1e-5, atol=1e-5)
        allclose_bwd = torch.allclose(
            model.q_proj.weight.cpu().detach(),
            model_fa.q_proj.weight.cpu().detach(),
            rtol=1e-5,
            atol=1e-5)
        assert allclose_fwd
        assert allclose_bwd
