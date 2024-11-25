import os
import unittest

import einops
import pytest
import torch
import torch.distributed as dist
import torchacc as ta
import torchacc.ops.context_parallel as context_parallel
from flash_attn import flash_attn_func
from torchacc.ops import flash_attn_xla

from utils.distributed import (MultiProcessTestBase,
                               instantiate_parametrized_tests, parametrize,
                               skip_if_lt_x_gpu)

from torchacc.utils.import_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


@pytest.fixture(autouse=True, scope="module")
def setup_env():
    orign_env = os.getenv('PJRT_ALLOCATOR_FRACTION')
    os.environ['PJRT_ALLOCATOR_FRACTION'] = '0.5'
    yield
    if orign_env is None:
        os.environ.pop('PJRT_ALLOCATOR_FRACTION', None)
    else:
        os.environ['PJRT_ALLOCATOR_FRACTION'] = orign_env


@instantiate_parametrized_tests
class ContextParallelTest(MultiProcessTestBase):

    @skip_if_lt_x_gpu(2)
    @parametrize("is_cuda", [False, True])
    @parametrize("test_varlen", [False, True])
    @parametrize("cp_type", ["ulysses", "ring", "context_parallel_2d"])
    # @parametrize("batch_size", [1, 4])
    @parametrize("batch_size", [4])
    @parametrize("seq_len", [512, 2048])
    @parametrize("n_heads", [8, 32])
    # @parametrize("dims", [16, 128])
    @parametrize("dims", [16])
    def test_cp(
        self,
        is_cuda,
        test_varlen,
        cp_type,
        batch_size,
        seq_len,
        n_heads,
        dims,
    ):
        # TODO(huangyitong.hyt): fix this
        if dims != 16 or batch_size == 1:
            raise unittest.SkipTest("Correctness issue")
        torch.manual_seed(101)
        if is_cuda:
            self.init_pg("nccl")
            device = dist.get_rank()
        else:
            self.init_pg("lazy")
            xm.set_rng_state(101)
            device = ta.lazy_device()

        if cp_type == 'context_parallel_2d':
            assert dist.get_world_size() % 2 == 0
            context_parallel.initialize_context_parallel(
                dist.get_world_size(),
                dist.get_world_size() // 2)
        else:
            context_parallel.initialize_context_parallel(dist.get_world_size())

        if cp_type == 'ulysses':
            cp_func = context_parallel.ulysses
        elif cp_type == 'ring':
            cp_func = context_parallel.ring_attention

        cp_size = dist.get_world_size()
        cp_group = context_parallel.get_context_parallel_group()

        dtype = torch.bfloat16
        hidden_states = torch.linspace(
            -0.5, 0.5, batch_size * seq_len * dims * n_heads * 3).reshape(
                batch_size, seq_len, dims * n_heads * 3).to(dtype).to(device)

        # Context parallel
        hidden_states_cp = hidden_states.detach().clone().requires_grad_()
        hidden_states_cp_tmp = context_parallel.split_forward_gather_backward(
            hidden_states_cp, 1, cp_group)
        B, N, C = hidden_states_cp_tmp.shape
        qkv = hidden_states_cp_tmp.reshape(B, N, 3, n_heads,
                                           dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]

        q_lens = None
        if test_varlen:
            q_lens = torch.tensor([N * cp_size for _ in range(B)],
                                  dtype=torch.int32)

        if cp_type == 'ring':
            # FIXME(wenting.swt): correct usage of flash_attn_varlen_xla
            self.skipTest("Correctness issue")
        if cp_type == 'context_parallel_2d':
            # FIXME(wenting.swt): correct usage of flash_attn_varlen_xla
            self.skipTest("Correctness issue")
            output_cp = context_parallel.context_parallel_2d(
                q,
                k,
                v,
                q_lens=q_lens,
                k_lens=q_lens,
                softmax_scale=None,
                dropout_p=0.0,
                intra_process_group=context_parallel.get_intra_cp_process_group(
                ),
                inter_process_group=context_parallel.get_inter_cp_process_group(
                ))
        else:
            output_cp = cp_func(
                q,
                k,
                v,
                q_lens=q_lens,
                k_lens=q_lens,
                softmax_scale=None,
                dropout_p=0.0,
                process_group=cp_group)

        output_cp = context_parallel.gather_forward_split_backward(
            output_cp, seq_dim=1, process_group=cp_group)

        loss_cp = torch.sum(output_cp)
        loss_cp.backward()

        # Flash Attention
        hidden_states_fa = hidden_states.detach().clone().requires_grad_()
        B, N, C = hidden_states_fa.shape
        qkv = hidden_states_fa.reshape(B, N, 3, n_heads,
                                       dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k, v = [x.transpose(1, 2) for x in [q, k, v]]

        if ta.is_lazy_tensor(q):
            output_fa = flash_attn_xla(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                return_attn_probs=False)
        else:
            output_fa = flash_attn_func(
                q,
                k,
                v,
                dropout_p=0.0,
                softmax_scale=None,
                causal=False,
                return_attn_probs=False)

        loss_fa = torch.sum(output_fa)
        loss_fa.backward()

        ta.sync()

        fwd_close = torch.allclose(
            output_fa.cpu().detach().to(torch.float32),
            output_cp.cpu().detach().to(torch.float32),
            rtol=1e-5,
            atol=1e-2)
        bwd_close = torch.allclose(
            hidden_states_fa.grad.cpu().detach().to(torch.float32),
            hidden_states_cp.grad.cpu().detach().to(torch.float32),
            rtol=1e-5,
            atol=1e-2)

        assert fwd_close
        assert bwd_close

        # clean up
        self.destroy_pg()
