from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

import torchacc as ta
from torchacc.ops import flash_attn_varlen_xla


def unflatten_output(shape, out, cu_seqlens):
    num_seq = len(cu_seqlens) - 1
    padded_out = torch.zeros(*shape, device=out.device, dtype=out.dtype)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        padded_out[i, :end - start] = out[start:end]
    return padded_out


def flash_attn_fixed_len(
    q: torch.Tensor,  # [bsz, seqlen, numhead, headdim]
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    alibi_slopes: Optional[tuple] = None,
    deterministic: bool = False,
) -> torch.Tensor:
    bsz, seq_q, nhead, headdim = q.shape
    seq_k = k.shape[1]
    if ta.is_lazy_tensor(q):
        q = q.flatten(0, 1).contiguous()
        k = k.flatten(0, 1).contiguous()
        v = v.flatten(0, 1).contiguous()
        cu_seqlens_q = torch.arange(
            0, (bsz + 1) * seq_q,
            step=seq_q,
            dtype=torch.int32,
            device=q.device)
        cu_seqlens_k = torch.arange(
            0, (bsz + 1) * seq_k,
            step=seq_k,
            dtype=torch.int32,
            device=k.device)
        out = flash_attn_varlen_xla(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_q,
            seq_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        ).unflatten(0, (bsz, seq_q))
    else:
        out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
    return out


def flash_attention(
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
) -> torch.Tensor:
    if q_lens is None and k_lens is None:
        return flash_attn_fixed_len(
            q,
            k,
            v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic)

    bsz, seq_q, nhead, headdim = q.shape
    seq_k = k.shape[1]
    if q_lens is None:
        q_lens = torch.tensor([seq_q] * bsz, dtype=torch.int32)
    if k_lens is None:
        k_lens = torch.tensor([seq_k] * bsz, dtype=torch.int32)

    q = torch.cat([u[:l] for u, l in zip(q, q_lens)])
    k = torch.cat([u[:l] for u, l in zip(k, k_lens)])
    v = torch.cat([u[:l] for u, l in zip(v, k_lens)])

    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
        0, dtype=torch.int32).to(
            q.device, non_blocking=True)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
        0, dtype=torch.int32).to(
            k.device, non_blocking=True)

    if ta.is_lazy_tensor(q):
        out = flash_attn_varlen_xla(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_q,
            seq_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
    else:
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seq_q,
            seq_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )

    return unflatten_output((bsz, seq_q, nhead, headdim), out,
                            cu_seqlens_q).clone()


def slice_forward(tensor: torch.Tensor,
                  seq_dim: int,
                  process_group: Optional[dist.ProcessGroup] = None):
    """ Slice the input tensor along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.

    Args:
        tensor (torch.Tensor): [batch_size, seqlen, nheads, headdim].
        seq_dim (int): The dimension to be split.
        process_group (torch.distributed.ProcessGroup, optional): The context parallel group.

    Returns:
        tensor (torch.Tensor): [batch_size, seqlen // cp_size, nheads, headdim].
    """
    cp_size = dist.get_world_size(process_group)
    cp_rank = dist.get_rank(process_group)
    if cp_size > 1:
        if tensor.shape[seq_dim] % cp_size != 0:
            raise ValueError(f"The seqlen {tensor.shape[seq_dim]} needs to" \
                             f" be divisible by the size of process group {cp_size}.")
        tensor = tensor.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
    return tensor


class GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from context parallel region and concatenate.
    """

    @staticmethod
    def forward(ctx, tensor, seq_dim, process_group):
        ctx.process_group = process_group
        ctx.seq_dim = seq_dim
        # skip if only one rank involved
        cp_size = dist.get_world_size(process_group)
        if cp_size == 1:
            return tensor
        tensor = tensor.view(
            *tensor.shape[0:seq_dim],
            2,
            -1,
            *tensor.shape[(seq_dim + 1):],
        )
        # all gather
        tensor_list = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(tensor_list, tensor, group=process_group)
        # concat
        tensor = torch.cat(tensor_list, dim=seq_dim).contiguous()
        index = list(range(0, cp_size * 2, 2)) + list(
            range(cp_size * 2 - 1, 0, -2))
        index = torch.tensor(index, device=tensor.device)
        tensor = tensor.index_select(seq_dim, index)
        tensor = tensor.view(*tensor.shape[0:seq_dim], -1,
                             *tensor.shape[(seq_dim + 2):])
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return slice_forward(grad_output, ctx.seq_dim,
                             ctx.process_group), None, None


def gather_forward_split_backward(tensor, seq_dim, process_group):
    """ Gather the input tensor along sequence dimension during forward
        and split the input tensor during backward, which are parallelized
        across GPUs in a context parallel group.

    Args:
        tensor (torch.Tensor): [batch_size, seqlen, nheads, headdim].
        seq_dim (int): The dimension for gather and split.
        process_group (torch.distributed.ProcessGroup, optional): The context parallel group.

    Returns:
        tensor (torch.Tensor): [batch_size, seqlen * cp_size, nheads, headdim].
    """
    return GatherForwardSplitBackward.apply(tensor, seq_dim, process_group)


def all_to_all(x, scatter_dim, gather_dim, group=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.
    """
    world_size = dist.get_world_size(group)
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


class AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input: input tensor
        scatter_dim: scatter dimension
        gather_dim: gather dimension
        group: process group
    """

    @staticmethod
    def forward(ctx, input, scatter_dim, gather_dim, group):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return all_to_all(input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        return (all_to_all(grad_output, ctx.gather_dim, ctx.scatter_dim,
                           ctx.group), None, None, None)


def diff_all_to_all(input, scatter_dim, gather_dim, group=None):
    return AllToAll.apply(input, scatter_dim, gather_dim, group)


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError(
                "first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse,
                                                   block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, :end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1),
                          dtype=torch.float32,
                          device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, :end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()


class RingComm:

    def __init__(self, process_group: dist.ProcessGroup, reverse: bool = False):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        if reverse:
            self.recv_rank = (self.rank + 1) % self.world_size
            self.send_rank = (self.rank - 1) % self.world_size
        else:
            self.send_rank = (self.rank + 1) % self.world_size
            self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group,
                                                  self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group,
                                                  self.recv_rank)

    def send_recv(self,
                  to_send: torch.Tensor,
                  recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(
            dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(
            dist.irecv, res, self.recv_rank, group=self._process_group)

        if self.rank % 2 == 0:
            self._ops.append(send_op)
            self._ops.append(recv_op)
        else:
            self._ops.append(recv_op)
            self._ops.append(send_op)
        return res

    def commit(self):
        assert len(self._ops) > 0
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            return
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []
