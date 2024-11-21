from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

import torchacc as ta
from torchacc.ops import flash_attn_xla, flash_attn_varlen_xla


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
    bsz, seq_q, _, _ = q.shape
    if ta.is_lazy_tensor(q):
        out = flash_attn_xla(
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

    if ta.is_lazy_tensor(q):
        cols = torch.arange(seq_k, device=k.device).unsqueeze(0)
        mask = cols < k_lens.to(k.device, non_blocking=True).unsqueeze(1)
        # FIXME(wenting.swt): support seperate attention_mask for q and k
        out = flash_attn_varlen_xla(
            q,
            k,
            v,
            mask,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
        )
        return out
    else:
        q = torch.cat([u[:l] for u, l in zip(q, q_lens)])
        k = torch.cat([u[:l] for u, l in zip(k, k_lens)])
        v = torch.cat([u[:l] for u, l in zip(v, k_lens)])

        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(
                q.device, non_blocking=True)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(
                k.device, non_blocking=True)
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


def all_gather(tensor: torch.Tensor,
               process_group: Optional[dist.ProcessGroup] = None):
    cp_size = dist.get_world_size(process_group)
    if cp_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(cp_size)]
    torch.distributed.all_gather(tensor_list, tensor, group=process_group)
    return tensor_list


def _split(input, dim, process_group):
    # skip if world_size == 1
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input

    # split sequence
    if input.size(dim) % world_size != 0:
        raise ValueError(f"The seqlen {input.size(dim)} needs to" \
                          f" be divisible by the size of process group {world_size}.")
    return input.chunk(world_size, dim=dim)[rank].contiguous()


def _gather(input, dim, process_group):
    # skip if world_size == 1
    world_size = dist.get_world_size(process_group)
    if world_size == 1:
        return input

    # gather sequence
    output = all_gather(input, process_group)
    return torch.cat(output, dim=dim).contiguous()


class SplitForwardGatherBackward(torch.autograd.Function):
    """Split the input and scatter to context parallel region.
    """

    @staticmethod
    def forward(ctx, tensor, seq_dim, process_group, grad_scale=None):
        ctx.process_group = process_group
        ctx.seq_dim = seq_dim
        ctx.grad_scale = grad_scale
        return _split(tensor, seq_dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(
                group=ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(
                group=ctx.process_group)
        return _gather(grad_output, ctx.seq_dim, ctx.process_group), None, None


class GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from context parallel region and concatenate.
    """

    @staticmethod
    def forward(ctx, tensor, seq_dim, process_group, grad_scale=None):
        ctx.process_group = process_group
        ctx.seq_dim = seq_dim
        ctx.grad_scale = grad_scale
        return _gather(tensor, seq_dim, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(
                group=ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(
                group=ctx.process_group)
        return _split(grad_output, ctx.seq_dim, ctx.process_group), None, None


def split_forward_gather_backward(tensor,
                                  seq_dim,
                                  process_group,
                                  grad_scale=None):
    """ Split the input tensor along sequence dimension during forward
        and gather the input tensor during backward, which are parallelized
        across GPUs in a context parallel group.
    Args:
        tensor (torch.Tensor): [batch_size, seqlen * cp_size, nheads, headdim].
        seq_dim (int): The dimension for split and gather.
        process_group (torch.distributed.ProcessGroup, optional): The context parallel group.
        grad_scale (str, optional): The gradient scale. 'up' or 'down'. 'up' means the
            gradient will be multiplied by the size of process group, and 'down' means the
            gradient will be divided by the size of process group.

    Returns:
        tensor (torch.Tensor): [batch_size, seqlen, nheads, headdim].
    """
    return SplitForwardGatherBackward.apply(tensor, seq_dim, process_group)


def gather_forward_split_backward(tensor,
                                  seq_dim,
                                  process_group,
                                  grad_scale=None):
    """ Gather the input tensor along sequence dimension during forward
        and split the input tensor during backward, which are parallelized
        across GPUs in a context parallel group.

    Args:
        tensor (torch.Tensor): [batch_size, seqlen, nheads, headdim].
        seq_dim (int): The dimension for gather and split.
        process_group (torch.distributed.ProcessGroup, optional): The context parallel group.
        grad_scale (str, optional): The gradient scale. 'up' or 'down'. 'up' means the
            gradient will be multiplied by the size of process group, and 'down' means the
            gradient will be divided by the size of process group.

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
