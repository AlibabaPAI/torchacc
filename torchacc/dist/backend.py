# This file is largely inspired by and partially follows the structure of
# https://github.com/pytorch/xla/blob/master/torch_xla/distributed/xla_backend.py
from functools import wraps
import warnings

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup, ProcessGroupNCCL
import torch_xla.core.xla_model as xm
from torchacc.utils.logger import logger

_BACKEND_NAME = 'lazy'
_EAGER_BACKEND_NAME = 'nccl'

# Note [Preserve send tensor]
# Save the output tensors of send to preserve send op in the
# computation graph. Since there is a token dependency between cc ops,
# we only need to preserve the last send.
_preserved_send_tensor = None

_default_nccl_pg = None
_group_count = 0


def _create_lazy_process_group(dist_backend_opts, pg_options):
    return ProcessGroupLazy(dist_backend_opts, pg_options)


def _register_lazy_backend():
    dist.Backend.register_backend(
        _BACKEND_NAME,
        _create_lazy_process_group,
        extended_api=True,
        devices=['xla', 'cuda'])


_register_lazy_backend()


def _process_group_name():
    global _group_count
    pg_name = str(_group_count)
    _group_count += 1
    return pg_name


def _ret_work(ret):
    fut = torch.futures.Future()
    fut.set_result(ret)
    return torch._C._distributed_c10d._create_work_from_future(fut)


def _get_full_group(ranks):
    world_size = dist.get_world_size()
    assert world_size % len(ranks) == 0
    mesh = [ranks]
    seen = set(ranks)
    gp = []
    for rank in range(world_size):
        if rank not in seen:
            gp.append(rank)
        if len(gp) == len(ranks):
            mesh.append(gp)
            gp = []
    return mesh


def _init_nccl_pg(
    group_size,
    group_rank,
    global_ranks_in_group,
    store,
    pg_options=None,
    timeout=None,
):
    global _default_nccl_pg
    # below is adapted from torch.distributed.distributed_c10d._new_process_group_helper
    if not dist.is_nccl_available():
        raise RuntimeError("Distributed package doesn't have NCCL built in")
    if pg_options is not None:
        assert isinstance(
            pg_options, ProcessGroupNCCL.Options
        ), "Expected pg_options argument to be of type ProcessGroupNCCL.Options"
        if pg_options._timeout != timeout:
            warnings.warn(
                "pg_options._timeout was specified, "
                "but timeout kwarg has a default value that will always override it. "
            )
    else:
        # default pg_options for NCCL
        pg_options = ProcessGroupNCCL.Options()
        pg_options.is_high_priority_stream = False
    pg_options._timeout = timeout

    # If our new group includes all ranks, we can reduce
    # overhead by splitting the communicator (`nccCommSplit`).

    # TODO: support this in the general case by calling
    # `nccCommSplit` with `NCCL_SPLIT_NOCOLOR` for the ranks
    # not in the communicator.
    split_from = None
    if (_default_nccl_pg is not None and
            len(global_ranks_in_group) == dist.get_world_size()):
        split_from = _default_nccl_pg

        if split_from:
            pg_options.split_from = split_from
            pg_options.split_color = torch.distributed.distributed_c10d._process_group_color(
                global_ranks_in_group)

    backend_class = ProcessGroupNCCL(store, group_rank, group_size, pg_options)
    backend_class._set_sequence_number_for_group()

    return backend_class


def cuda_use_nccl(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        self_or_cls = args[0]

        def check_tensors(arg):
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                return True
            elif isinstance(arg, (list, tuple)):
                return any(check_tensors(item) for item in arg)
            return False

        has_cuda_tensor = any(check_tensors(arg) for arg in args[1:]) or \
                            any(check_tensors(kwarg) for kwarg in kwargs.values())

        if has_cuda_tensor:
            return getattr(self_or_cls._nccl_pg, func.__name__)(*args[1:],
                                                                **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


class ProcessGroupLazy(ProcessGroup):
    '''ProcessGroup for lazy devices.
    '''

    def __init__(self, dist_backend_opts, pg_options=None):
        group_name = _process_group_name()
        prefix_store = dist.PrefixStore(f"{group_name}/",
                                        dist_backend_opts.store)
        base_pg_options = ProcessGroup.Options(backend='nccl')
        base_pg_options._timeout = dist_backend_opts.timeout
        super().__init__(prefix_store, dist_backend_opts.group_rank,
                         dist_backend_opts.group_size, base_pg_options)
        self._mesh = []

        # create nccl pg
        self._nccl_pg = _init_nccl_pg(dist_backend_opts.group_size,
                                      dist_backend_opts.group_rank,
                                      dist_backend_opts.global_ranks_in_group,
                                      dist_backend_opts.store, pg_options,
                                      dist_backend_opts.timeout)

        global _default_nccl_pg
        if _default_nccl_pg is None:
            _default_nccl_pg = self._nccl_pg

    def getBackendName(self):
        return _BACKEND_NAME

    def _get_reduce_type(self, reduce_op):
        if reduce_op == dist.ReduceOp.SUM:
            return xm.REDUCE_SUM
        elif reduce_op == dist.ReduceOp.PRODUCT:
            return xm.REDUCE_MUL
        elif reduce_op == dist.ReduceOp.BAND:
            return xm.REDUCE_AND
        elif reduce_op == dist.ReduceOp.BOR:
            return xm.REDUCE_OR
        elif reduce_op == dist.ReduceOp.MIN:
            return xm.REDUCE_MIN
        elif reduce_op == dist.ReduceOp.MAX:
            return xm.REDUCE_MAX
        elif reduce_op == dist.ReduceOp.BXOR:
            raise NotImplementedError(f'reduce op {reduce_op}')
        else:
            raise ValueError(f'Invalid reduce op {reduce_op}')

    @cuda_use_nccl
    def allreduce(self, tensors, opts):
        reduce_type = self._get_reduce_type(opts.reduceOp)
        xm.all_reduce(reduce_type, tensors, groups=self._mesh, pin_layout=False)
        return _ret_work(tensors)

    @cuda_use_nccl
    def allreduce_coalesced(self, tensors, all_reduce_options):
        reduce_type = self._get_reduce_type(all_reduce_options.reduceOp)
        xm.all_reduce(reduce_type, tensors, groups=self._mesh, pin_layout=False)
        return _ret_work(tensors)

    @cuda_use_nccl
    def allgather(self, output_tensors_list, input_tensors, opts=None):
        for input_tensor, output_tensors in zip(input_tensors,
                                                output_tensors_list):
            is_scalar = (input_tensor.dim() == 0)
            if is_scalar:
                input_tensor = torch.reshape(input_tensor, (1,))
            result = xm.all_gather(
                input_tensor, groups=self._mesh, pin_layout=False)
            for i, slice in enumerate(
                    torch.split(result, input_tensor.shape[0])):
                with torch.no_grad():
                    output_tensors[i].copy_(
                        slice if not is_scalar else torch.reshape(slice, ()))

        return _ret_work(
            [t for sublist in output_tensors_list for t in sublist])

    @cuda_use_nccl
    def allgather_coalesced(self,
                            output_tensors_list,
                            input_tensors,
                            opts=None):
        results = xm.all_gather(
            input_tensors, groups=self._mesh, pin_layout=False)
        for i, result in enumerate(results):
            for j, slice in enumerate(
                    torch.split(result, input_tensors[i].shape[0])):
                output_tensors_list[i][j].copy_(slice)

        return _ret_work(
            [t for sublist in output_tensors_list for t in sublist])

    @cuda_use_nccl
    def broadcast(self, tensors, opts):
        root_tensor = tensors[opts.rootTensor]
        # opts.rootRank is group rank
        root_global_rank = dist.get_global_rank(self, opts.rootRank)
        xm.collective_broadcast([root_tensor],
                                root_global_rank,
                                groups=self._mesh,
                                pin_layout=False)

        return _ret_work([root_tensor])

    @cuda_use_nccl
    def reduce_scatter(self, output_tensors, input_tensors_list, opts):
        for input_tensors, output_tensor in zip(input_tensors_list,
                                                output_tensors):
            # Ensure all inputs have the same shape.
            first_shape = input_tensors[0].shape
            for i, t in enumerate(input_tensors[1:]):
                if first_shape != t.shape:
                    raise ValueError(
                        f"Input {i+1}'s shape is different from input 0: "
                        f"{t.shape} vs {first_shape}")
            input_tensor = torch.cat(input_tensors)
            reduce_type = self._get_reduce_type(opts.reduceOp)
            groups = self._mesh
            shard_count = len(groups[0]) if groups else self.size()
            xm.reduce_scatter(
                reduce_type,
                input_tensor,
                scatter_dim=0,
                shard_count=shard_count,
                scale=1,
                groups=groups,
                output=output_tensor,
                pin_layout=False)

        return _ret_work(output_tensors)

    @cuda_use_nccl
    def reduce_scatter_coalesced(self, output_tensors, input_tensors_list,
                                 opts):
        input_tensor_list = []
        for input_tensors in input_tensors_list:
            # Ensure all inputs have the same shape.
            first_shape = input_tensors[0].shape
            for i, t in enumerate(input_tensors[1:]):
                if first_shape != t.shape:
                    raise ValueError(
                        f"Input {i+1}'s shape is different from input 0: "
                        f"{t.shape} vs {first_shape}")
            input_tensor = torch.cat(input_tensors)
            input_tensor_list.append(input_tensor)

        reduce_type = self._get_reduce_type(opts.reduceOp)
        groups = self._mesh
        shard_count = len(groups[0]) if groups else self.size()
        xm.reduce_scatter(
            reduce_type,
            input_tensor_list,
            scatter_dim=0,
            shard_count=shard_count,
            scale=1,
            groups=groups,
            output=output_tensors,
            pin_layout=False)

        return _ret_work(output_tensors)

    def barrier(self, opts):
        return self._nccl_pg.barrier(opts)

    @cuda_use_nccl
    def reduce(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def alltoall(self, output_tensor_list, input_tensor_list, opts):
        world_size = dist.get_world_size(self)
        x = torch.cat(input_tensor_list, dim=0)
        x = xm.all_to_all(x, 0, 0, world_size, self._mesh)
        for i, tensor in enumerate(x.chunk(world_size, dim=0)):
            with torch.no_grad():
                output_tensor_list[i].copy_(tensor)
        return _ret_work(output_tensor_list)

    @cuda_use_nccl
    def alltoall_base(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def gather(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def scatter(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def send(self, tensors, dst_rank, tag=0):
        # dst_rank is group rank
        dst_rank = dist.get_global_rank(self, dst_rank)
        rank = dist.get_rank()
        send_group = [rank, dst_rank] if dst_rank > rank else [dst_rank, rank]
        groups = _get_full_group(send_group)
        xm.all_reduce('sum', tensors, groups=groups, pin_layout=False)
        # Save the output tensors of send to preserve send op in the
        # computation graph. See Note [Preserve send tensor].
        global _preserved_send_tensor
        _preserved_send_tensor = tensors[-1]
        return _ret_work(tensors)

    @cuda_use_nccl
    def recv(self, out_tensors, src_rank, tag=0):
        # src_rank is group rank
        src_rank = dist.get_global_rank(self, src_rank)
        rank = dist.get_rank()
        send_group = [rank, src_rank] if src_rank > rank else [src_rank, rank]
        groups = _get_full_group(send_group)
        tensors = [torch.zeros_like(t) for t in out_tensors]
        xm.all_reduce('sum', tensors, groups=groups, pin_layout=False)
        for i, tensor in enumerate(tensors):
            out_tensors[i].copy_(tensor)
        return _ret_work(out_tensors)

    @cuda_use_nccl
    def recv_anysource(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def monitored_barrier(self, *args):
        raise NotImplementedError

    @cuda_use_nccl
    def Options(self, *args):
        raise NotImplementedError


# -------------------------------------
# Override torch.distributed.new_group.
# -------------------------------------
_orig_new_group_fn = dist.new_group


def new_lazy_process_group(ranks=None,
                           timeout=dist.default_pg_timeout,
                           backend=None,
                           pg_options=None):
    pg = _orig_new_group_fn(
        ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)
    if isinstance(pg, ProcessGroupLazy) and ranks is not None:
        world_pg = dist.group.WORLD
        if not isinstance(world_pg, ProcessGroupLazy):
            raise RuntimeError(
                'lazy backend requires the default ProcessGroup to be '
                'a ProcessGroupLazy')

        if isinstance(ranks, range):
            ranks = list(ranks)

        if ranks == list(range(world_pg.size())):
            pg._mesh = [ranks]
        elif len(ranks) == 1:
            if ranks[0] not in range(world_pg.size()):
                raise ValueError(
                    'Given ranks is out of range: '
                    f'World size: {world_pg.size()}, ranks: {ranks}')
            pg._mesh = [[r] for r in range(world_pg.size())]
        elif len(ranks) < world_pg.size() and len(ranks) > 1:
            pg._mesh = _get_full_group(ranks)
        else:
            logger.warn(
                f'Can\'t infer process group mesh from given ranks "{str(ranks)}". '
                'The process group will use the entire world as its collective comm group.'
            )

    return pg


dist.new_group = new_lazy_process_group
# -------------------------------------------
# End overriding torch.distributed.new_group.
# -------------------------------------------
