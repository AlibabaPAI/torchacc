import os

import numpy as np
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.core.xla_model as xm

import torchacc as ta

# register lazy backend
from . import backend

from .mesh import Mesh

from .parallel_module import ParallelModule  # isort: skip
from .dp import DataParallel
from .fsdp import FullyShardedDataParallel
from .spmd_fsdp import SpmdFullyShardedDataParallel
from .pp import PipelineParallel
from .distributed_parallel import DistributedParallel

from . import fsdp, pp, tp

BACKEND_NAME = backend._BACKEND_NAME
EAGER_BACKEND_NAME = backend._EAGER_BACKEND_NAME
_NCCL_CONTEXT_INITED = False


def world_size():
    return int(os.getenv('WORLD_SIZE', '1'))


def rank():
    return int(os.getenv('RANK', '0'))


def local_rank():
    return int(os.getenv('LOCAL_RANK', 0))


def init_process_group(config) -> None:
    backend = EAGER_BACKEND_NAME if config.is_eager_backend() else BACKEND_NAME
    if dist.is_initialized():
        assert dist.get_backend() == backend, "The backend for initializing the distributed" \
            f" process group should be {backend}."
    else:
        dist.init_process_group(backend=backend)
        # do not use dist.barrier() for lazy backend here,
        # since lazy backend will use extra nccl group to do barrier
        if not config.is_lazy_backend():
            dist.barrier()


def init_nccl_context(config) -> None:
    global _NCCL_CONTEXT_INITED
    if _NCCL_CONTEXT_INITED:
        return

    if config.is_eager_backend():
        _NCCL_CONTEXT_INITED = True
        return

    device = ta.lazy_device()
    if config.dist.pp.size > 1:
        # Initialize the communication for send/recv to prevent the first recv in later stages
        # from waiting for an excessive amount of time and potentially causing a timeout.
        mesh = config.get_mesh()
        tmp = torch.tensor(0.0, requires_grad=False).to(device)
        if not mesh.is_last_stage():
            dst_rank = mesh.stage_to_global(stage_id=mesh.get_stage_id() + 1)
            dist.send(tmp, dst_rank)
        if not mesh.is_first_stage():
            src_rank = mesh.stage_to_global(stage_id=mesh.get_stage_id() - 1)
            dist.recv(tmp, src_rank)
        # Execute the above computation graph first to ensure that the execution order of send/recv
        # does not interfere with the subsequent all reduce operations.
        ta.sync()

        if torch_xla.runtime.is_spmd():
            # Initialize the communication for collective operations (such as collective permute, all reduce) to
            # prevent hangs in PP where only some devices participate in the communication.
            world_size = world_size()
            device_ids = np.array(list(range(world_size)))
            tp_mesh = tp.Mesh(device_ids, (1, world_size))
            a = torch.zeros(1, world_size).to(device)
            b = torch.zeros(world_size, 2).to(device)
            tp.mark_sharding(a, tp_mesh, (None, 1))
            tp.mark_sharding(b, tp_mesh, (1, None))
            c = torch.einsum('ij,jk->ik', a, b)
            # SPMD will insert all reduce here
            tp.mark_sharding(c, tp_mesh, (None, None))
            ta.sync()

    _NCCL_CONTEXT_INITED = True


def rendezvous(tag, payload=b'', replicas=[]):
    """Waits for all the mesh clients to reach the named rendezvous.
  We use the rendezvous api of xla directly.
  
  Args:
    tag (string): The name of the rendezvous to join.
    payload (bytes, optional): The payload to be sent to the rendezvous.
    replicas (list, int): The replica ordinals taking part of the rendezvous.
      Empty means all replicas in the mesh.
      Default: []

  Returns:
    The payloads exchanged by all the other cores, with the payload of core
    ordinal `i` at position `i` in the returned tuple.
  """
    return xm.rendezvous(tag, payload, replicas)
