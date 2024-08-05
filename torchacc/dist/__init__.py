import os

import torch_xla.core.xla_model as xm

# register lazy backend
from . import backend

from .mesh import Mesh

from .parallel_module import ParallelModule
from .dp import DataParallel
from .fsdp import FullyShardedDataParallel
from .spmd_fsdp import SpmdFullyShardedDataParallel
from .pp import PipelineParallel
from .distributed_parallel import DistributedParallel

from . import fsdp, pp, tp

BACKEND_NAME = backend._BACKEND_NAME


def world_size():
    return int(os.getenv('WORLD_SIZE', '1'))


def rank():
    return int(os.getenv('RANK', '0'))


def local_rank():
    return int(os.getenv('LOCAL_RANK', 0))
