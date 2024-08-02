from itertools import product
from typing import List, Optional

import numpy as np
import torch.distributed as dist

_CONTEXT_PARALLEL_INITIALIZED = False
_INTRA_CP_GROUP = None
_INTER_CP_GROUP = None
_CONTEXT_PARALLEL_GROUP = None

__all__ = [
    "initialize_context_parallel",
    "get_inter_cp_process_group",
    "get_intra_cp_process_group",
    "get_context_parallel_group",
]


def initialize_parallel_group(sizes: List[int]):
    world_size = dist.get_world_size()
    assert world_size == np.prod(sizes)
    global_rank = dist.get_rank()
    groups = np.asarray(range(world_size)).reshape(*sizes)

    process_groups = []
    for dim in range(len(sizes)):
        prev_size = sizes[dim]
        sizes[dim] = 1
        for combination in product(*(range(size) for size in sizes)):
            combination = list(combination)
            combination[dim] = slice(None)
            ranks = groups[tuple(combination)].tolist()
            group = dist.new_group(ranks)
            if global_rank in ranks:
                process_groups.append(group)
        sizes[dim] = prev_size

    return process_groups


def initialize_context_parallel(
    context_parallel_size: int,
    intra_parallel_size: Optional[int] = None,
):
    """Initialize the context parallel groups.

    Args:
        context_parallel_size (int): The size of context parallel. The world size needs to be
            divisible by the size of context parallel.
        intra_parallel_size (int, optional): The size of intra-node context parallel. If `None`,
            it is the same as context_parallel_size. The size of context parallel needs to be
            divisible by the size of intra-node context parallel.
    """
    assert dist.is_initialized()
    global _CONTEXT_PARALLEL_INITIALIZED
    assert not _CONTEXT_PARALLEL_INITIALIZED, "initialize_context_parallel calls twice"
    world_size = dist.get_world_size()

    if world_size % context_parallel_size != 0:
        raise ValueError(f"The world size {world_size} needs to be divisible by the size" \
                         f" of context parallel {context_parallel_size}.")
    if intra_parallel_size is None:
        intra_parallel_size = context_parallel_size
    if context_parallel_size % intra_parallel_size != 0:
        raise ValueError(f"The size of context parallel {context_parallel_size} needs to be" \
                         f" divisible by the size of intra-node context parallel " \
                         f"{intra_parallel_size}.")
    inter_parallel_size = context_parallel_size // intra_parallel_size

    data_parallel_size = world_size // context_parallel_size

    global _INTRA_CP_GROUP, _INTER_CP_GROUP, _CONTEXT_PARALLEL_GROUP
    assert _INTRA_CP_GROUP is None, "Intra context parallel group is already initialized"
    assert _INTER_CP_GROUP is None, "Inter context parallel group is already initialized"
    assert _CONTEXT_PARALLEL_GROUP is None, "Context parallel group is already initialized"
    sizes = [data_parallel_size, inter_parallel_size, intra_parallel_size]
    process_groups = initialize_parallel_group(sizes)
    _INTER_CP_GROUP = process_groups[1]
    _INTRA_CP_GROUP = process_groups[2]

    if intra_parallel_size == 1:
        _CONTEXT_PARALLEL_GROUP = _INTER_CP_GROUP
    elif inter_parallel_size == 1:
        _CONTEXT_PARALLEL_GROUP = _INTRA_CP_GROUP
    else:
        sizes = [data_parallel_size, context_parallel_size]
        process_groups = initialize_parallel_group(sizes)
        _CONTEXT_PARALLEL_GROUP = process_groups[1]

    _CONTEXT_PARALLEL_INITIALIZED = True


def get_inter_cp_process_group():
    """The process group of inter-node context parallel."""
    global _INTER_CP_GROUP, _CONTEXT_PARALLEL_INITIALIZED
    assert _CONTEXT_PARALLEL_INITIALIZED, "Context parallel group is not initialized"
    return _INTER_CP_GROUP


def get_intra_cp_process_group():
    """The process group of intra-node context parallel."""
    global _INTRA_CP_GROUP, _CONTEXT_PARALLEL_INITIALIZED
    assert _CONTEXT_PARALLEL_INITIALIZED, "Context parallel group is not initialized"
    return _INTRA_CP_GROUP


def get_context_parallel_group():
    """The process group of context parallel."""
    global _CONTEXT_PARALLEL_GROUP, _CONTEXT_PARALLEL_INITIALIZED
    assert _CONTEXT_PARALLEL_INITIALIZED, "Context parallel group is not initialized"
    return _CONTEXT_PARALLEL_GROUP
