import os
from dataclasses import dataclass, field

from . import dist, ops
from .config import Config
from .core import (AsyncLoader, amp, fetch_gradients, is_lazy_device,
                   is_lazy_tensor, lazy_device, sync)
from .core.accelerate_hf_trainer import accelerate_hf_trainer
from .core.dynamic import mark_dynamic
from .llm.qwen_patch import patch_qwen_model
from .utils import decompose, patch, import_utils
from .version import __version__

from .accelerate import accelerate  # isort: skip

_XLA_COORDINATOR_PORT_OFFSET = 10
_global_context = None

if import_utils.is_torch_xla_available():
    from torch_xla.amp import syncfree
    from .core import mark_step, save, send_cpu_data_to_device


@dataclass
class GlobalContext:
    config: Config = field(default_factory=Config)
    mesh: dist.Mesh = field(default_factory=dist.Mesh)
    python_dispatcher = None


def get_global_context():
    global _global_context
    if _global_context is None:
        _global_context = GlobalContext()

    return _global_context


def _set_env():
    # Avoid all processes running on GPU:0 in distirbuted training.
    # Related PR: https://github.com/pytorch/xla/pull/6208
    os.environ['PJRT_LOCAL_PROCESS_RANK'] = os.getenv('LOCAL_RANK', '0')
    os.environ['PJRT_DEVICE'] = os.getenv('PJRT_DEVICE', 'CUDA')
    os.environ['PJRT_ALLOCATOR_FRACTION'] = os.getenv('PJRT_ALLOCATOR_FRACTION',
                                                      '0.95')
    os.environ['XLA_IR_SHAPE_CACHE_SIZE'] = os.getenv('XLA_IR_SHAPE_CACHE_SIZE',
                                                      '100000000')
    # Assign the XLA_COORDINATOR_PORT environment variable by adding an offset
    # to MASTER_PORT to avoid XLA_COORDINATOR_PORT conflict.
    if 'XLA_COORDINATOR_PORT' not in os.environ and 'MASTER_PORT' in os.environ:
        os.environ['XLA_COORDINATOR_PORT'] = str(
            int(os.getenv('MASTER_PORT')) + _XLA_COORDINATOR_PORT_OFFSET)
    # OpenXLA attempts to reuse as many as parameter buffers that are marked as
    # donor buffer. However, during the BufferInsertion pass, too many copy
    # operations may be inserted, potentially causing excessive peak memory
    # usage. Using the SetupAlias method by default by setting XLA_USING_BUFFER_DONOR
    # to false.
    # TODO: The method of alias still has some issues need to fix.
    #if 'XLA_USING_BUFFER_DONOR' not in os.environ:
    #    os.environ['XLA_USING_BUFFER_DONOR'] = '0'

    # Use the value of the upper bound for shape comparison.
    # TODO: This is experimental. It is necessary to remove this environment variable
    # to support users to directly perform `if tensor.shape[0] > 10`.
    os.environ['USE_BOUND_FOR_SHAPE_COMPARE'] = os.getenv(
        'USE_BOUND_FOR_SHAPE_COMPARE', '1')

    if 'CUDA_DEVICE_MAX_CONNECTIONS' not in os.environ:
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    xla_flags = os.getenv("XLA_FLAGS", "")
    default_flags = {
        "--xla_gpu_enable_latency_hiding_scheduler":
            "true",
        "--xla_gpu_enable_async_all_gather":
            "true",
        "--xla_gpu_enable_async_collective_permute":
            "true",
        "--xla_gpu_enable_async_reduce_scatter":
            "true",
        "--xla_multiheap_size_constraint_per_heap":
            "4831838208",
        # "--xla_gpu_enable_flash_attention":
        #     "true",
        "--xla_gpu_enable_triton_softmax_fusion":
            "false",
        "--xla_gpu_force_compilation_parallelism":
            "32",
        "--xla_disable_hlo_passes":
            "rematerialization,gpu-convert-async-collectives-to-sync,triton-autotuner,all-gather-combiner,reduce-scatter-combiner,all-reduce-combiner,",
        # This config controls the memory limit for the scheduling of latency hiding scheduler (LHS),
        # you can increase it to allow the scheduling of LHS to prioritize optimizing the overlap
        # between computation and communication.
        "--xla_gpu_memory_limit_slop_factor":
            "100",
        "--xla_gpu_enable_triton_gemm":
            "false",
        "--xla_gpu_graph_level":
            "0",
        "--xla_gpu_enable_highest_priority_async_stream":
            "true",
        "--xla_gpu_all_reduce_combine_threshold_bytes":
            "1073741824",
        "--xla_gpu_all_gather_combine_threshold_bytes":
            "1073741824",
        "--xla_gpu_reduce_scatter_combine_threshold_bytes":
            "1073741824",
        "--xla_gpu_enable_pipelined_all_gather":
            "true",
        "--xla_gpu_enable_pipelined_reduce_scatter":
            "true",
        "--xla_gpu_enable_pipelined_all_reduce":
            "true",
        "--xla_gpu_enable_while_loop_double_buffering":
            "true",
        "--xla_gpu_enable_all_gather_combine_by_dim":
            "false",
        "--xla_gpu_enable_reduce_scatter_combine_by_dim":
            "false"
    }

    for flag, value in default_flags.items():
        if flag not in xla_flags:
            xla_flags += f" {flag}={value}"
    os.environ["XLA_FLAGS"] = xla_flags


patch.patch_fa()
decompose.replace_decompose()
_set_env()
