import inspect
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm

import torchacc as ta
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.patch as patch
import torchacc.utils.trace as trace

_TORCHDISTX_AVAIL = bool(int(os.environ.get('LOW_CPU_MEM_USAGE', "0")))
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False


def _setup_env(config: ta.Config) -> None:
    xla_flags = os.getenv("XLA_FLAGS", "")

    if config.dist.dp.size > 1 and not config.dist.tp.size > 1:
        if "--xla_gpu_enable_all_reduce_splitter" not in xla_flags:
            xla_flags += " --xla_gpu_enable_all_reduce_splitter=true"

    if config.dist.fsdp.size > 1:
        if "--xla_gpu_enable_all_reduce_splitter" not in xla_flags:
            xla_flags += " --xla_gpu_enable_all_reduce_splitter=false"

    if config.dist.pp.size == 1:
        if "--xla_gpu_enable_all_reduce_splitter" not in xla_flags:
            xla_flags += " --xla_gpu_enable_all_reduce_splitter=true"
    os.environ["XLA_FLAGS"] = xla_flags


def broadcast_master_param(model: torch.nn.Module, config: ta.Config) -> None:
    # DP
    if config.dist.dp.size > 1 and config.dist.dp.size == ta.dist.world_size():
        xm.broadcast_master_param(model)
    # TODO: support DP+FSDP, DP+PP, etc.


def accelerate(
    model: torch.nn.Module,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    config: ta.Config = ta.Config()
) -> Union[Tuple[torch.nn.Module, torch.utils.data.DataLoader],
           torch.nn.Module]:
    """Optimize the model using TorchAcc, the optimization strategy is specified by the config.

    Args:
        model (torch.nn.Module): The model to be optimized.
        dataloader (Optional[torch.utils.data.DataLoader]): The dataloader to be optimized.
        config (torchacc.Config): Configuration for optimization.

    Returns:
        Union[Tuple[torch.nn.Module, torch.utils.data.DataLoader], torch.nn.Module]: Optimized model and dataloader (if provided).
    """
    config.validate()
    ta.get_global_context().config = config
    _setup_env(config)

    if config.is_distributed_parallel():
        ta.dist.init_process_group(config)
        ta.dist.init_nccl_context(config)

    if config.is_eager_backend():
        if dist.is_initialized():
            device = dist.get_rank() % torch.cuda.device_count()
        else:
            device = torch.cuda.current_device()
        torch.cuda.set_device(device)
        print(f"setting device to cuda:{device}")
    else:
        device = ta.lazy_device()

    if dataloader and config.is_lazy_backend():
        dataloader = ta.AsyncLoader(
            dataloader,
            device,
            buckets=config.dataloader.buckets,
            max_length=config.dataloader.max_length,
            num_buckets=config.dataloader.num_buckets,
            pad_value_dict=config.dataloader.pad_value_dict)

    # compute
    if config.compute.acc_scaled_dot_attn:
        torch.nn.functional.scaled_dot_product_attention = ta.ops.scaled_dot_product_attention

    if config.compute.acc_llama:
        ta.ops.apply_liger_kernel_to_llama()

    # replace the optimizer and grad scaler with the syncfree optimizer and the torchacc grad scaler
    if config.compute.fp16 and config.is_lazy_backend():
        patch.patch_amp()

    # tracing
    orig_forward_sig = None
    if config.is_tracing_enabled():
        ta.dist.pp.preprocess_config(config, model)
        orig_forward_sig = inspect.signature(model.forward)
        model = trace.trace(model, config.dist.pp.input_names)

    # distributed parallel
    if config.is_distributed_parallel():
        model = ta.dist.DistributedParallel(
            model, config, orig_forward_sig=orig_forward_sig)

        m = model._get_underlay_model()
        is_torchdistX_deferred_init = (
            _TORCHDISTX_AVAIL and
            any(fake.is_fake(param) for param in m.parameters()))
        if is_torchdistX_deferred_init:
            deferred_init.materialize_module(m)

    # gradient checkpoint
    if config.memory.gc and config.memory.gc_cls is not None:
        if config.dist.fsdp.size > 1:
            # full gc has already been done by fsdp
            if len(config.memory.gc_cls) > 0 and \
                    (config.memory.gc_cls != config.dist.fsdp.wrap_layer_cls):
                underlay_model = model._get_underlay_model()
                underlay_model = checkpoint.gradient_checkpoint(
                    underlay_model, config.memory.gc_cls)
        else:
            if isinstance(model, ta.dist.ParallelModule):
                underlay_model = model._get_underlay_model()
                underlay_model = checkpoint.gradient_checkpoint(
                    underlay_model, config.memory.gc_cls)
                model._update_underlay_model(underlay_model)
            else:
                model = checkpoint.gradient_checkpoint(model,
                                                       config.memory.gc_cls)

    model = model.to(device)

    # broadcast parameters
    if config.is_lazy_backend():
        broadcast_master_param(model, config)

    if not hasattr(model, "device"):
        model.device = device

    return (model, dataloader) if dataloader else model
