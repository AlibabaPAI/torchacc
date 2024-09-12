import inspect
import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch_xla
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

_COMMUNICATION_INITED = False


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


def init_comm(config: ta.Config) -> None:
    global _COMMUNICATION_INITED
    if _COMMUNICATION_INITED:
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
            world_size = ta.dist.world_size()
            device_ids = np.array(list(range(world_size)))
            tp_mesh = ta.dist.tp.Mesh(device_ids, (1, world_size))
            a = torch.zeros(1, world_size).to(device)
            b = torch.zeros(world_size, 2).to(device)
            ta.dist.tp.mark_sharding(a, tp_mesh, (None, 1))
            ta.dist.tp.mark_sharding(b, tp_mesh, (1, None))
            c = torch.einsum('ij,jk->ik', a, b)
            # SPMD will insert all reduce here
            ta.dist.tp.mark_sharding(c, tp_mesh, (None, None))
            ta.sync()

    _COMMUNICATION_INITED = True


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
    device = ta.lazy_device()
    config.validate()
    ta.get_global_context().config = config
    _setup_env(config)
    init_comm(config)

    if dataloader:
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

    # replace the optimizer and grad scaler with the syncfree optimizer and the torchacc grad scaler
    if config.compute.fp16:
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
    broadcast_master_param(model, config)

    if not hasattr(model, "device"):
        model.device = device

    return (model, dataloader) if dataloader else model
