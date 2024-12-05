import inspect
import os
from typing import Optional, Tuple, Union

import torch
import torch.distributed as dist

import torchacc as ta
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.patch as patch
import torchacc.utils.trace as trace

_TORCHDISTX_AVAIL = bool(int(os.environ.get('LOW_CPU_MEM_USAGE', "0")))
try:
    from torchdistx import deferred_init, fake  # type: ignore[import]
except ImportError:
    _TORCHDISTX_AVAIL = False

from torchacc.utils.import_utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


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

def find_decoder_layers(model: torch.nn.Module) -> list:
    """
    遍历模型及其子模型的所有类名，返回名字中包含 'DecoderLayer' 的类名。

    :param model: 输入的 PyTorch 模型
    :return: 包含 'DecoderLayer' 的类名列表
    """
    decoder_layers = set()

    def traverse_module(module: torch.nn.Module):
        # 获取模块的类名
        class_name = module.__class__.__name__
        if 'DecoderLayer' in class_name:
            decoder_layers.add(class_name)
            return

        # 递归遍历子模块
        for child in module.children():
            traverse_module(child)

    # 从根模块开始遍历
    traverse_module(model)

    return decoder_layers

def find_modulelist_classes(model: torch.nn.Module) -> list:
    """
    遍历模型中的所有 ModuleList，返回这些 ModuleList 中包含的类名。

    :param model: 输入的 PyTorch 模型
    :return: ModuleList 中包含的类名列表
    """
    modulelist_classes = set()

    def traverse_module(module: torch.nn.Module):
        # 遍历当前模块的所有属性
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ModuleList):
                # 如果是 ModuleList，遍历其中的模块
                for i, sub_module in enumerate(child):
                    class_name = sub_module.__class__.__name__
                    modulelist_classes.add(class_name)
            else:
                # 递归遍历子模块
                traverse_module(child)

    # 从根模块开始遍历
    traverse_module(model)

    return modulelist_classes

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
        if hasattr(model, "config"):
            model.config.use_cache = False
        elif hasattr(model, "model"):
            model.model.config.use_cache = False
        if dist.is_initialized():
            device = dist.get_rank() % torch.cuda.device_count()
        else:
            device = torch.cuda.current_device()
        torch.cuda.set_device(device)
    else:
        device = ta.lazy_device()

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

    if not config.compute.disable_kernel_patches and config.is_eager_backend() and not config.use_dynamo:
        ta.ops.apply_liger_kernel()

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
        if "auto" in config.dist.fsdp.wrap_layer_cls:
            decoder_layers = find_modulelist_classes(model)
            assert len(decoder_layers) == 1
            config.dist.fsdp.wrap_layer_cls = decoder_layers
            if ta.dist.rank() == 0:
                ta.utils.logger.info(f"Auto wrap decoder layer: {decoder_layers}")
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
        if config.dist.fsdp.size > 1 and config.is_lazy_backend():
            # full gc has already been done by xla fsdp
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
                # PyTorch checkpoint modify model inplace
                if not config.is_eager_backend():
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

    if config.use_dynamo:
        try:
            import transformers.modeling_flash_attention_utils as modeling_flash_attention_utils
            torch.compiler.disable(modeling_flash_attention_utils._flash_attention_forward, recursive=False)
        except ImportError:
            pass
        except AttributeError:
            pass

    if config.use_dynamo and config.dist.fsdp.size == 1:
        model = torch.compile(model, backend="openxla")

        if config.is_eager_backend():
            torch._dynamo.disallow_in_graph(torch.nn.functional.scaled_dot_product_attention)

            stream = torch_xla._XLAC._get_stream_for_cuda_device(device)
            stream = 1 if stream == 0 else stream
            assert stream is None or type(stream) is int
            external_stream = torch.cuda.ExternalStream(stream)
            torch.cuda.set_stream(external_stream)
    
    # if dist.get_rank() == 0:
    #     import logging
    #     # torch._logging.set_logs(graph_breaks=True, recompiles=True, graph=True, graph_code=True)
    #     torch._logging.set_logs(all=logging.DEBUG)
        # torch._logging.set_logs(dynamo=logging.DEBUG)

    return (model, dataloader) if dataloader else model
