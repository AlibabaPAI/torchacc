import functools

import numpy as np
import torch
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2

from torchacc.core import lazy_device
from torchacc.config import Config
from torchacc.dist import ParallelModule
import torchacc.utils.checkpoint as checkpoint
import torchacc.utils.utils as utils


class SpmdFullyShardedDataParallel(ParallelModule):
    """Implementation of SPMD based fully sharded data parallel.

    Args:
        model (torch.nn.Module): The model to enable fully sharded data parallel.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__(model, config)

        self.shard_output_callable = config.dist.fsdp.shard_output_callable
        self.model = self.fsdp(model, config)

    def _get_underlay_model(self):
        return self.model

    def _get_mesh(self, mesh_shape, device_ids=None, axis_names=None):
        assert type(mesh_shape) is tuple, 'mesh_shape must be Tuple[int]'
        n_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(n_devices))
        return xs.Mesh(device_ids, mesh_shape, axis_names)

    def fsdp(self, model: torch.nn.Module, config: Config):
        layer_cls = set()
        for name in config.dist.fsdp.wrap_layer_cls:
            cls = utils.get_module_class_from_name(model, name)
            assert cls, f"class {name} in fsdp.wrap_layer_cls not found in model"
            layer_cls.add(cls)
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            # Transformer layer class to wrap
            transformer_layer_cls=layer_cls,
        )

        auto_wrapper_callable = None
        if config.memory.gc and (config.memory.gc_cls
                                 == config.dist.fsdp.wrap_layer_cls):
            gc_cnt = config.memory.gc_cnt

            def auto_wrapper_callable(m, *args, **kwargs):
                nonlocal gc_cnt
                if gc_cnt is None:
                    m = checkpoint.checkpoint_module(m)
                elif gc_cnt > 0:
                    m = checkpoint.checkpoint_module(m)
                    gc_cnt -= 1
                return FSDPv2(m, *args, **kwargs)

        mesh = self._get_mesh((self.mesh.get_fsdp_num(), 1), None,
                              ('fsdp', 'tensor'))

        model = FSDPv2(
            model.to(lazy_device()),
            mesh,
            shard_output=self.shard_output_callable,
            auto_wrap_policy=auto_wrap_policy,
            auto_wrapper_callable=auto_wrapper_callable)
        return model

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self.model, "clip_grad_norm_"):
            self.model.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
