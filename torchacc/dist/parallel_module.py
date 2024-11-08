from abc import ABC, abstractmethod

import torch
from torchacc.core import lazy_device
from torchacc.config import Config


class ParallelModule(torch.nn.Module, ABC):
    """Base class for different parallel strategies.

    Args:
        model (torch.nn.Module): The model to enable different parallel strategies.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__()

        config.validate()

        self._config = config
        self._device = lazy_device() if config.is_lazy_backend(
        ) else torch.cuda.current_device()

        self.mesh = self._config.get_mesh()
        self.global_rank = self.mesh.get_global_rank()

        self.has_dp = self._config.dist.dp.size > 1
        self.has_tp = self._config.dist.tp.size > 1
        self.has_pp = self._config.dist.pp.size > 1
        self.has_fsdp = self._config.dist.fsdp.size > 1
        self.spmd_fsdp = self._config.dist.fsdp.use_spmd

    @property
    def config(self) -> Config:
        return self._config

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def _get_underlay_model(self):
        pass

    def _update_underlay_model(self, model: torch.nn.Module):
        raise NotImplementedError("Not yet implemented.")

    def clip_grad_norm_(self, max_grad_norm):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)

    def forward_backward(self, *args, output_fn=None, **kwargs):
        """This function will perform forward and backward, and return the averaged loss.
           This function is only supported for pipeline parallel and will split the input
           batch into config.dist.pp.num_micro_batches micro batches for pipelining.

        Args:
            *args: input args of the model forward.
            output_fn: A function that processes the model's output and returns a loss for backward.
                The first argument of this function is the return value of the model's forward, and
                the subsequent arguments can be arguments from kwargs. It is worth noting that tensors
                have already been split along the batchsize dimension.
            **kwargs: input kwargs of the model forward and optional input arguments of the output_fn.

        Returns:
            The loss averaged according to num_micro_batches.
        """
        raise NotImplementedError(
            "forward_backward is only supported for pipeline parallel.")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._get_underlay_model(), name)
