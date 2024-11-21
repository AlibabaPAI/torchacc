import torch

from torchacc.config import Config
from torchacc.dist import ParallelModule, DataParallel, FullyShardedDataParallel, PipelineParallel, SpmdFullyShardedDataParallel


class DistributedParallel(ParallelModule):
    """Enable different distributed parallel for the model.

    Args:
        model (torch.nn.Module): The model to enable different parallel strategies.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__(model, config, **kwargs)
        self._module = None
        if self.has_pp:
            self._module = PipelineParallel(model, self.config, **kwargs)

        fsdp_wrapper = SpmdFullyShardedDataParallel if self.spmd_fsdp else FullyShardedDataParallel
        if self.has_fsdp:
            if self._module is None:
                self._module = fsdp_wrapper(model, self.config, **kwargs)
            else:
                model = self._module._get_underlay_model()
                model = fsdp_wrapper(model, self.config, **kwargs)
                self._module._update_underlay_model(model)

        need_wrap_dp = False
        if config.is_eager_backend():
            need_wrap_dp = self.has_dp and not self.has_fsdp
        elif config.is_lazy_backend():
            need_wrap_dp = self.has_dp and not self.has_tp

        if need_wrap_dp:
            if self._module is None:
                self._module = DataParallel(model, self.config, **kwargs)
            else:
                module = self._module._get_underlay_model()
                module = DataParallel(model, self.config, **kwargs)
                self._module._update_underlay_model(module)

        if self._module is None:
            self._module = module

    def _get_underlay_model(self):
        if isinstance(self._module, ParallelModule):
            return self._module._get_underlay_model()
        return self._module

    def _update_underlay_model(self, module: torch.nn.Module):
        if isinstance(self._module, ParallelModule):
            self._module._update_underlay_model(module)
        else:
            self._module = module

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self._module, "clip_grad_norm_"):
            self._module.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self._module.parameters(),
                                           max_grad_norm)

    def forward(self, *args, output_fn=None, **kwargs):
        if not self.has_pp and output_fn is not None:
            raise ValueError(
                "output_fn is only supported for pipeline parallel")
        if output_fn:
            kwargs["output_fn"] = output_fn
        return self._module(*args, **kwargs)

    def forward_backward(self, *args, output_fn=None, **kwargs):
        if not self.has_pp:
            raise NotImplementedError(
                "forward_backward is only supported for pipeline parallel.")
        assert isinstance(self._module, PipelineParallel)
        return self._module.forward_backward(
            *args, output_fn=output_fn, **kwargs)
