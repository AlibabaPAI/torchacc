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

        self.model = None
        if self.has_pp:
            self.model = PipelineParallel(model, self.config, **kwargs)

        fsdp_wrapper = SpmdFullyShardedDataParallel if self.spmd_fsdp else FullyShardedDataParallel
        if self.has_fsdp:
            if self.model is None:
                self.model = fsdp_wrapper(model, self.config, **kwargs)
            else:
                model = self.model._get_underlay_model()
                model = fsdp_wrapper(model, self.config, **kwargs)
                self.model._update_underlay_model(model)

        if self.has_dp and not self.has_tp:
            if self.model is None:
                self.model = DataParallel(model, self.config, **kwargs)
            else:
                model = self.model._get_underlay_model()
                model = DataParallel(model, self.config, **kwargs)
                self.model._update_underlay_model(model)

        if self.model is None:
            self.model = model

    def _get_underlay_model(self):
        if isinstance(self.model, ParallelModule):
            return self.model._get_underlay_model()
        return self.model

    def _update_underlay_model(self, model: torch.nn.Module):
        if isinstance(self.model, ParallelModule):
            self.model._update_underlay_model(model)
        else:
            self.model = model

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self.model, "clip_grad_norm_"):
            self.model.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)

    def forward(self, *args, output_fn=None, **kwargs):
        if not self.has_pp and output_fn is not None:
            raise ValueError(
                "output_fn is only supported for pipeline parallel")
        if output_fn:
            kwargs["output_fn"] = output_fn
        return self.model(*args, **kwargs)

    def forward_backward(self, *args, output_fn=None, **kwargs):
        if not self.has_pp:
            raise NotImplementedError(
                "forward_backward is only supported for pipeline parallel.")
        assert isinstance(self.model, PipelineParallel)
        return self.model.forward_backward(*args, output_fn=output_fn, **kwargs)
