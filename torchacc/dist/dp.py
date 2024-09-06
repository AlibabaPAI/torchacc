import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchacc as ta
from torchacc.dist import ParallelModule
from torchacc.config import Config
import torchacc.utils.utils as utils


class DataParallel(ParallelModule):
    """Implementation of data parallel.

    Args:
        model (torch.nn.Module): The model to enable data parallel.
        config (torchacc.Config): Configuration for TorchAcc.
    """

    def __init__(self, model: torch.nn.Module, config: Config, **kwargs):
        super().__init__(model, config, **kwargs)

        if config.is_eager_backend():
            # for torch DDP, we need to move the model to cuda first.
            model = model.to(self.device)
            self.model = DDP(
                model,
                device_ids=[self.device],
                process_group=self.mesh.get_dp_proc_group())
        else:
            self.model = model
        self.is_lazy_backend = config.is_lazy_backend()
        self.is_eager_backend = config.is_eager_backend()

        self.dp_size = self.mesh.get_dp_num()
        self.dp_proc_group = self.mesh.get_dp_proc_group()

        self._post_backward_callback_queued = False

    @torch.no_grad()
    def _post_backward_callback(self):
        gradients = []
        for p in self.model.parameters():
            if p.grad is not None:
                gradients.append(p.grad)
        with dist._coalescing_manager(group=self.dp_proc_group):
            for grad in gradients:
                dist.all_reduce(grad, group=self.dp_proc_group)
        for grad in gradients:
            grad.mul_(1.0 / self.dp_size)

    def _register_pre_backward_hooks(self, outputs):
        if not torch.is_grad_enabled():
            return outputs

        self._post_backward_callback_queued = False

        def _pre_backward_hook(t_grad: torch.Tensor) -> None:
            if not self._post_backward_callback_queued:
                self._post_backward_callback_queued = True
                Variable._execution_engine.queue_callback(
                    self._post_backward_callback)

        def _register_hook(t: torch.Tensor) -> torch.Tensor:
            if t.requires_grad:
                t.register_hook(_pre_backward_hook)
            return t

        # Attach hooks to Tensor outputs.
        outputs = utils.apply_to_tensors(_register_hook, outputs)
        return outputs

    def _get_underlay_model(self):
        if self.is_eager_backend:
            return self.model.module
        return self.model

    def _update_underlay_model(self, model: torch.nn.Module):
        if self.is_eager_backend:
            raise NotImplementedError(
                "Cannot update underlay model in DataParallel for eager backend."
            )
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if self.is_lazy_backend:
            outputs = self._register_pre_backward_hooks(outputs)
        return outputs
