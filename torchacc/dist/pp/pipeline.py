import inspect
import torch
import torch.fx as fx

from torchacc.dist import ParallelModule
from torchacc.dist.pp import utils
from torchacc.dist.pp.executor import PipeExecutor
from torchacc.config import Config
from torchacc.utils.logger import logger
from torchacc.utils import trace


def preprocess_config(config: Config, model: torch.nn.Module):
    if config.dist.pp.size > 1:
        if any([isinstance(p, str) for p in config.dist.pp.split_points]):
            return
        points = set(config.dist.pp.split_points)
        new_points = []
        for name, m in model.named_modules():
            if m in points:
                new_points.append(name)
        assert len(new_points) == len(config.dist.pp.split_points), "Some values in " \
            "config.dist.pp.split_points were not found in the model.named_modules()."
        config.dist.pp.split_points = new_points


class PipelineParallel(ParallelModule):

    def __init__(self,
                 model: torch.nn.Module,
                 config: Config,
                 orig_forward_sig: inspect.Signature = None,
                 **kwargs):
        super().__init__(model, config)
        self.num_stages = self.mesh.get_pp_num()
        self.stage_id = self.mesh.get_stage_id()

        if isinstance(model, fx.GraphModule):
            assert orig_forward_sig, "When using the fx GraphModule, needs to provide the inspect.Signature " \
                "of the original model's forward."
        else:
            preprocess_config(config, model)
            orig_forward_sig = inspect.signature(model.forward)
            model = trace.trace(model, config.dist.pp.input_names)
        self.orig_forward_sig = orig_forward_sig

        assert self.num_stages == len(config.dist.pp.split_points) + 1
        self._set_stage_model(model, config.dist.pp.split_points)

        self.executor = PipeExecutor(
            self.model,
            self.config.dist.pp,
            self.mesh,
            self.orig_forward_sig,
            self.post_process,
            self.input_tensor_attr,
            self.device,
        )

        self.forward_warn = True

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _set_stage_model(self, fx_model, split_points):
        split_model, qualname_map = utils.split(fx_model, None, split_points,
                                                self.num_stages)
        self.qualname_map = qualname_map
        if self.mesh.get_global_rank() == 0:
            # we only want to print the structure of module
            logger.info(
                f"After splitting: {super(torch.nn.Module, split_model).__str__()}"
            )

        all_models = list(split_model.named_children())
        assert len(all_models) == self.num_stages, "The number of stages for pipeline parallel is not enough, "\
            "and this is likely due to incorrect setting of config.dist.pp.split_points."

        self.model_name, self.model = all_models[self.stage_id]

        self.post_process = None
        if self.is_last_stage():
            self.post_process = utils.create_post_process(
                split_model, self.model_name, self.model)

        self.input_tensor_attr = utils.get_input_tensor_attr(
            split_model, self.stage_id)

    def _get_underlay_model(self):
        if isinstance(self.model, ParallelModule):
            return self.model._get_underlay_model()
        return self.model

    def _update_underlay_model(self, model: torch.nn.Module):
        self.model = model
        self.executor._update_underlay_model(model)

    def clip_grad_norm_(self, max_grad_norm):
        if hasattr(self.model, "clip_grad_norm_"):
            self.model.clip_grad_norm_(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_grad_norm)

    def forward(self, *args, output_fn=None, **kwargs):
        """Split the input batch into config.dist.pp.num_micro_batches micro batches for pipelining.
           This function will only perform forward, and return the averaged loss.

        Args:
            *args: input args of the model forward.
            output_fn: A function that processes the model's output. The first argument of this function
                is the return value of the model's forward, and the subsequent arguments can be arguments
                from kwargs. It is worth noting that tensors have already been split along the batchsize
                dimension.
            **kwargs: input kwargs of the model forward and optional input arguments of the output_fn.

        Returns:
            The loss averaged according to num_micro_batches.
        """
        if self.model.training and torch.is_grad_enabled(
        ) and self.forward_warn:
            logger.warning(
                "When using pipeline parallel for training, you need to use model.forward_backward"
            )
            self.forward_warn = False
        return self.executor.forward(*args, output_fn=output_fn, **kwargs)

    def forward_backward(self, *args, output_fn=None, **kwargs):
        """Split the input batch into config.dist.pp.num_micro_batches micro batches for pipelining.
           This function will perform forward and backward, and return the averaged loss.

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
        return self.executor.forward_backward(
            *args, output_fn=output_fn, **kwargs)
