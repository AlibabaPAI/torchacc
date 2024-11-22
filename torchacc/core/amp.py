import torch.distributed as dist

import torchacc as ta

from torchacc.utils.import_utils import is_torch_xla_available
if is_torch_xla_available():
    import torch_xla

    class GradScaler(torch_xla.amp.GradScaler):
        """This is the TorchAcc version of GradScaler, which has the same functionality and args as torch.cuda.amp.GradScaler,
        with support for TorchAcc and distributed scenarios such as pipeline parallel.
        """

        def __init__(
            self,
            init_scale=2.0**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
            enabled=True,
            use_zero_grad=False,
        ):
            super().__init__(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
                enabled=enabled,
                use_zero_grad=use_zero_grad,
            )

            self._lazy_init_scale_growth_tracker(ta.lazy_device())

        def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
            per_device_found_infs = super()._unscale_grads_(
                optimizer, inv_scale, found_inf, allow_fp16)
            if ta.get_global_context().mesh.get_pp_num() > 1:
                for found in per_device_found_infs.values():
                    dist.all_reduce(
                        found,
                        group=ta.get_global_context().mesh.get_pp_proc_group())
            return per_device_found_infs
