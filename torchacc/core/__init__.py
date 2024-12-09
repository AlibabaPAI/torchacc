import warnings

import torch

from torchacc.utils.import_utils import is_torch_xla_available

from .async_loader import AsyncLoader

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    save = xm.save
    mark_step = xm.mark_step
    send_cpu_data_to_device = xm.send_cpu_data_to_device


def lazy_device():
    if not is_torch_xla_available():
        raise NotImplementedError(
            "The lazy backend of TorchAcc requires the installation of torch_xla. Please use `config.backend='eager'`"
            "or follow the instructions in https://torchacc.readthedocs.io/en/stable/install.html to use the recommended Docker image."
        )
    device = xm.xla_device()
    xm.set_replication(device, [device])
    return device


def is_lazy_device(device: torch.device):
    return device.type == 'xla'


def is_lazy_tensor(tensor: torch.tensor):
    if not is_torch_xla_available():
        return False
    return xm.is_xla_tensor(tensor)


def fetch_gradients(optimizer):
    gradients = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        gradients.append(p.grad.data)
    return gradients


def sync(wait: bool = False):
    """Generate a computation graph for all pending operations. Compile and
    optimize the computation graph according to the torchacc config, and then execute it.

    Args:
        wait (bool): Wait utils the graph execution finished if True. Otherwise, execute the
    computation graph asynchronously.
    """
    if not is_torch_xla_available():
        warnings.warn(
            "Sync is only valid in the lazy backend of TorchAcc and has no effect in eager backend"
        )
        return
    mark_step(wait)
