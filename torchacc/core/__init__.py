import torch
import torch_xla.core.xla_model as xm

from .async_loader import AsyncLoader

save = xm.save
mark_step = xm.mark_step
send_cpu_data_to_device = xm.send_cpu_data_to_device


def lazy_device():
    device = xm.xla_device()
    xm.set_replication(device, [device])
    return device


def is_lazy_device(device: torch.device):
    return device.type == 'xla'


def is_lazy_tensor(tensor: torch.tensor):
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
