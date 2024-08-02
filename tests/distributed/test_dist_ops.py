import torch
import torch.distributed as dist
import torchacc as ta

from utils.distributed import MultiProcessTestBase, init_pg, skip_if_lt_x_gpu


class DistTest(MultiProcessTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_lazy_send_recv(self):
        device = ta.lazy_device()
        torch.manual_seed(1024)

        new_group = dist.new_group([0, 1])
        send_tensor = torch.randn(4, 10).to(device)
        recv_tensor = torch.randn(4, 10).to(device)

        if dist.get_rank() == 0:
            dist.send(send_tensor, 1, group=new_group)
        elif dist.get_rank() == 1:
            dist.recv(recv_tensor, 0, group=new_group)
        ta.mark_step(wait=True)
        if dist.get_rank() == 1:
            assert torch.allclose(recv_tensor.to('cpu'), send_tensor.to('cpu'))

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_lazy_backend_for_cuda(self):
        device = dist.get_rank()
        tensor = torch.tensor(list(range(10)))
        if dist.get_rank() != 0:
            tensor = torch.tensor(list(range(10, 20)))
        tensor = tensor.to(device)
        dist.broadcast(tensor, 0)
        assert torch.allclose(tensor,
                              torch.tensor(list(range(10)), device=device))

    @skip_if_lt_x_gpu(2)
    @init_pg("lazy")
    def test_lazy_backend_mixed(self):
        # lazy
        device = ta.lazy_device()
        torch.manual_seed(1024)

        new_group = dist.new_group([0, 1])
        send_tensor = torch.randn(4, 10).to(device)
        recv_tensor = torch.randn(4, 10).to(device)

        if dist.get_rank() == 0:
            dist.send(send_tensor, 1, group=new_group)
        elif dist.get_rank() == 1:
            dist.recv(recv_tensor, 0, group=new_group)
        ta.mark_step()
        if dist.get_rank() == 1:
            assert torch.allclose(recv_tensor.to('cpu'), send_tensor.to('cpu'))

        # cuda
        device = dist.get_rank()
        tensor = torch.tensor(list(range(10)))
        if dist.get_rank() != 0:
            tensor = torch.tensor(list(range(10, 20)))
        tensor = tensor.to(device)
        dist.broadcast(tensor, 0)
        assert torch.allclose(tensor,
                              torch.tensor(list(range(10)), device=device))
