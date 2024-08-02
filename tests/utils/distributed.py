import os
import sys
from functools import wraps

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (TEST_SKIPS,
                                                        MultiProcessTestCase,
                                                        skip_if_lt_x_gpu)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests, parametrize)


class MultiProcessTestBase(MultiProcessTestCase):
    def run_test(self, test_name: str, parent_pipe) -> None:
        os.environ['LOCAL_RANK'] = str(self.rank)
        os.environ['RANK'] = str(self.rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['PJRT_LOCAL_PROCESS_RANK'] = str(self.rank)
        super().run_test(test_name, parent_pipe)

    def init_pg(self, backend) -> None:
        if ("nccl" in backend or "lazy"
                in backend) and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if backend not in [
                "nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl", "lazy"
        ]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
        )

        # set device for nccl pg for collectives
        if "nccl" in backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()


def init_pg(backend):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.init_pg(backend)
            func(self, *args, **kwargs)
            self.destroy_pg()

        return wrapper

    return decorator
