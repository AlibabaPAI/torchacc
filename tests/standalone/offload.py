import argparse
import time

import torch
import torch.distributed as dist
import torchacc as ta

from utils import EchoDataset, set_seed


class LinearNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1024)
        self.fc4 = torch.nn.Linear(1024, 1024)
        self.fc5 = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x1)
        x4 = self.fc4(x2)
        x5 = x3 + x4
        return x5


class Net(torch.nn.Module):

    def __init__(self, num_layers=4, num_offload_layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [LinearNet() for _ in range(num_layers)])
        self.num_offload_layers = num_offload_layers
        self.num_prefetch_layers = 1
        self.num_offload_sync_layers = 1

    def forward(self, x):

        if self.num_offload_layers > 0 and not hasattr(self,
                                                       "cpu_offload_context"):
            self.cpu_offload_context, self.cpu_offload_synchronizer = \
                ta.utils.cpu_offload.get_cpu_offload_context(num_offload_layers=self.num_offload_layers,
                                        num_prefetch_layers=self.num_prefetch_layers,
                                        num_offload_sync_layers=self.num_offload_sync_layers,
                                        debug=False)

        for layer in self.layers:
            if self.num_offload_layers > 0:
                with self.cpu_offload_context:
                    x = layer(x)
                x = self.cpu_offload_synchronizer(x)
            else:
                x = layer(x)
        return x


def train(args, model, device, train_loader, optimizer, scaler):
    model.train()
    e_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)

        scaler.scale(loss).backward()
        gradients = ta.fetch_gradients(optimizer)
        for grad in gradients:
            dist.all_reduce(grad)
            grad.mul_(1.0 / dist.get_world_size())
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % args.log_interval == 0 and dist.get_rank() == 0:
            batch_time = float(time.time() - e_time) / float(args.log_interval)
            e_time = time.time()
            samples_per_step = float(args.batch_size / batch_time)
            print(f"[TRAIN] iteration: {batch_idx}, "
                  f"batch_size: {args.batch_size}, loss: {loss:E}, "
                  f"throughput: {samples_per_step:3.3f} samples/sec")


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar='N',
        help='input batch size for training (default: 32)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    set_seed()

    dist.init_process_group(backend='nccl')

    train_loader = EchoDataset(
        data=[
            torch.randn(args.batch_size, 1024),
            torch.zeros(args.batch_size, dtype=torch.int64)
        ],
        repeat_count=1000 // args.batch_size // dist.get_world_size())

    device = dist.get_rank()
    model = Net()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, device, train_loader, optimizer, scaler)


if __name__ == '__main__':
    main()
