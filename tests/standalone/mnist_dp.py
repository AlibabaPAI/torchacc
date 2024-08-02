import argparse
import time

import torch
import torch.distributed as dist
import torchacc as ta

from utils import EchoDataset


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(9216, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output


def train(args, model, train_loader, optimizer, epoch, scaler):
    model.train()
    e_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
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
            ta.mark_step()
            batch_time = float(time.time() - e_time) / float(args.log_interval)
            e_time = time.time()
            samples_per_step = float(args.batch_size / batch_time)
            print(f"[TRAIN] epoch: {epoch}, iteration: {batch_idx}, "
                  f"batch_size: {args.batch_size}, loss: {loss:E}, "
                  f"throughput: {samples_per_step:3.3f} samples/sec")


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=50,
        metavar='N',
        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    dist.init_process_group(backend='lazy')

    train_loader = EchoDataset(
        data=(torch.zeros(args.batch_size, 1, 28,
                          28), torch.zeros(args.batch_size,
                                           dtype=torch.int64)),
        repeat_count=60000 // args.batch_size // ta.dist.world_size())

    device = ta.lazy_device()
    model = Net()
    model.to(device)
    train_loader = ta.AsyncLoader(train_loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, scaler)


if __name__ == '__main__':
    main()
