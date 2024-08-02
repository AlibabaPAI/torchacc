import argparse
import contextlib

import torch
import torchacc as ta
import torchacc.utils.checkpoint as checkpoint
import torchvision

from utils import EchoDataset, set_seed


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
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


class SkipNet(torch.nn.Module):
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


def autocast_amp(amp_enabled=True,
                 amp_dtype=torch.bfloat16,
                 cache_enabled=True):
    if not amp_enabled:
        return contextlib.nullcontext()
    ctx_manager = torch.cuda.amp.autocast(cache_enabled=cache_enabled,
                                          dtype=amp_dtype)
    return ctx_manager


def main(args):
    pp_num = args.pp_num
    batch_size = args.batch_size * args.num_micro_batches if pp_num > 1 else args.batch_size
    steps_per_print = args.steps_per_print
    num_micro_batches = args.num_micro_batches
    train_steps = args.train_steps * args.gradient_accumulation_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps

    model = Net() if not args.test_skip else SkipNet()

    # set PP config
    config = ta.Config()
    config.memory.gc = args.gc
    config.memory.gc_cls = {"Linear"}
    config.dist.pp.size = pp_num
    config.dist.pp.num_micro_batches = num_micro_batches
    if pp_num > 1:
        if args.test_skip:
            config.dist.pp.split_points = ["fc2", "fc3", "fc4"]
        else:
            config.dist.pp.split_points = [model.conv2, model.fc1, model.fc2]
        assert len(config.dist.pp.split_points) + 1 == pp_num

    device = ta.lazy_device()
    if pp_num > 1:
        model = ta.dist.PipelineParallel(model, config)
    # gradient checkpoint
    if config.memory.gc:
        if isinstance(model, ta.dist.ParallelModule):
            underlay_model = model._get_underlay_model()
            underlay_model = checkpoint.gradient_checkpoint(
                underlay_model, config.memory.gc_cls)
            model._update_underlay_model(underlay_model)
        else:
            model = checkpoint.gradient_checkpoint(model, config.memory.gc_cls)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = ta.amp.GradScaler() if args.fp16 else None

    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    amp_enabled = args.fp16 or args.bf16

    if args.test_skip:
        train_loader = EchoDataset(data=[
            torch.zeros(batch_size, 1024),
            torch.zeros(batch_size, dtype=torch.int64)
        ],
                                   repeat_count=train_steps)
    else:
        mnist_trainset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   drop_last=True)

    train_loader = ta.AsyncLoader(train_loader, device)
    total_loss = torch.tensor(0.0).to(device)
    global_step = 1
    for step, data in enumerate(train_loader):
        if pp_num > 1:

            def output_fn(x, labels):
                loss = torch.nn.functional.nll_loss(x, labels)
                if scaler is not None:
                    return scaler.scale(loss)
                return loss

            with autocast_amp(amp_enabled, amp_dtype):
                loss = model.forward_backward(data[0],
                                              labels=data[1],
                                              output_fn=output_fn)
        else:
            with autocast_amp(amp_enabled, amp_dtype):
                loss = model(data[0])
                loss = torch.nn.functional.nll_loss(loss, data[1])
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        step += 1
        loss = loss.clone().detach() / gradient_accumulation_steps
        total_loss += loss
        if step % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if ta.dist.local_rank() == 0:
                if global_step % steps_per_print == 0:
                    ta.mark_step()
                    print(f"step: {global_step}, loss: {total_loss}")
            if global_step == train_steps:
                ta.mark_step()
                return
            global_step += 1
            total_loss.zero_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TorchAcc Pipeline Parallel')
    parser.add_argument('--pp_num', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=10)
    parser.add_argument('--num_micro_batches', type=int, default=4)
    parser.add_argument("--gc", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--test_skip", action="store_true", default=False)
    args = parser.parse_args()

    if args.fp16:
        raise NotImplementedError("Currently not supported for fp16.")

    set_seed()
    main(args)
