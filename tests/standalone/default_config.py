import argparse

import torch
import torchacc as ta

from utils import EchoDataset, set_seed


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1024, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 1024)
        self.fc4 = torch.nn.Linear(1024, 1024)
        self.fc5 = torch.nn.Linear(1024, 1024)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def main(args):
    batch_size = args.batch_size
    steps_per_print = args.steps_per_print
    train_steps = args.train_steps * args.gradient_accumulation_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps

    model = Net()

    # accelerate
    model = ta.accelerate(model)

    device = model.device

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scaler = ta.amp.GradScaler() if args.fp16 else None

    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    amp_enabled = args.fp16 or args.bf16

    train_loader = EchoDataset(
        data=[
            torch.zeros(batch_size, 1024),
            torch.zeros(batch_size, dtype=torch.int64)
        ],
        repeat_count=train_steps)

    train_loader = ta.AsyncLoader(train_loader, device)
    total_loss = torch.tensor(0.0).to(device)
    global_step = 1
    for step, data in enumerate(train_loader):
        with torch.cuda.amp.autocast(
                enabled=amp_enabled, cache_enabled=True, dtype=amp_dtype):
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
                    ta.sync()
                    ta.utils.logger.info(
                        f"step: {global_step}, loss: {total_loss}")
            if global_step == train_steps:
                ta.sync()
                return
            global_step += 1
            total_loss.zero_()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TorchAcc accelerate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=10)
    parser.add_argument('--num_micro_batches', type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    args = parser.parse_args()

    set_seed()
    main(args)
