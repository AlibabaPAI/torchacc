import argparse
import os

import torch
from utils import EchoDataset, set_seed

import torchacc as ta
from torchacc.dist.fsdp import FullyShardedDataParallel as FSDP


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


def compare_optim_dict(state_dict1, state_dict2, rank):
    state1 = state_dict1['state']
    state2 = state_dict2['state']
    if state1.keys() != state2.keys():
        print("optimizer state keys are different")
        return

    difference = False
    for key in state2.keys():
        dict1 = state1[key]
        dict2 = state2[key]
        for state_name in dict1.keys():
            tensor1 = dict1[state_name]
            tensor2 = dict2[state_name]

            if not torch.equal(tensor1, tensor2):
                print(f"Difference found at state key: {key}-{state_name}")
                print(f"Tensor 1: {tensor1}")
                print(f"Tensor 2: {tensor2}")
                difference = True

    param_list1 = state_dict1['param_groups']
    param_list2 = state_dict2['param_groups']

    for param1, param2 in zip(param_list1, param_list2):
        if param1.keys() != param2.keys():
            print("optimizer param_groups keys are different")
            return

        for key in param2.keys():
            if param2[key] != param1[key]:
                print(f"Difference found at param_group key: {key}")
                print(f"value 1: {param1[key]}")
                print(f"value 2: {param2[key]}")
                difference = True

    if not difference:
        print(f"The optim dict shard of rank {rank} are same.")


def train(args, model, device, train_loader, optimizer):
    steps_per_print = args.steps_per_print
    train_steps = args.train_steps * args.gradient_accumulation_steps

    scaler = ta.amp.GradScaler() if args.fp16 else None

    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    amp_enabled = args.fp16 or args.bf16
    gradient_accumulation_steps = args.gradient_accumulation_steps

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


def main(args):
    fsdp_num = args.fsdp_num
    batch_size = args.batch_size
    train_steps = args.train_steps * args.gradient_accumulation_steps

    model = Net()

    # set config
    config = ta.Config()
    config.backend.mode = args.backend
    config.compute.fp16 = args.fp16
    config.compute.bf16 = args.bf16

    config.dist.fsdp.size = fsdp_num
    config.dist.fsdp.wrap_layer_cls = {"Linear"}
    config.dist.fsdp.flatten_parameters = True

    # accelerate
    model = ta.accelerate(model, config=config)
    device = model.device

    optim = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_loader = EchoDataset(
        data=[
            torch.zeros(batch_size, 1024),
            torch.zeros(batch_size, dtype=torch.int64)
        ],
        repeat_count=train_steps)

    train_loader = ta.AsyncLoader(train_loader, device)

    # train model
    train(args, model, device, train_loader, optim)
    optim_1 = optim.state_dict()

    # test for full eager optim ckpt.
    ckpt_dir = "standalone/ckpt"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    optim_save_path = os.path.join(ckpt_dir, "eager_full_optim.pth")

    optim_state_dict = FSDP.full_optim_state_dict(model, optim)
    if ta.dist.local_rank() == 0:
        ta.save(optim_state_dict, optim_save_path)

    # load full eager optim ckpts.
    if ta.dist.local_rank() == 0:
        optim_state_dict = torch.load(optim_save_path)
    optim_state_dict = FSDP.optim_state_dict_to_load(model, optim,
                                                     optim_state_dict)
    optim.load_state_dict(optim_state_dict)

    # the optim_state_dict of current optim(sharded) should equal to origin state.
    optim_2 = optim.state_dict()
    compare_optim_dict(optim_1, optim_2, ta.dist.local_rank())

    # test for sharded eager optim ckpt.
    sharded_optim = FSDP.sharded_optim_state_dict(model, optim)

    optim_save_path = os.path.join(
        ckpt_dir, f"rank{ta.dist.rank()}-of-{fsdp_num}-optim.pth")

    ta.dist.rendezvous("saving_optim")
    ta.save(sharded_optim, optim_save_path, master_only=False)
    sharded_optim = torch.load(optim_save_path)

    sharded_optim = FSDP.optim_state_dict_to_load(model, optim, sharded_optim)
    optim.load_state_dict(sharded_optim)

    optim_3 = optim.state_dict()

    compare_optim_dict(optim_1, optim_3, ta.dist.local_rank())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TorchAcc eager save and load fsdp ckpts')
    parser.add_argument('--fsdp_num', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="eager")

    args = parser.parse_args()

    set_seed()
    main(args)
