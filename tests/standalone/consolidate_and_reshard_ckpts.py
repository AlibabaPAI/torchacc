import argparse
import os

import torch
import torch_xla.core.xla_model as xm
import torchacc as ta
from torchacc.dist.state_dict_utils import (
    consolidate_and_reshard_fsdp_model_dict,
    consolidate_and_reshard_fsdp_optim_dict, load_checkpoints)
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


def compare_model_dict(dict1, dict2, idx):
    if dict1.keys() != dict2.keys():
        print("model dict keys are different")
        return

    difference = False

    for key in dict2.keys():
        tensor1 = dict1[key]
        tensor2 = dict2[key]

        if not torch.equal(tensor1, tensor2):
            print(f"Difference found at key: {key}")
            print(f"Tensor 1: {tensor1}")
            print(f"Tensor 2: {tensor2}")
            difference = True

    if not difference:
        print(f"The model dict shard {idx} are same.")


def compare_optim_dict(state_dict1, state_dict2, idx):
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
        print(f"The optim dict shard {idx} are same.")


def main(args):
    fsdp_num = args.fsdp_num
    batch_size = args.batch_size
    train_steps = args.train_steps * args.gradient_accumulation_steps
    ckpt_dir = args.ckpt_dir
    reshard_num = args.reshard_num

    model = Net()

    # set config
    config = ta.Config()
    config.backend = args.backend
    config.compute.fp16 = args.fp16
    config.compute.bf16 = args.bf16

    config.dist.fsdp.size = fsdp_num
    config.dist.fsdp.wrap_layer_cls = {"Linear"}
    config.dist.fsdp.flatten_parameters = True

    # accelerate
    model = ta.accelerate(model, config=config)
    device = model.device

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_loader = EchoDataset(
        data=[
            torch.zeros(batch_size, 1024),
            torch.zeros(batch_size, dtype=torch.int64)
        ],
        repeat_count=train_steps)

    train_loader = ta.AsyncLoader(train_loader, device)

    # train model
    train(args, model, device, train_loader, optimizer)

    # save shard model and optimizer
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_ckpt = {
        "model": model.state_dict(),
        "shard_metadata": model.model.model.get_shard_metadata(
        ),  # we need first get the xla model
    }
    model_ckpt_path = os.path.join(
        ckpt_dir,
        f"rank{ta.dist.local_rank()}-of-{ta.dist.world_size()}-model.pth")
    ta.save(model_ckpt, model_ckpt_path, master_only=False)
    xm.rendezvous("saving_model")

    optim_ckpt = {
        "optimizer": optimizer.state_dict(),
        "shard_metadata": model.model.model.get_shard_metadata(),
    }
    optim_ckpt_path = os.path.join(
        ckpt_dir,
        f"rank{ta.dist.local_rank()}-of-{ta.dist.world_size()}-optim.pth")
    ta.save(optim_ckpt, optim_ckpt_path, master_only=False)
    xm.rendezvous("saving_optim")

    # rank 0 do consolidate and reshard:
    if ta.dist.local_rank() == 0:
        # consolidate and reshard model and optimizer
        model_reshard_dicts, _ = consolidate_and_reshard_fsdp_model_dict(
            ckpt_dir=ckpt_dir,
            ckpt_name=f"rank*-of-*-model.pth",
            reshard_num=reshard_num,
            save_model=False,
        )
        print(f"model consolidate and reshard done.")

        optim_reshard_dicts, _ = consolidate_and_reshard_fsdp_optim_dict(
            ckpt_dir=ckpt_dir,
            ckpt_name=f"rank*-of-*-optim.pth",
            reshard_num=reshard_num,
            save_optimizer=False,
        )
        print(f"optimizer consolidate and reshard done.")

        # compare shard model and optimizer
        if reshard_num == fsdp_num:
            model_shard_dicts = load_checkpoints(
                ckpt_dir=ckpt_dir, ckpt_name=f"rank*-of-*-model.pth")
            optim_shard_dicts = load_checkpoints(
                ckpt_dir=ckpt_dir, ckpt_name=f"rank*-of-*-optim.pth")

            for idx, (dict1, dict2) in enumerate(
                    zip(model_shard_dicts, model_reshard_dicts)):
                compare_model_dict(dict1['model'], dict2, idx)

            for idx, (dict1, dict2) in enumerate(
                    zip(optim_shard_dicts, optim_reshard_dicts)):
                compare_optim_dict(dict1['optimizer'], dict2, idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TorchAcc Consolidate And Reshard FSDP Checkpoints')
    parser.add_argument('--fsdp_num', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--steps_per_print', type=int, default=1)
    parser.add_argument('--train_steps', type=int, default=10)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--backend", type=str, default="lazy")

    DEFAULT_MODEL_NAME_PATTERN = "rank*-of-*-model.pth"
    DEFAULT_OPTIM_NAME_PATTERN = "rank*-of-*-optimizer.pth"
    # ckpt arguments
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help=(
            f"The name dir of the XLA FSDP checkpoint files to be consolidated and reshard. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name`` will be load."
            f"For model, the default pattern is {DEFAULT_MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {DEFAULT_OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="",
        help=(
            f"The name pattern of the XLA FSDP checkpoint files to be consolidated and reshard. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name`` will be load."
            f"For model, the default pattern is {DEFAULT_MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {DEFAULT_OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--reshard_num",
        type=int,
        default=1,
        help=(
            "We now support the reshard of XLA FSDP checkpoint according to the reshard_num."
        ))
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help=(
            f"The save dir of the output checkpoint files, the default value will be set to arg: ckpt_dir."
            f"Files will be saved in path: ``save_dir + save_name``."
            f"For consolidated checkpoint, the default path is: ``save_dir + model/optimizer_consolidated.pth``."
            f"For reshard checkpoints, the default path is: ``save_dir + {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}``."
        ),
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="",
        help=(
            f"The save name pattern of the output checkpoint files, the default value is {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}."
            f"Files will be saved in path: ``save_dir + save_name`.`"
            f"For consolidated checkpoint, the default path is: ``save_dir + model/optimizer_consolidated.pth``"
            f"For reshard checkpoints, the default path is: ``save_dir + {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}``."
            f"For reshard checkpoints, please use the same name patthern as {DEFAULT_MODEL_NAME_PATTERN} and {DEFAULT_OPTIM_NAME_PATTERN}."
        ),
    )

    args = parser.parse_args()

    set_seed()
    main(args)
