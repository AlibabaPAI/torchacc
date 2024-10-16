from argparse import ArgumentParser

from torchacc.dist.state_dict_utils import (
    consolidate_and_reshard_fsdp_model_dict,
    consolidate_and_reshard_fsdp_optim_dict)

DEFAULT_MODEL_NAME_PATTERN = "rank-*-of-*-model.pth"
DEFAULT_OPTIM_NAME_PATTERN = "rank-*-of-*-optimizer.pth"


def main():
    parser = ArgumentParser()
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
        "--ckpt_type",
        type=str,
        choices=["model", "optimizer"],
        default="model",
        help=(
            "The type of checkpoint to consolidate, you can choose model or optimizer. Please consolidate model fisrt, and then consolidate optimizer."
        ),
    )
    parser.add_argument(
        "--reshard_num",
        type=int,
        default=1,
        help=(
            "We now support the reshard of XLA FSDP checkpoint according to the reshard_num."
        ),
    )
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

    if args.ckpt_type == "model":
        if args.ckpt_name == "":
            args.ckpt_name = DEFULT_MODEL_NAME_PATTERN
        if args.save_dir == "":
            args.save_dir = args.ckpt_dir
        if args.save_name == "":
            if args.reshard_num == 1:
                args.save_name = "model_consolidated.pth"
            else:
                args.save_name = DEFAULT_MODEL_NAME_PATTERN

        consolidate_and_reshard_fsdp_model_dict(args.ckpt_dir, args.ckpt_name,
                                                args.save_dir, args.save_name,
                                                args.reshard_num)
    else:
        if args.ckpt_name == "":
            args.ckpt_name = DEFULT_OPTIM_NAME_PATTERN
        if args.save_dir == "":
            args.save_dir = args.ckpt_dir
        if args.save_name == "":
            if args.reshard_num == 1:
                args.save_name = "optimizer_consolidated.pth"
            else:
                args.save_name = DEFAULT_OPTIM_NAME_PATTERN
        print(args.ckpt_dir)
        print(args.save_dir)
        print(args.save_name)
        consolidate_and_reshard_fsdp_optim_dict(args.ckpt_dir, args.ckpt_name,
                                                args.save_dir, args.save_name,
                                                args.reshard_num)


if __name__ == "__main__":
    main()
