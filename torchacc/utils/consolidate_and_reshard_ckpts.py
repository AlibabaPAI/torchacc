from argparse import ArgumentParser

from torchacc.dist.state_dict_utils import (
    consolidate_and_reshard_fsdp_checkpoint,
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
            f"Files matching the pattern ``ckpt_dir + ckpt_name_pattern`` will be load."
            f"For model, the default pattern is {DEFAULT_MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {DEFAULT_OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--model_ckpt_name_pattern",
        type=str,
        default=DEFAULT_MODEL_NAME_PATTERN,
        help=(
            f"The name pattern of the XLA FSDP checkpoint files to be consolidated and reshard. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name_pattern`` will be load."
            f"For model, the default pattern is {DEFAULT_MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {DEFAULT_OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--optimizer_ckpt_name_pattern",
        type=str,
        default=DEFAULT_OPTIM_NAME_PATTERN,
        help=(
            f"The name pattern of the XLA FSDP checkpoint files to be consolidated and reshard. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name_pattern`` will be load."
            f"For model, the default pattern is {DEFAULT_MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {DEFAULT_OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
        choices=["all", "model", "optimizer"],
        default="all",
        help=(
            f"The type of checkpoint to consolidate, you can choose to consolidate model and optimizer all or seperately."
            f"Please consolidate model first and then optimizer."),
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
            f"Files will be saved in path: ``save_dir + save_name_pattern``."
            f"For consolidated checkpoint, the default path is: ``save_dir + model/optimizer_consolidated.pth``."
            f"For reshard checkpoints, the default path is: ``save_dir + {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}``."
        ),
    )
    parser.add_argument(
        "--model_save_name_pattern",
        type=str,
        default="",
        help=(
            f"The save name pattern of the output checkpoint files, the default value is {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}."
            f"Files will be saved in path: ``save_dir + save_name_pattern`.`"
            f"For consolidated checkpoint, the default path is: ``save_dir + model/optimizer_consolidated.pth``"
            f"For reshard checkpoints, the default path is: ``save_dir + {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}``."
            f"For reshard checkpoints, please use the same name patthern as {DEFAULT_MODEL_NAME_PATTERN} and {DEFAULT_OPTIM_NAME_PATTERN}."
        ),
    )
    parser.add_argument(
        "--optimizer_save_name_pattern",
        type=str,
        default="",
        help=(
            f"The save name pattern of the output checkpoint files, the default value is {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}."
            f"Files will be saved in path: ``save_dir + save_name_pattern`.`"
            f"For consolidated checkpoint, the default path is: ``save_dir + model/optimizer_consolidated.pth``"
            f"For reshard checkpoints, the default path is: ``save_dir + {DEFAULT_MODEL_NAME_PATTERN}/{DEFAULT_OPTIM_NAME_PATTERN}``."
            f"For reshard checkpoints, please use the same name patthern as {DEFAULT_MODEL_NAME_PATTERN} and {DEFAULT_OPTIM_NAME_PATTERN}."
        ),
    )

    args = parser.parse_args()
    if args.ckpt_type == "all":
        if args.save_dir == "":
            args.save_dir = args.ckpt_dir
        if args.model_save_name_pattern == "":
            if args.reshard_num == 1:
                args.model_save_name_pattern = "model_consolidated.pth"
            else:
                args.model_save_name_pattern = DEFAULT_MODEL_NAME_PATTERN
        if args.optimizer_save_name_pattern == "":
            if args.reshard_num == 1:
                args.optimizer_save_name_pattern = "optimizer_consolidated.pth"
            else:
                args.optimizer_save_name_pattern = DEFAULT_OPTIM_NAME_PATTERN

        consolidate_and_reshard_fsdp_checkpoint(
            ckpt_dir=args.ckpt_dir,
            model_ckpt_name_pattern=args.model_ckpt_name_pattern,
            optimizer_ckpt_name_pattern=args.optimizer_ckpt_name_pattern,
            save_dir=args.save_dir,
            model_save_name_pattern=args.model_save_name_pattern,
            optimizer_save_name_pattern=args.optimizer_save_name_pattern,
            reshard_num=args.reshard_num)
    elif args.ckpt_type == "model":
        if args.save_dir == "":
            args.save_dir = args.ckpt_dir
        if args.model_save_name_pattern == "":
            if args.reshard_num == 1:
                args.model_save_name_pattern = "model_consolidated.pth"
            else:
                args.model_save_name_pattern = DEFAULT_MODEL_NAME_PATTERN

        consolidate_and_reshard_fsdp_model_dict(
            ckpt_dir=args.ckpt_dir,
            model_ckpt_name_pattern=args.model_ckpt_name_pattern,
            optimizer_ckpt_name_pattern=args.optimizer_ckpt_name_pattern,
            save_dir=args.save_dir,
            model_save_name_pattern=args.model_save_name_pattern,
            optimizer_save_name_pattern=args.optimizer_save_name_pattern,
            reshard_num=args.reshard_num)
    else:
        if args.save_dir == "":
            args.save_dir = args.ckpt_dir
        if args.optimizer_save_name_pattern == "":
            if args.reshard_num == 1:
                args.optimizer_save_name_pattern = "optimizer_consolidated.pth"
            else:
                args.optimizer_save_name_pattern = DEFAULT_OPTIM_NAME_PATTERN

        consolidate_and_reshard_fsdp_optim_dict(
            ckpt_dir=args.ckpt_dir,
            model_ckpt_name_pattern=args.model_ckpt_name_pattern,
            optimizer_ckpt_name_pattern=args.optimizer_ckpt_name_pattern,
            save_dir=args.save_dir,
            model_save_name_pattern=args.model_save_name_pattern,
            optimizer_save_name_pattern=args.optimizer_save_name_pattern,
            reshard_num=args.reshard_num)


if __name__ == "__main__":
    main()
