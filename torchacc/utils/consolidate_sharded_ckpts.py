from argparse import ArgumentParser
from torchacc.dist.state_dict_utils import consolidate_and_reshard_model_dict, consolidate_and_reshard_optim_dict

MODEL_NAME_PATTERN = "rank*-of-*-model.pth"
OPTIM_NAME_PATTERN = "rank*-of-*-optimizer.pth"


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help=(
            f"The name pattern of the XLA FSDP checkpoint files to be consolidated. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name`` will be loaded."
            f"For model, the default pattern is {MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="",
        help=(
            f"The name pattern of the XLA FSDP checkpoint files to be consolidated. "
            f"Files matching the pattern ``ckpt_dir + ckpt_name`` will be loaded."
            f"For model, the default pattern is {MODEL_NAME_PATTERN}. For optimizer,"
            f"the default pattern is {OPTIM_NAME_PATTERN}"),
    )
    parser.add_argument(
        "--ckpt_type",
        type=str,
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
        ))
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help=(
            f"The save path of the output state dict "
            f"(default consolidate_path is ``ckpt_dir + model/optimizer_consolidated.pth``)"
            f"If you need to reshard the checkpoint, please only pass the save_dir(default is ckpt_dir),"
            f"we will save the file in path ``save_path + {MODEL_NAME_PATTERN}/{OPTIM_NAME_PATTERN}``"
        ),
    )
    args = parser.parse_args()
    assert args.ckpt_type in ['model', 'optimizer'
                             ], ('the ckpt_type should be model or optimizer')

    if args.ckpt_type == "model":
        if args.ckpt_name == "":
            args.ckpt_name = MODEL_NAME_PATTERN
        consolidate_and_reshard_model_dict(args.ckpt_dir, args.ckpt_name,
                                           args.reshard_num, args.save_path)
    else:
        if args.ckpt_name == "":
            args.ckpt_name = OPTIM_NAME_PATTERN
        consolidate_and_reshard_optim_dict(args.ckpt_dir, args.ckpt_name,
                                           args.reshard_num, args.save_path)


if __name__ == "__main__":
    main()
