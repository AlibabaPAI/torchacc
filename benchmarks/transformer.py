import argparse
import os
import time

import torch
import torch.distributed as dist
import torchacc as ta
import torchacc.utils.logger as logger
import tqdm
from dataset import get_hf_dataset_loader
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler)

import torch_xla
import torch_xla.core.xla_model as xm

def _args_validation(args):
    if not args.acc:
        if args.tp_size > 1:
            raise ValueError('Torch native mode does not support TP. '
                             'Use --acc to enable TorchAcc when tp_size > 1.')
        if args.pp_size > 1:
            raise ValueError('Torch native mode does not support PP. '
                             'Use --acc to enable TorchAcc when pp_size > 1.')
        if args.fsdp_size > 1:
            raise ValueError('Torch native mode does not support FSDP. '
                             'Use --acc to enable TorchAcc when fsdp_size > 1.')
        if args.gc > 1:
            raise ValueError('Torch native mode does not support GC. '
                             'Use --acc to enable TorchAcc then using --gc.')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument(
        '--dataset_config', type=str, default='wikitext-2-raw-v1')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--model_block', type=str, default='GPT2Block')
    parser.add_argument('--tb_folder', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--max_step', type=int, default=None)
    parser.add_argument('--dp_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--fsdp_size', type=int, default=1)
    parser.add_argument('--num_micro_batches', type=int, default=1)
    parser.add_argument('--num_train_epochs', type=int, default=2)
    parser.add_argument('--acc', action='store_true', default=False)
    parser.add_argument('--backend', type=str, default='lazy')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bf16', action='store_true', default=False)
    parser.add_argument('--gc', action='store_true', default=False)
    parser.add_argument('--use_flash_attn', action='store_true', default=False)
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument(
        '--disable_loss_print', action='store_true', default=False)

    args = parser.parse_args()
    args.global_rank = int(os.getenv('RANK', '0'))
    args.local_rank = int(os.getenv('LOCAL_RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    if args.local_rank == 0:
        logger.info(f'Job running args: {args}')

    _args_validation(args)
    return args


def _setup_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()


def _get_config(args):
    config = ta.Config()
    config.backend = args.backend
    config.compute.fp16 = args.fp16
    config.compute.bf16 = args.bf16

    config.memory.gc = args.gc
    config.memory.gc_cls = {args.model_block}

    config.dist.dp.size = args.dp_size
    config.dist.tp.size = args.tp_size
    config.dist.fsdp.size = args.fsdp_size
    config.dist.fsdp.wrap_layer_cls = {args.model_block}
    config.dist.fsdp.flatten_parameters = False

    return config


def _build_model_and_loader(args):
    config = AutoConfig.from_pretrained(
        args.model_name, cache_dir='./log/model_cache')
    config.use_cache = False
    if args.use_flash_attn:
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation='flash_attention_2')
    else:
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.model_max_length = args.max_seq_length
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_hf_dataset_loader(tokenizer, args.dataset,
                                         args.max_seq_length, args.batch_size,
                                         args.global_rank, args.dataset_config,
                                         args.world_size)

    if args.acc:
        config = _get_config(args)
        model, train_loader = ta.accelerate(model, train_loader, config)
        if args.backend == "lazy" and args.use_flash_attn:
            ta.utils.patch.patch_llama(True)
    else:
        model = model.to(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model)

    return model, train_loader


def _build_lr_scheduler(optimizer, loader, num_train_epochs):
    num_training_steps = len(loader) * num_train_epochs
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


def train_gpt(args):
    model, train_loader = _build_model_and_loader(args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8)
    lr_scheduler = _build_lr_scheduler(optimizer, train_loader,
                                       args.num_train_epochs)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    amp_enabled = args.fp16 or args.bf16
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    model.train()
    if args.global_rank == 0 and args.tb_folder:
        writer = SummaryWriter(
            f'./{args.tb_folder}/{args.model_name.split("/")[-1]}_acc-{args.backend}_gpu-{args.world_size}_'
            f'amp-{amp_enabled}-{amp_dtype}_'
            f'dp-{args.dp_size}_pp-{args.pp_size}_fsdp-{args.fsdp_size}_'
            f'tp-{args.tp_size}-bs{args.batch_size}_'
            f'disable_loss_print-{args.disable_loss_print}')
        writer.add_text('args', str(args))
    else:
        writer = None

    if args.global_rank == 0 and args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                './log/profile'))
    iteration_time = time.time()
    with tqdm.tqdm(range(args.num_train_epochs * len(train_loader))) as pbar:
        for epoch in range(args.num_train_epochs):
            for step, inputs in enumerate(train_loader):
                if not args.acc:
                    inputs = {
                        key: value.to(args.local_rank)
                        for key, value in inputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(
                        enabled=amp_enabled, dtype=amp_dtype):
                    outputs = model(**inputs)
                    loss = outputs['loss']
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                lr_scheduler.step()

                if step % args.log_interval == 0 and args.global_rank == 0:
                    if args.disable_loss_print:
                        loss = 0.0
                    elif args.acc:
                        ta.sync()
                    step = step + epoch * len(train_loader)
                    time_cost = time.time() - iteration_time
                    iteration_time = time.time()
                    if writer is not None:
                        writer.add_scalar(
                            'samples/s',
                            args.batch_size / time_cost,
                            global_step=step)
                        writer.add_scalar('loss', loss, global_step=step)
                        writer.add_scalar(
                            'lr', lr_scheduler.get_last_lr()[0], global_step=step)
                    logger.info(
                        f'[Iteration {step}/{len(train_loader)*args.num_train_epochs}] '
                        f'loss: {loss:.6f}, '
                        f'complete in {time_cost:.2f} s')
                pbar.update(1)
                if args.global_rank == 0 and args.profile:
                    prof.step()


if __name__ == '__main__':
    seed = 101
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['TF_CUDNN_DETERMINISTIC']='1'
    os.environ['TF_DETERMINISTIC_OPS']='1'
    xm.set_rng_state(seed)
    torch_xla._XLAC._xla_set_use_full_mat_mul_precision(
      use_full_mat_mul_precision=True)

    args = _parse_args()

    if not args.acc or args.backend != "lazy":
        _setup_ddp(args.local_rank)

    train_gpt(args)
