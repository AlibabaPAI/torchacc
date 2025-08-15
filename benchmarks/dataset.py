import itertools

import datasets
import torch
import transformers


def get_hf_dataset_loader(tokenizer,
                          dataset_name_or_path,
                          max_seq_length,
                          batch_size,
                          global_rank=0,
                          dataset_config=None,
                          data_num_replicas=1):
    if dataset_name_or_path.endswith(".json"):
        raw_datasets = datasets.load_dataset(
            "json",
            data_files=dataset_name_or_path,
            split='train',
            cache_dir='./log/dataset')
    else:
        raw_datasets = datasets.load_dataset(
            dataset_name_or_path,
            dataset_config,
            split='train',
            cache_dir='./log/dataset')
    column_names = list(raw_datasets.features)
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    block_size = max_seq_length

    def group_texts(examples):
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result['labels'] = result['input_ids'].copy()
        return result

    train_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    # DataLoader creation
    train_sampler = None
    if data_num_replicas > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=data_num_replicas,
            rank=global_rank,
            shuffle=True)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=transformers.default_data_collator,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4,
        drop_last=True)
    return train_dataloader
