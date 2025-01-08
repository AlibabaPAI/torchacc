from typing import Any, Dict, List

import binpacking
import numpy as np
import torch


def flatten_mapfn_for_swift(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Data collator used for padding free approach. Does the following:
    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    Args:
        batch(`List[Dict[str, Any]]`): The input data in batch
        padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
            will be padded to the `longest`
    """
    packed_data = {}
    position_id_lengths = [len(item['input_ids']) for item in batch]
    packed_data['input_ids'] = np.concatenate(
        [item['input_ids'] for item in batch])
    packed_data['labels'] = np.concatenate([item['labels'] for item in batch])
    packed_data['position_ids'] = np.concatenate(
        [list(range(pil)) for pil in position_id_lengths])
    return packed_data


class SmartBatchingSampler:
    """Smart batching sampler for Megatron-LM.
    Args:
        dataset: A list of sequence lengths, each length is the length of a sequence.
        total_samples: Total number of samples to be consumed.
        micro_batch_size: Micro batch size.
        data_parallel_rank: Data parallel rank.
        data_parallel_size: Data parallel size.
        consumed_samples: Consumed samples, mainly usedfor continue train from the last checkpoint.
    """

    def __init__(
            self,
            dataset,  # Lengths of sequences,
            dataset_type,  # Workload type
            total_samples,  # Total number of samples
            micro_batch_size,  # Micro batch size
            data_parallel_rank,  # Data parallel rank
            data_parallel_size,  # Data parallel size
            consumed_samples=0,  # Consumed samples, mainly used for continue train from the last checkpoint
            balance_strategy='micro-batch',  # Balance strategy
    ):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.dataset_type = dataset_type
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.balance_strategy = balance_strategy
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)
        assert self.balance_strategy in ['micro-batch', "none"], \
            'invalid balance_strategy: {}, only {} and {} are supported'.format(self.balance_strategy, 'micro-batch', 'none')
        assert self.dataset_type in ['swift'], \
            'invalid dataset_type: {}, only {} are supported'.format(self.dataset_type, 'swift')

    def __len__(self):
        return self.total_samples // self.data_parallel_size // self.micro_batch_size

    def get_sequence_length(self, idx):
        if self.dataset_type == "swift":
            return len(self.dataset[idx]['input_ids'])

    def construct_balanced_batch(self, batch):
        # No balancing, just flatten the batch
        if self.balance_strategy == "none":
            return batch[self.data_parallel_rank::self.data_parallel_size]
        # Micro-batch level balancing
        if self.balance_strategy == "micro-batch":
            packages = {}
            for idx, sample_idx in enumerate(batch):
                packages[idx] = self.get_sequence_length(sample_idx)
            bins = binpacking.to_constant_bin_number(packages,
                                                     self.data_parallel_size)
            current_batch = []
            for idx in bins[self.data_parallel_rank].keys():
                current_batch.append(batch[idx])
            return current_batch

    def __iter__(self):
        # Sanity checks:
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # Continue train from where it left
        g = torch.Generator()
        g.manual_seed(self.epoch)
        shuffle_samples = torch.randperm(
            self.total_samples, generator=g).tolist()
        shuffle_samples = shuffle_samples[current_epoch_samples:]
        # Get one batch
        batch = []
        for idx in shuffle_samples:
            batch.append(idx)
            # Balance micro-batch across data parallel ranks
            if (self.balance_strategy == "micro-batch" or self.balance_strategy == "none") and \
                len(batch) == self.micro_batch_times_data_parallel_size:
                self.consumed_samples += self.micro_batch_size
                yield self.construct_balanced_batch(batch)
                batch.clear()
