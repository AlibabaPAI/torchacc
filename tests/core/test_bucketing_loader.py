import torch
from torch.utils.data import DataLoader

import torchacc as ta
from utils import RawDataset


class TestBucketingDataLoader:

    @classmethod
    def setup_class(cls):
        bs = 8
        seq_len = 512
        rand_int = torch.randint(-500, 500, size=(20,))
        data = [{
            "input_ids":
                torch.zeros((bs, seq_len + rand_int[i]), dtype=torch.int64),
            "attention_mask":
                torch.zeros((bs, seq_len + rand_int[i]), dtype=torch.int64),
            "labels":
                torch.zeros((bs, seq_len + rand_int[i]), dtype=torch.int64)
        } for i in range(20)]
        dataloader = DataLoader(RawDataset(data), batch_size=None, shuffle=True)
        cls.dataloader = dataloader

    def test_buckets(self):
        device = ta.lazy_device()
        loader = ta.AsyncLoader(
            self.dataloader,
            device,
            max_length=1024,
            num_buckets=8,
            pad_value_dict={
                'input_ids': 0,
                'attention_mask': 0,
                'labels': -100
            })
        buckets = [1024 // 8 * (i + 1) for i in range(1024)]
        for batch in loader:
            assert batch['input_ids'].cpu().shape[-1] in buckets
            assert batch['attention_mask'].cpu().shape[-1] in buckets
            assert batch['labels'].cpu().shape[-1] in buckets
        del loader

    def test_uniform_buckets(self):
        device = ta.lazy_device()
        buckets = [128, 256, 512, 1024]
        loader = ta.AsyncLoader(
            self.dataloader,
            device,
            buckets=buckets,
            pad_value_dict={
                'input_ids': 0,
                'attention_mask': 0,
                'labels': -100
            })
        for batch in loader:
            assert batch['input_ids'].cpu().shape[-1] in buckets
            assert batch['attention_mask'].cpu().shape[-1] in buckets
            assert batch['labels'].cpu().shape[-1] in buckets
        del loader
