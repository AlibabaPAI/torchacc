import sys

import torch.nn.functional as F
import torch_xla.distributed.parallel_loader as pl

from torchacc.utils.logger import logger


def _uniform_buckets(max_length, num_buckets=8):
    """ uniform bucketing.
    """
    return [max_length // num_buckets * (i + 1) for i in range(num_buckets)]


def _get_closet_bucket(bucket_sizes, data_length):
    """Select the one from bucket_sizes that is closest in distance to
    data_length.
    """
    closet_length = sys.maxsize
    for b in bucket_sizes:
        if b == data_length or ((b < closet_length) and (b > data_length)):
            closet_length = b

    if closet_length == sys.maxsize:
        bucket_sizes.append(data_length)
        closet_length = data_length

    return closet_length


class BucketingParallelLoader(pl.ParallelLoader):
    """Wraps an existing PyTorch DataLoader with background data upload.
    If the last dimension of the output data from the original dataloader is dynamic,
    it will first be bucketed and then padded to a consistent shape corresponding
    to its respective bucket.

    Args:
        loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
           wrapped.
        devices (`torch.device`...): The list of devices where the data has to be
            sent. The i-th sample returned by the `loader` will be sent to `devices[i
            % len(devices)]`.
        batchdim (int, optional): The dimension which is holding the batch size.
            Default: 0
        loader_prefetch_size (int, optional): The max capacity of the queue used by
            the thread which is reading samples from the `loader`, to be processed by
            the worker threads which upload data to the devices.
            Default: 8
        device_prefetch_size (int, optional): The max size of the per-device queues,
            where the worker threads deposit tensors which have already been sent to
            devices.
            Default: 4
        host_to_device_transfer_threads (int, optional): The number of threads that
            work in parallel to transfer data from loader queue to device queue.
            Default: 1
        input_sharding (ShardingSpec, optional): Sharding spec to apply to
            compatible input tensors after loading.
            Default: None
        buckets (list): A list of integers that records the sizes of each bucket.
            When it is not None, the following args `max_length` and `num_buckets`
            will be invalid. Default setting is None.
        max_length (int): Max last dim length used for bucketing data loader. Default
            setting is None, indicating that bucketing will not be employed.
        num_buckets (int): The total count of buckets employed within the bucketing data loader.
        pad_value_dict (dict): The default padding value for each type of element in
            bucketing dataloader's output. The default setting is as follows:
            {'input_ids': 0, 'attention_mask': 0, 'labels': -100}
    """

    def __init__(self,
                 loader,
                 devices,
                 batchdim=0,
                 batches_per_execution=1,
                 loader_prefetch_size=8,
                 device_prefetch_size=4,
                 host_to_device_transfer_threads=1,
                 input_sharding=None,
                 buckets=None,
                 max_length=None,
                 num_buckets=8,
                 pad_value_dict=None):
        self.buckets = buckets
        if self.buckets is None and max_length is not None:
            self.buckets = _uniform_buckets(max_length, num_buckets)

        self.max_length = max_length
        self.num_buckets = num_buckets
        if pad_value_dict is None:
            self.pad_value_dict = {
                'input_ids': 0,
                'attention_mask': 0,
                'labels': -100
            }
        else:
            self.pad_value_dict = pad_value_dict

        super().__init__(loader, devices, batchdim, batches_per_execution,
                         loader_prefetch_size, device_prefetch_size,
                         host_to_device_transfer_threads, input_sharding)

    def _get_batch(self, dqueue):
        batch = []
        while dqueue.queue.max_size() > len(batch):
            item = dqueue.loader_queue.get()
            if item is None:
                break

            # bucketing
            if self.buckets is not None:
                if not isinstance(item, dict):
                    logger.warn(
                        'Data required to be dict when using bucketing.')
                else:
                    assert self.pad_value_dict.keys() == item.keys()
                    longest_len = item[list(self.pad_value_dict)[0]].shape[-1]
                    bucket_data_length = _get_closet_bucket(
                        self.buckets, longest_len)
                    padding_length = bucket_data_length - longest_len

                    logger.debug(f'original length: {longest_len}, '
                                 f'bucket length: {bucket_data_length}, '
                                 f'padding length: {padding_length}')
                    if padding_length > 0:
                        for k, v in self.pad_value_dict.items():
                            item[k] = F.pad(item[k], (0, padding_length),
                                            'constant', v)

            batch.append(item)
        return batch


class AsyncLoader(object):
    """Wraps an existing PyTorch DataLoader with background data upload.

    Args:
        loader (:class:`torch.utils.data.DataLoader`): The PyTorch DataLoader to be
          wrapped.
        device (`torch.device`...): The device where the data has to be sent.
        buckets (list): A list of integers that records the sizes of each bucket.
            When it is not None, the following args `max_length` and `num_buckets`
            will be invalid. Default setting is None.
        max_length (int): Max last dim length used for bucketing data loader. Default
            setting is None, indicating that bucketing will not be employed.
        num_buckets (int): The total count of buckets employed within the bucketing data loader.
        pad_value_dict (dict): The default padding value for each type of element in
            bucketing dataloader's output. The default setting is as follows:
            {'input_ids': 0, 'attention_mask': 0, 'labels': -100}
        kwargs: Named arguments for the `BucketingParallelLoader` constructor.
    """

    def __init__(self,
                 loader,
                 device,
                 buckets=None,
                 max_length=None,
                 num_buckets=8,
                 pad_value_dict=None,
                 **kwargs):
        self._loader = loader
        self._device = device
        self.buckets = buckets
        self.max_length = max_length
        self.num_buckets = num_buckets
        self.pad_value_dict = pad_value_dict
        self._parallel_loader_kwargs = kwargs

    def __iter__(self):
        parallel_loader = BucketingParallelLoader(
            self._loader, [self._device],
            buckets=self.buckets,
            max_length=self.max_length,
            num_buckets=self.num_buckets,
            pad_value_dict=self.pad_value_dict,
            **self._parallel_loader_kwargs)
        return parallel_loader.per_device_loader(self._device)

    def __len__(self):
        return len(self._loader)
