import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks.base import Callback
from torch._six import container_abcs, int_classes, string_classes
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, RandomSampler, Sampler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Logging(Callback):
    def __init__(self, save_dir: str):
        super().__init__()
        self.filename = os.path.join(save_dir, "results.jsonl")

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        pl_module.reset_metrics()

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        with open(self.filename, "a") as fp:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith("val_"):
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    logs[k] = v
            logs["step"] = trainer.global_step
            print(json.dumps(logs), file=fp)

    def on_test_start(self, trainer, pl_module):
        """Called when the test begins."""
        pl_module.reset_metrics()

    def on_test_end(self, trainer, pl_module):
        """Called when the test ends."""
        with open(self.filename, "a") as fp:
            logs = dict()
            for k, v in trainer.callback_metrics.items():
                if k.startswith("tst_") or k == "select":
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    logs[k] = v
            # assert "select" in logs
            print(json.dumps(logs), file=fp)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def get_inverse_square_root_schedule_with_warmup(
    optimizer, warmup_steps, last_epoch=-1
):
    """
    Create a schedule with linear warmup and then inverse square root decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Inverse square root decreases learning rate from 1. to 0. over remaining steps.
    """

    def lr_lambda(step):
        decay_factor = warmup_steps ** 0.5
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return decay_factor * step ** -0.5

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer, warmup_steps, training_steps, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        ratio = (training_steps - step) / max(1, training_steps - warmup_steps)
        return max(ratio, 0)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def pad_batch(batch, padding=-1):
    max_len = max([len(b) for b in batch])
    new_batch = []
    for b in batch:
        b_ = np.zeros(max_len, dtype=b.dtype) + padding
        b_[: len(b)] = b
        new_batch.append(b_)
    return new_batch


def default_collate(batch, padding):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate(
                [torch.as_tensor(b) for b in pad_batch(batch, padding)], padding
            )  # auto padding
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {
            key: default_collate([d[key] for d in batch], padding[key]) for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(
            *(default_collate(samples, padding) for samples in zip(*batch))
        )
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples, padding) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class ConcatSampler(Sampler):
    def __init__(self, concat_dataset: ConcatDataset, samples_per_dataset: int):
        assert isinstance(concat_dataset, ConcatDataset)
        self.concat_dataset = concat_dataset
        self.nb_datasets = len(concat_dataset.datasets)
        self.samples_per_dataset = samples_per_dataset

        weight = torch.tensor([len(d) for d in concat_dataset.datasets]).float()
        self.weight = weight / weight.sum()

    def sample_dataset(self):
        return torch.multinomial(self.weight, 1, replacement=True).item()

    def __iter__(self):
        iterators = [iter(RandomSampler(d)) for d in self.concat_dataset.datasets]
        done = np.array([False] * self.nb_datasets)
        while not done.all():
            dataset_id = self.sample_dataset()
            if done[dataset_id]:
                continue
            batch = []
            for _ in range(self.samples_per_dataset):
                try:
                    idx = next(iterators[dataset_id])
                except StopIteration:
                    done[dataset_id] = True
                    break
                if dataset_id > 0:
                    idx += self.concat_dataset.cumulative_sizes[dataset_id - 1]
                batch.append(idx)

            if len(batch) == self.samples_per_dataset:
                yield from batch

    def __len__(self):
        n = self.samples_per_dataset
        return sum([len(d) // n * n for d in self.concat_dataset.datasets])


def masked_log_softmax(
    vector: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of
    ``vector`` should be masked.  This performs a log_softmax on just the non-masked
    portions of ``vector``.  Passing ``None`` in for the mask is also acceptable; you'll
    just get a regular log_softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that
    ``mask`` is broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions
    than ``vector``, we will unsqueeze on dimension 1 until they match.  If you need a
    different unsqueezing of your mask, do it yourself before passing the mask into this
    function.

    In the case that the input vector is completely masked, the return value of this
    function is arbitrary, but not ``nan``.  You should be masking the result of whatever
    computation comes out of this in that case, anyway, so the specific values returned
    shouldn't matter.  Also, the way that we deal with this case relies on having
    single-precision floats; mixing half-precision floats with fully-masked vectors will
    likely give you ``nans``.

    If your logits are all extremely negative (i.e., the max value in your logit vector
    is -50 or lower), the way we handle masking here could mess you up.  But if you've
    got logit values that extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but
        # it results in nans when the whole vector is masked.  We need a very small value
        # instead of a zero in the mask for these cases.  log(1 + 1e-45) is still
        # basically 0, so we can safely just add 1e-45 before calling mask.log().  We use
        # 1e-45 because 1e-46 is so small it becomes 0 - this is just the smallest value
        # we can actually use.
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)
