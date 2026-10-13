import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from visaudio.data.extracted_BiAudio import ExtractedBiAudio
from visaudio.data.mm_dataset import MultiModalDataset
from visaudio.utils.dist_utils import global_rank, world_size


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class EvaluationSampler(Sampler):
    """Shard without DistributedSampler's duplicate padding at evaluation."""

    def __init__(self, dataset):
        self.indices = range(global_rank, len(dataset), world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def load_BiAudio_data(cfg, data_cfg):
    return ExtractedBiAudio(
        data_cfg.tsv,
        feature_dir=data_cfg.memmap_dir,
        data_dim=cfg.data_dim,
        stats_path=data_cfg.get("stats_path"),
    )


def setup_training_datasets(cfg):
    datasets = []
    for name, repeats in cfg.train_datasets.items():
        if repeats < 1 or int(repeats) != repeats:
            raise ValueError(f"Training repeat count must be a positive integer: {name}")
        dataset = load_BiAudio_data(cfg, cfg.data[f"Extracted_{name}_train"])
        datasets.extend([dataset] * int(repeats))
    if not datasets:
        raise ValueError("Configure at least one training dataset.")
    dataset = MultiModalDataset(datasets, [])
    sampler, loader = construct_loader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=cfg.pin_memory,
    )
    if not len(loader):
        raise ValueError("No full training batch. Reduce batch_size or add more samples.")
    return dataset, sampler, loader


def setup_val_datasets(cfg):
    dataset = load_BiAudio_data(cfg, cfg.data[f"Extracted_{cfg.val_dataset}_test"])
    # Validation uses DDP forward, so all ranks must have equally many batches.
    _, val_loader = construct_loader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.pin_memory,
    )
    _, eval_loader = construct_loader(
        dataset,
        cfg.eval_batch_size,
        cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.pin_memory,
        exact_shard=True,
    )
    return dataset, val_loader, eval_loader


def setup_eval_dataset(dataset_name, cfg):
    from visaudio.data.eval.video_dataset import BiAudio

    if dataset_name not in cfg.eval_data:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    data = cfg.eval_data[dataset_name]
    dataset = BiAudio(data.video_path, data.csv_path, duration_sec=cfg.duration_s)
    _, loader = construct_loader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=cfg.pin_memory,
        exact_shard=True,
        error_avoidance=True,
    )
    return dataset, loader


def error_avoidance_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    return default_collate(batch) if batch else None


def construct_loader(
    dataset,
    batch_size,
    num_workers,
    *,
    shuffle=True,
    drop_last=True,
    pin_memory=False,
    error_avoidance=False,
    exact_shard=False,
):
    sampler = (
        EvaluationSampler(dataset)
        if exact_shard
        else DistributedSampler(dataset, num_replicas=world_size, rank=global_rank, shuffle=shuffle)
    )
    loader = DataLoader(
        dataset,
        batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=error_avoidance_collate if error_avoidance else None,
    )
    return sampler, loader
