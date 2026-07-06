import hashlib
import random
from typing import Dict, Iterator, Union, List, Optional, Any
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .dataset import CrystalGraph, TugaGraphBuilder


class OnTheFlyDataset(Dataset):
    """
    Map-style PyTorch dataset that constructs CrystalGraph objects on the fly
    from a random-access adapter (e.g. ListStoreAdapter, AseLmdbAdapter, PickleAdapter).
    
    Supports deterministic train/val splitting via per-index hashing.
    """
    def __init__(
        self,
        adapter: Any,
        builder: TugaGraphBuilder,
        train_ratio: float = 1.0,
        is_train: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.adapter = adapter
        self.builder = builder
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.seed = seed
        
        total_len = len(adapter)
        if train_ratio >= 1.0:
            if is_train:
                self.indices = list(range(total_len))
            else:
                self.indices = []
        elif train_ratio <= 0.0:
            if is_train:
                self.indices = []
            else:
                self.indices = list(range(total_len))
        else:
            # Deterministic split via md5 hashing
            self.indices = []
            for idx in range(total_len):
                h = hashlib.md5(f"{seed}:0:{idx}".encode()).hexdigest()
                frac = int(h[:8], 16) / 0xFFFFFFFF
                is_row_train = frac < train_ratio
                if is_row_train == is_train:
                    self.indices.append(idx)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> CrystalGraph:
        actual_idx = self.indices[idx]
        item = self.adapter[actual_idx]
        
        structure = item["structure"]
        y = item.get("y", None)
        mat_id = item.get("mat_id", None)
        
        properties = {}
        if y is not None:
            properties["y"] = y
            
        if hasattr(structure, "get_atomic_numbers"):
            graph = self.builder.get_graph_from_atoms(structure, properties=properties, mat_id=mat_id)
        else:
            graph = self.builder.get_graph(structure, properties=properties, mat_id=mat_id)
            
        return graph


class OnTheFlyIterableDataset(IterableDataset):
    """
    Iterable-style PyTorch dataset for streaming structures from an iterable adapter.
    Handles multi-worker stream partitioning and shuffle buffering.
    """
    def __init__(
        self,
        adapter: Any,
        builder: TugaGraphBuilder,
        train_ratio: float = 1.0,
        is_train: bool = True,
        seed: int = 42,
        shuffle: bool = True,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.adapter = adapter
        self.builder = builder
        self.train_ratio = train_ratio
        self.is_train = is_train
        self.seed = seed
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[CrystalGraph]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        
        rng = random.Random(self.seed + worker_id + 1000)
        
        def generator():
            for idx, item in enumerate(self.adapter):
                # Distribute records across workers
                if idx % num_workers != worker_id:
                    continue
                
                # Check train/val split ratio
                if self.train_ratio < 1.0:
                    h = hashlib.md5(f"{self.seed}:0:{idx}".encode()).hexdigest()
                    frac = int(h[:8], 16) / 0xFFFFFFFF
                    is_row_train = frac < self.train_ratio
                    if is_row_train != self.is_train:
                        continue
                        
                structure = item["structure"]
                y = item.get("y", None)
                mat_id = item.get("mat_id", None)
                
                properties = {}
                if y is not None:
                    properties["y"] = y
                    
                if hasattr(structure, "get_atomic_numbers"):
                    graph = self.builder.get_graph_from_atoms(structure, properties=properties, mat_id=mat_id)
                else:
                    graph = self.builder.get_graph(structure, properties=properties, mat_id=mat_id)
                yield graph

        stream = generator()
        if not self.shuffle or self.buffer_size <= 1:
            yield from stream
        else:
            buffer = []
            for _ in range(self.buffer_size):
                try:
                    buffer.append(next(stream))
                except StopIteration:
                    break
            while buffer:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer[idx]
                try:
                    buffer[idx] = next(stream)
                except StopIteration:
                    buffer.pop(idx)
