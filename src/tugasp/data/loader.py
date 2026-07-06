from typing import Optional, Any
import pytorch_lightning as L
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from .onthefly_dataset import OnTheFlyDataset, OnTheFlyIterableDataset
from .dataset import TugaGraphBuilder


class TugaDataModule(L.LightningDataModule):
    """
    Original list-based datamodule. Useful when graphs are built beforehand in memory.
    """
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        if self.train_data:
            return DataLoader(
                self.train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                follow_batch=["edge_index", "triplet_index"],
            )
        return None

    def val_dataloader(self):
        if self.val_data:
            return DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                follow_batch=["edge_index", "triplet_index"],
            )
        return None

    def test_dataloader(self):
        if self.test_data:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                follow_batch=["edge_index", "triplet_index"],
            )
        return None


class DynamicBatchWrapper(IterableDataset):
    """
    Iterates over a dataset and packs graphs greedily into batches until
    the budget for nodes, edges, or triplets is met.
    """
    def __init__(
        self,
        dataset,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        max_triplets: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_triplets = max_triplets

    def __iter__(self):
        batch = []
        cur_n, cur_e, cur_t = 0, 0, 0

        for g in self.dataset:
            n = g.num_nodes
            e = g.num_edges
            t = (
                g.triplet_index.size(1)
                if hasattr(g, "triplet_index") and g.triplet_index is not None
                else 0
            )

            # Yield batch if the incoming graph would exceed the budget limits
            if len(batch) > 0 and (
                (self.max_nodes and cur_n + n > self.max_nodes)
                or (self.max_edges and cur_e + e > self.max_edges)
                or (self.max_triplets and cur_t + t > self.max_triplets)
            ):
                yield batch
                batch = []
                cur_n, cur_e, cur_t = 0, 0, 0

            batch.append(g)
            cur_n += n
            cur_e += e
            cur_t += t

        if batch:
            yield batch


class OnTheFlyDataModule(L.LightningDataModule):
    """
    Lightning DataModule that handles on-the-fly graph construction and batching.
    Supports list-based, map-style, and iterable-style data store adapters.
    """
    def __init__(
        self,
        train_adapter: Optional[Any] = None,
        val_adapter: Optional[Any] = None,
        test_adapter: Optional[Any] = None,
        builder: Optional[TugaGraphBuilder] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        train_ratio: float = 1.0,
        seed: int = 42,
        max_nodes_per_batch: Optional[int] = None,
        max_edges_per_batch: Optional[int] = None,
        max_triplets_per_batch: Optional[int] = None,
        shuffle: bool = True,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.train_adapter = train_adapter
        self.val_adapter = val_adapter
        self.test_adapter = test_adapter
        self.builder = builder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed
        self.max_nodes_per_batch = max_nodes_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.max_triplets_per_batch = max_triplets_per_batch
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def _get_dataset(self, adapter, is_train: bool):
        if adapter is None:
            return None

        # Check if the adapter is iterable-only
        is_iterable = not hasattr(adapter, "__getitem__") and hasattr(adapter, "__iter__")

        if is_iterable:
            return OnTheFlyIterableDataset(
                adapter=adapter,
                builder=self.builder,
                train_ratio=self.train_ratio if is_train else 1.0,
                is_train=is_train,
                seed=self.seed,
                shuffle=self.shuffle if is_train else False,
                buffer_size=self.buffer_size,
            )
        else:
            return OnTheFlyDataset(
                adapter=adapter,
                builder=self.builder,
                train_ratio=self.train_ratio,
                is_train=is_train,
                seed=self.seed,
            )

    def _get_dataloader(self, dataset, shuffle: bool = False):
        if dataset is None:
            return None

        use_dynamic = (
            self.max_nodes_per_batch is not None
            or self.max_edges_per_batch is not None
            or self.max_triplets_per_batch is not None
        )

        persistent_workers = self.num_workers > 0

        if not use_dynamic:
            is_iterable = isinstance(dataset, IterableDataset)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle if not is_iterable else False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                follow_batch=["edge_index", "triplet_index"],
            )
        else:
            wrapped_dataset = DynamicBatchWrapper(
                dataset,
                max_nodes=self.max_nodes_per_batch,
                max_edges=self.max_edges_per_batch,
                max_triplets=self.max_triplets_per_batch,
            )
            return DataLoader(
                wrapped_dataset,
                batch_size=None,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                follow_batch=["edge_index", "triplet_index"],
            )

    def train_dataloader(self):
        dataset = self._get_dataset(self.train_adapter, is_train=True)
        return self._get_dataloader(dataset, shuffle=self.shuffle)

    def val_dataloader(self):
        # If val_adapter is the same as train_adapter, self._get_dataset
        # automatically filters val rows based on train_ratio and is_train=False
        dataset = self._get_dataset(self.val_adapter, is_train=False)
        return self._get_dataloader(dataset, shuffle=False)

    def test_dataloader(self):
        dataset = self._get_dataset(self.test_adapter, is_train=False)
        return self._get_dataloader(dataset, shuffle=False)


def create_loader(graphs, batch_size=32, shuffle=False):
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=["edge_index", "triplet_index"],
    )
