import pytorch_lightning as L
from torch_geometric.loader import DataLoader


class TugaDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data=None,
        val_data=None,
        test_data=None,
        batch_size=32,
        num_workers=0,
    ):
        """
        Args:
            train_data: List of PyG Data objects or structures for training.
            val_data: List of PyG Data objects or structures for validation.
            test_data: List of PyG Data objects or structures for testing.
        """
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


def create_loader(graphs, batch_size=32, shuffle=False):
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=["edge_index", "triplet_index"],
    )
