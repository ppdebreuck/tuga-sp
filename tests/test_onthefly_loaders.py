import os
import tempfile
import pickle
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from ase import Atoms

from tugasp.data.dataset import TugaGraphBuilder
from tugasp.data.adapters import ListStoreAdapter, PickleAdapter
from tugasp.data.onthefly_dataset import OnTheFlyDataset
from tugasp.data.loader import OnTheFlyDataModule, DynamicBatchWrapper

def make_dummy_structure(a: float) -> Structure:
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    s = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.add_site_property("mag_moment", [0.5, 1.5])
    return s

def make_dummy_atoms(a: float) -> Atoms:
    atoms = Atoms("NaCl", positions=[[0, 0, 0], [a/2, a/2, a/2]], cell=[a, a, a], pbc=True)
    atoms.set_initial_magnetic_moments([0.5, 1.5])
    return atoms

def test_list_store_adapter_and_graph_equivalence():
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(5)]
    targets = [1.2 * i for i in range(5)]
    
    # Standard builder
    builder = TugaGraphBuilder(site_properties=["mag_moment"])
    
    # 1. Offline graph building
    offline_graphs = builder.get_graphs(structures, properties=[{"y": y} for y in targets])
    
    # 2. On-the-fly graph building via ListStoreAdapter
    adapter = ListStoreAdapter(structures, targets)
    dataset = OnTheFlyDataset(adapter, builder)
    
    assert len(dataset) == len(structures)
    for i in range(len(structures)):
        g_off = offline_graphs[i]
        g_on = dataset[i]
        
        # Verify equivalence
        assert torch.allclose(g_off.x, g_on.x)
        assert torch.allclose(g_off.edge_index, g_on.edge_index)
        assert torch.allclose(g_off.edge_attr, g_on.edge_attr)
        assert torch.allclose(g_off.triplet_index, g_on.triplet_index)
        assert torch.allclose(g_off.angle_attr, g_on.angle_attr)
        assert torch.allclose(g_off.y, g_on.y)
        assert torch.allclose(g_off.site_feat, g_on.site_feat)

def test_get_graph_from_atoms_equivalence():
    # Verify that get_graph_from_atoms on an Atoms object produces the exact same graph as get_graph on equivalent Structure
    struct = make_dummy_structure(5.5)
    atoms = make_dummy_atoms(5.5)
    
    builder = TugaGraphBuilder(site_properties=["mag_moment"])
    
    g_struct = builder.get_graph(struct)
    g_atoms = builder.get_graph_from_atoms(atoms)
    
    assert torch.allclose(g_struct.x, g_atoms.x)
    assert torch.allclose(g_struct.edge_index, g_atoms.edge_index)
    assert torch.allclose(g_struct.edge_attr, g_atoms.edge_attr)
    assert torch.allclose(g_struct.triplet_index, g_atoms.triplet_index)
    assert torch.allclose(g_struct.angle_attr, g_atoms.angle_attr)
    assert torch.allclose(g_struct.lattice_params, g_atoms.lattice_params)
    assert torch.allclose(g_struct.site_feat, g_atoms.site_feat)

def test_pickle_adapter():
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(3)]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = os.path.join(tmpdir, "structures.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(structures, f)
            
        adapter = PickleAdapter(pkl_path)
        assert len(adapter) == 3
        assert isinstance(adapter[0]["structure"], Structure)

def test_dataset_splitting():
    # Test split ratio logic
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(20)]
    targets = [float(i) for i in range(20)]
    adapter = ListStoreAdapter(structures, targets)
    builder = TugaGraphBuilder()
    
    train_ds = OnTheFlyDataset(adapter, builder, train_ratio=0.7, is_train=True, seed=42)
    val_ds = OnTheFlyDataset(adapter, builder, train_ratio=0.7, is_train=False, seed=42)
    
    # Assert splits are partition-disjoint and cover the full range
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    
    assert train_indices.isdisjoint(val_indices)
    assert len(train_indices) + len(val_indices) == 20
    assert len(train_indices) > 10

def test_dynamic_batch_wrapper():
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(10)]
    adapter = ListStoreAdapter(structures)
    builder = TugaGraphBuilder()
    dataset = OnTheFlyDataset(adapter, builder)
    
    # Let's check max_nodes constraint
    # Each structure has 2 atoms
    wrapper = DynamicBatchWrapper(dataset, max_nodes=5)
    batches = list(wrapper)
    
    # Since each graph has 2 nodes:
    # Batch 1: 2 graphs (4 nodes)
    # Batch 2: 2 graphs (4 nodes)
    # Batch 3: 1 graph (2 nodes)
    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 2
        total_nodes = sum(g.num_nodes for g in batch)
        assert total_nodes <= 5

def test_ase_lmdb_adapter():
    from ase.db import connect
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = connect(db_path)
        
        atoms1 = make_dummy_atoms(5.2)
        atoms2 = make_dummy_atoms(5.8)
        
        db.write(atoms1, data={"y": 1.5, "mat_id": "test-1"})
        db.write(atoms2, data={"y": 2.5, "mat_id": "test-2"})
        
        from tugasp.data.adapters import AseLmdbAdapter
        adapter = AseLmdbAdapter(db_path)
        
        assert len(adapter) == 2
        
        item1 = adapter[0]
        assert abs(item1["y"] - 1.5) < 1e-5
        assert item1["mat_id"] == "test-1"
        assert isinstance(item1["structure"], Atoms)


class _IterableAdapter:
    """Stream-only adapter (no __getitem__) for testing the iterable pipeline."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield {"structure": make_dummy_structure(5.0 + i * 0.1), "y": float(i)}


def _count(dataset):
    return sum(1 for _ in dataset)


def test_separate_val_adapter_default_ratio_not_empty():
    # Regression: separate val_adapter with default train_ratio=1.0 must be used
    # in full, not silently emptied.
    train_ad = ListStoreAdapter([make_dummy_structure(5.0 + i * 0.1) for i in range(10)])
    val_ad = ListStoreAdapter([make_dummy_structure(6.0 + i * 0.1) for i in range(4)])
    dm = OnTheFlyDataModule(train_adapter=train_ad, val_adapter=val_ad, builder=TugaGraphBuilder())

    assert len(dm._get_dataset(train_ad, is_train=True)) == 10
    assert len(dm._get_dataset(val_ad, is_train=False)) == 4


def test_iterable_shared_adapter_split_is_disjoint():
    # Regression: shared iterable adapter with train_ratio<1.0 must split into
    # disjoint train/val partitions (no leakage of train rows into val).
    adapter = _IterableAdapter(20)
    dm = OnTheFlyDataModule(
        train_adapter=adapter, val_adapter=adapter,
        builder=TugaGraphBuilder(), train_ratio=0.7,
    )
    n_train = _count(dm._get_dataset(adapter, is_train=True))
    n_val = _count(dm._get_dataset(adapter, is_train=False))

    assert n_train + n_val == 20
    assert n_train > 0 and n_val > 0
    assert n_train > n_val  # 70/30 split


def test_iterable_adapter_flag_routes_to_iterable_pipeline():
    # Regression: a BaseStoreAdapter subclass opting in via is_iterable=True must
    # be routed through the iterable pipeline (which never calls __getitem__).
    from tugasp.data.adapters import BaseStoreAdapter
    from tugasp.data.onthefly_dataset import OnTheFlyIterableDataset

    class StreamingAdapter(BaseStoreAdapter):
        is_iterable = True

        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __iter__(self):
            for i in range(self.n):
                yield {"structure": make_dummy_structure(5.0 + i * 0.1), "y": float(i)}

    adapter = StreamingAdapter(6)
    dm = OnTheFlyDataModule(train_adapter=adapter, builder=TugaGraphBuilder())
    ds = dm._get_dataset(adapter, is_train=True)

    assert isinstance(ds, OnTheFlyIterableDataset)
    assert _count(ds) == 6  # would raise NotImplementedError if routed map-style


def _run_wrapper_across_workers(wrapper, num_workers, monkeypatch_target="tugasp.data.loader.get_worker_info"):
    import tugasp.data.loader as loader_mod
    from unittest import mock

    seen = []
    for wid in range(num_workers):
        worker = type("W", (), {"id": wid, "num_workers": num_workers})()
        with mock.patch.object(loader_mod, "get_worker_info", return_value=worker):
            for batch in wrapper:
                seen.extend(id(g) for g in batch)
    return seen


def test_dynamic_wrapper_shards_map_style_across_workers():
    # Regression: DynamicBatchWrapper must not emit the full map-style dataset in
    # every worker (which would duplicate data num_workers times).
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(8)]
    dataset = OnTheFlyDataset(ListStoreAdapter(structures), TugaGraphBuilder())
    wrapper = DynamicBatchWrapper(dataset, max_nodes=100)

    seen = _run_wrapper_across_workers(wrapper, num_workers=2)
    assert len(seen) == 8            # each graph emitted exactly once
    assert len(set(seen)) == 8       # no duplicates across workers


def test_dynamic_wrapper_shuffle_changes_order_and_preserves_set():
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(12)]
    dataset = OnTheFlyDataset(ListStoreAdapter(structures, [float(i) for i in range(12)]), TugaGraphBuilder())

    ordered = [g.y.item() for batch in DynamicBatchWrapper(dataset, max_nodes=3, shuffle=False)
               for g in batch]
    wrapper = DynamicBatchWrapper(dataset, max_nodes=3, shuffle=True, seed=0)
    epoch1 = [g.y.item() for batch in wrapper for g in batch]
    epoch2 = [g.y.item() for batch in wrapper for g in batch]

    # Shuffling changes the emission order but preserves the multiset of items,
    # and successive epochs differ (epoch-dependent seed).
    assert sorted(epoch1) == sorted(ordered)
    assert epoch1 != ordered
    assert epoch1 != epoch2


# --- End-to-end tests with real DataLoader worker processes (num_workers>0) ---
# These spawn actual subprocesses, exercising the sharding logic for real rather
# than mocking get_worker_info. Everything referenced must be picklable and defined
# at module level (macOS/Windows use the 'spawn' start method).

def _collect_targets(loader):
    ys = []
    for batch in loader:
        ys.extend(batch.y.view(-1).tolist())
    return ys


def test_real_workers_dynamic_batching_full_coverage_no_duplication():
    n = 12
    structures = [make_dummy_structure(5.0 + i * 0.1) for i in range(n)]
    targets = [float(i) for i in range(n)]
    dm = OnTheFlyDataModule(
        train_adapter=ListStoreAdapter(structures, targets),
        builder=TugaGraphBuilder(),
        num_workers=2,
        max_nodes_per_batch=5,  # dynamic-cost batching -> DynamicBatchWrapper
        shuffle=True,
        seed=0,
    )
    ys = _collect_targets(dm.train_dataloader())
    assert sorted(ys) == targets


def test_real_workers_iterable_full_coverage_no_duplication():
    n = 12
    dm = OnTheFlyDataModule(
        train_adapter=_IterableAdapter(n),
        builder=TugaGraphBuilder(),
        batch_size=3,
        num_workers=2,
        shuffle=True,
        seed=0,
    )
    ys = _collect_targets(dm.train_dataloader())
    assert sorted(ys) == [float(i) for i in range(n)]
