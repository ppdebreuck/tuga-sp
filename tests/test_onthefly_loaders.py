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
