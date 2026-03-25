import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from tugasp.data.dataset import TugaGraphBuilder
from tugasp.inference import Predictor
from tugasp.train import train_model


def _make_nacl(a=5.64):
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    return Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_graph_creation():
    structure = _make_nacl()

    dos_fake = np.random.rand(10)
    builder = TugaGraphBuilder()
    graph = builder.get_graph(structure, properties={"y": dos_fake})

    assert graph.x.dtype == torch.long
    assert graph.y.shape == (10,), f"Expected shape (10,), got {graph.y.shape}"


def test_graph_scalar_target():
    structure = _make_nacl()
    builder = TugaGraphBuilder()
    graph = builder.get_graph(structure, properties={"y": structure.volume})

    assert graph.y.shape == (1,), f"Expected shape (1,), got {graph.y.shape}"
    assert graph.y.dtype == torch.float32


def test_graph_has_edges_and_triplets():
    structure = _make_nacl()
    builder = TugaGraphBuilder()
    graph = builder.get_graph(structure)

    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 1
    assert graph.edge_index.shape[1] > 0
    assert graph.triplet_index.shape[0] == 2
    assert graph.angle_attr.shape[1] == 1


def test_training_flow_explicit_targets():
    structures = [_make_nacl(5.0 + i * 0.1) for i in range(5)]
    targets = [s.volume for s in structures]

    train_model(
        train_structures=structures,
        train_targets=targets,
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )


def test_training_flow_vector_mapper():
    structures = [_make_nacl(5.0 + i * 0.1) for i in range(5)]

    model = train_model(
        train_structures=structures,
        target_mapper=lambda s: np.random.rand(20),
        d_model=32,
        d_out=20,
        num_layers=1,
        loss_type="mse",
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )
    assert model is not None


def test_inference_single():
    structures = [_make_nacl(5.0 + i * 0.1) for i in range(5)]
    targets = [s.volume for s in structures]

    model = train_model(
        train_structures=structures,
        train_targets=targets,
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )

    predictor = Predictor(model=model, device="cpu")
    pred = predictor.predict(_make_nacl(6.0))

    assert pred.ndim == 1
    assert pred.shape[0] == 1


def test_inference_batch():
    structures = [_make_nacl(5.0 + i * 0.1) for i in range(5)]
    targets = [s.volume for s in structures]

    model = train_model(
        train_structures=structures,
        train_targets=targets,
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )

    test_structures = [_make_nacl(5.5), _make_nacl(6.0), _make_nacl(6.5)]
    predictor = Predictor(model=model, device="cpu")
    preds = predictor.predict(test_structures)

    assert preds.shape == (3, 1)


if __name__ == "__main__":
    try:
        test_graph_creation()
        test_graph_scalar_target()
        test_graph_has_edges_and_triplets()
        test_training_flow_explicit_targets()
        test_training_flow_vector_mapper()
        test_inference_single()
        test_inference_batch()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
