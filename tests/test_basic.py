import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from tugasp.data.dataset import TugaGraphBuilder
from tugasp.inference import Predictor
from tugasp.train import train_model


def test_graph_creation():
    print("Testing graph creation...")
    a = 5.64
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test Vector Property (e.g. DOS of length 10)
    dos_fake = np.random.rand(10)
    builder = TugaGraphBuilder()
    graph = builder.get_graph(structure, properties={"y": dos_fake})

    assert graph.x.dtype == torch.long
    assert graph.y.shape == (10,), f"Expected shape (10,), got {graph.y.shape}"

    print(f"Graph created with vector target size {graph.y.shape[0]}.")
    return structure


def test_training_flow_explicit_targets():
    print("Testing training flow with EXPLICIT targets...")
    structures = []
    targets = []
    for i in range(5):
        a = 5.0 + i * 0.1
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        s = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        structures.append(s)
        targets.append(s.volume)

    # Standardized Params + Explicit Targets
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
    print("Explicit targets training finished.")


def test_training_flow_vector_mapper():
    print("Testing VECTOR training flow (DOS) with Mapper...")
    structures = []
    for i in range(5):
        a = 5.0 + i * 0.1
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        s = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        structures.append(s)

    def target_mapper_vector(s):
        # Return random 20-dim vector
        return np.random.rand(20)

    model = train_model(
        train_structures=structures,
        target_mapper=target_mapper_vector,
        d_model=32,
        d_out=20,
        num_layers=1,
        loss_type="mse",
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )
    print("Vector training finished.")
    return model


def test_inference(model):
    print("Testing inference...")
    a = 6.0
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    s = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    predictor = Predictor(model=model, device="cpu")
    pred = predictor.predict(s)
    print(f"Prediction shape: {pred.shape}")


if __name__ == "__main__":
    try:
        s = test_graph_creation()
        test_training_flow_explicit_targets()
        model = test_training_flow_vector_mapper()
        test_inference(model)
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
