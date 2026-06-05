import pytest
import torch
from pymatgen.core import Lattice, Structure

from tugasp.data.dataset import TugaGraphBuilder
from tugasp.inference import Predictor
from tugasp.train import train_model


def _make_nacl(a=5.64):
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    return Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_state_feat_extraction_scalar_string():
    s = _make_nacl()
    s.properties["pressure"] = 10.0

    builder = TugaGraphBuilder(state_properties="pressure")
    graph = builder.get_graph(s)

    assert graph.state_feat is not None
    assert graph.state_feat.shape == (1, 1)
    assert graph.state_feat.dtype == torch.float32
    assert torch.allclose(graph.state_feat, torch.tensor([[10.0]]))


def test_state_feat_extraction_multiple_scalars():
    s = _make_nacl()
    s.properties["temperature"] = 300.0
    s.properties["pressure"] = 1.0

    builder = TugaGraphBuilder(state_properties=["temperature", "pressure"])
    graph = builder.get_graph(s)

    assert graph.state_feat.shape == (1, 2)
    assert torch.allclose(graph.state_feat, torch.tensor([[300.0, 1.0]]))


def test_missing_state_property_raises():
    s = _make_nacl()
    builder = TugaGraphBuilder(state_properties="pressure")

    with pytest.raises(ValueError, match="Missing required state property"):
        builder.get_graph(s)


def test_non_scalar_state_property_raises():
    s = _make_nacl()
    s.properties["pressure"] = [1.0, 2.0]
    builder = TugaGraphBuilder(state_properties="pressure")

    with pytest.raises(ValueError, match="must be a scalar"):
        builder.get_graph(s)


def test_non_numeric_state_property_raises():
    s = _make_nacl()
    s.properties["pressure"] = "high"
    builder = TugaGraphBuilder(state_properties="pressure")

    with pytest.raises(ValueError, match="must be a numeric scalar"):
        builder.get_graph(s)


def test_training_with_state_properties():
    structures = []
    targets = []
    for i in range(5):
        s = _make_nacl(5.0 + i * 0.1)
        s.properties["pressure"] = float(i)
        structures.append(s)
        targets.append(s.volume)

    model = train_model(
        train_structures=structures,
        train_targets=targets,
        state_properties=["pressure"],
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )

    assert model.hparams.state_properties == ["pressure"]
    assert model.hparams.state_property_dim == 1


def test_predictor_restores_state_properties_from_model():
    structures = []
    targets = []
    for i in range(5):
        s = _make_nacl(5.0 + i * 0.1)
        s.properties["pressure"] = float(i)
        structures.append(s)
        targets.append(s.volume)

    model = train_model(
        train_structures=structures,
        train_targets=targets,
        state_properties=["pressure"],
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )

    predictor = Predictor(model=model, device="cpu")
    test_structure = _make_nacl(6.0)
    test_structure.properties["pressure"] = 2.5
    pred = predictor.predict(test_structure)

    assert pred.ndim == 1
    assert pred.shape[0] == 1
