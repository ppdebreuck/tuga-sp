import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from tugasp.data.dataset import TugaGraphBuilder
from tugasp.train import train_model


def _make_nacl(a=5.64):
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    return Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_site_feat_extraction_scalar():
    s = _make_nacl()
    s.add_site_property("mag_moment", [0.0, 2.2])
    builder = TugaGraphBuilder(site_properties=["mag_moment"])
    graph = builder.get_graph(s)
    assert graph.site_feat is not None
    assert graph.site_feat.shape == (2, 1), f"Expected (2, 1), got {graph.site_feat.shape}"
    assert graph.site_feat.dtype == torch.float32


def test_site_feat_extraction_vector():
    s = _make_nacl()
    s.add_site_property("force", [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    builder = TugaGraphBuilder(site_properties=["force"])
    graph = builder.get_graph(s)
    assert graph.site_feat is not None
    assert graph.site_feat.shape == (2, 3), f"Expected (2, 3), got {graph.site_feat.shape}"
    assert graph.site_feat.dtype == torch.float32


def test_site_feat_extraction_mixed():
    s = _make_nacl()
    s.add_site_property("mag_moment", [0.0, 2.2])
    s.add_site_property("force", [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    builder = TugaGraphBuilder(site_properties=["mag_moment", "force"])
    graph = builder.get_graph(s)
    assert graph.site_feat is not None
    assert graph.site_feat.shape == (2, 4), f"Expected (2, 4), got {graph.site_feat.shape}"


def test_training_with_site_properties():
    structures = []
    targets = []
    for i in range(5):
        a = 5.0 + i * 0.1
        s = _make_nacl(a)
        s.add_site_property("mag_moment", [float(i), float(i) * 0.5])
        structures.append(s)
        targets.append(s.volume)

    train_model(
        train_structures=structures,
        train_targets=targets,
        site_properties=["mag_moment"],
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )


def test_missing_site_property_robustness():
    # One structure has mag_moment, the other doesn't
    s1 = _make_nacl(5.0)
    s1.add_site_property("mag_moment", [0.0, 1.0])
    s2 = _make_nacl(5.5)  # no mag_moment

    builder = TugaGraphBuilder(site_properties=["mag_moment"])
    g1 = builder.get_graph(s1)
    g2 = builder.get_graph(s2)

    assert g1.site_feat.shape == (2, 1)
    assert g2.site_feat.shape == (2, 1)
    # Missing property should be zero-filled
    assert torch.all(g2.site_feat == 0.0)


if __name__ == "__main__":
    test_site_feat_extraction_scalar()
    test_site_feat_extraction_vector()
    test_site_feat_extraction_mixed()
    test_training_with_site_properties()
    test_missing_site_property_robustness()
    print("ALL SITE PROPERTY TESTS PASSED")
