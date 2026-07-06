"""
Full run smoke test: training loop with vector site properties.

Structures: simple NaCl-like cubic cells with two site properties:
  - mag_moment: scalar (spin)
  - force:      3-vector (force on each atom)

Both are combined into a 4-dim per-node feature vector and projected into
the model alongside the standard atom embedding.

Run with:
    uv run python tests/full_run_site_prop.py
"""

import numpy as np
from pymatgen.core import Lattice, Structure

from tugasp.train import train_model


def make_structure(a: float) -> Structure:
    lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
    s = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    # Scalar site property
    s.add_site_property("mag_moment", [float(a % 1.0), float((a * 1.3) % 2.0)])
    # Vector site property (3-component force per site)
    s.add_site_property(
        "force",
        [
            [np.sin(a), np.cos(a), a * 0.01],
            [-np.sin(a), -np.cos(a), -a * 0.01],
        ],
    )
    return s


def main():
    print("Building structures with vector site properties...")
    structures = [make_structure(5.0 + i * 0.2) for i in range(8)]
    targets = [s.volume for s in structures]

    train_structs = structures[:6]
    train_targets = targets[:6]
    val_structs = structures[6:]
    val_targets = targets[6:]

    print(f"Train: {len(train_structs)} structures | Val: {len(val_structs)} structures")
    print("Site properties: mag_moment (scalar) + force (3-vec) => 4-dim node features\n")

    model = train_model(
        train_structures=train_structs,
        train_targets=train_targets,
        val_structures=val_structs,
        val_targets=val_targets,
        site_properties=["mag_moment", "force"],
        d_model=64,
        num_layers=2,
        nhead=4,
        max_epochs=5,
        batch_size=2,
        accelerator="cpu",
        logger=False,
    )

    print(f"\nsite_property_dim stored in hparams: {model.hparams.site_property_dim}")
    print(f"site_properties stored in hparams:  {model.hparams.site_properties}")
    print("\nFull run completed successfully.")


if __name__ == "__main__":
    main()
