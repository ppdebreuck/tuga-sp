from pymatgen.core import Lattice, Structure

from tugasp.train import train_model


def test_training_flow_with_dihedrals():
    print("Testing training flow WITH DIHEDRALS...")
    structures = []
    targets = []
    # Create CsCl structures to ensure 4-body dihedral terms are present.
    for i in range(5):
        a = 4.0 + i * 0.1
        lattice = Lattice.from_parameters(a, a, a, 90, 90, 90)
        s = Structure(lattice, ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        s.make_supercell([2, 2, 2])  # Make it larger to ensure connections
        structures.append(s)
        targets.append(s.volume)

    # Standardized Params + Explicit Targets + Dihedrals
    model = train_model(
        train_structures=structures,
        train_targets=targets,
        d_model=32,
        num_layers=1,
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        logger=False,
        use_dihedrals=True,  # Enable Dihedrals
    )
    print("Dihedral training finished successfully.")


if __name__ == "__main__":
    try:
        test_training_flow_with_dihedrals()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
