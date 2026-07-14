<p align="center">
  <img src="assets/tuga.svg" alt="Tuga Logo" width="120">
  <img src="assets/graph_icon.svg" alt="GNN Icon" width="100">
</p>

# Tuga-SP
**TugaSP** is a lightweight, easy-to-use graph neural network for materials and chemistry.
If Tuga can use it, so can you. *Waf!*

The model is an invariant Graph Neural Network (GNN) for material property prediction. It leverages the atomic and line (dual) graph representations, using Transformer attention mechanisms to achieve state-of-the-art accuracy on crystal structure tasks.

## Key Features
- **Hierarchical Graph Representation**: Nodes (atoms), Edges (bonds), and Triplets (angles).
- **Transformer Backbone**: Multi-head attention across all graph components.
- **Physics-Informed Embeddings**: Supports Pettifor embeddings and custom element features.
- **Parallel Processing**: Fast graph construction using multi-core processing.
- **Multi-GPU Training**: Built-in support for DDP training via PyTorch Lightning.
- **SOTA Modules**: SwiGLU activations, RMSNorm.
- **Built on the Shoulders of Giants**: Optimized with `PyTorch`, `PyTorch Geometric`, `PyTorch Lightning`, and `Pymatgen`.

## Installation

Use `uv` for lightning-fast dependency management, but `pip` works too.

### Using `uv` (Recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install in a venv
git clone https://github.com/ppdebreuck/tuga-sp.git
cd tuga-sp
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using `pip`

```bash
pip install -e .
```

To install with benchmarking dependencies (Hydra, WandB, Matminer):
```bash
pip install ".[benchmarks]"
```

## Quick Start

### 1. Training

TugaSP provides a high-level `train_model` function that handles the conversion of pymatgen structures to graphs and sets up the Lightning trainer.

```python
from tugasp.train import train_model
from matminer.datasets import load_dataset

# Load a sample dataset (requires matminer)
df = load_dataset("matbench_mp_e_form")
structures = df["structure"].tolist()
targets = df["e_form"].tolist()

# Train the model
model = train_model(
    train_structures=structures[:1000], # list of pymatgen structures
    train_targets=targets[:1000], # list of targets
    val_structures=structures[1000:1200], # list of pymatgen structures
    val_targets=targets[1000:1200], # list of targets
    max_epochs=50,
    batch_size=32,
    n_jobs=4 # Use 4 CPUs for graph building
)
```

### 2. Prediction

Easily predict properties for new structures using the `Predictor` class.

```python
from tugasp.inference import Predictor
from pymatgen.core import Structure

# Initialize predictor from a checkpoint
predictor = Predictor(model_path="path/to/checkpoint.ckpt")

# Predict for a single structure
struct = Structure.from_file("my_structure.cif") # pymatgen object
prediction = predictor.predict(struct)
print(f"Predicted Property: {prediction.item()}")

# Predict for a list of structures in parallel
structures = [struct1, struct2, struct3]
predictions = predictor.predict(structures, n_jobs=-1)
```

### Performance Benchmarks

Below are preliminary results on several [Matbench](https://matbench.materialsproject.org/) datasets.

*Note: These results are from a single fold without extensive hyperparameter tuning. Further improvements are expected with optimized settings.*

| Dataset | TugaSP (MAE) | coGN (MAE) | coNGN (MAE) | ALIGNN (MAE) | MODNet (MAE) | CGCNN (MAE) | Unit |
|---------|--------------|------------|-------------|--------------|--------------|-------------|------|
| Matbench E-form | 21 | 17 | 18 | 22 | 45 | 34 | meV/atom |
| Matbench Perovskites | 32 | 27 | 29 | 29 | 91 | 45 | meV/atom |
| Matbench Band Gaps | 158 | 156 | 170 | 186 | 220 | 297 | meV |

## Hardware Selection (GPU / Multi-GPU)

TugaSP is built on PyTorch Lightning and supports flexible hardware configuration.

### Training
By default, `train_model` uses `accelerator="auto"` and `devices="auto"`. To specify hardware:

```python
# Force single GPU training
model = train_model(..., accelerator="gpu", devices=1)

# Multi-GPU training (Distributed Data Parallel)
model = train_model(..., accelerator="gpu", devices=2)

# Specific GPU indices
model = train_model(..., accelerator="gpu", devices=[0, 2])

# CPU only
model = train_model(..., accelerator="cpu")
```

### Prediction
For prediction, you can specify the device during initialization:

```python
from tugasp.inference import Predictor

# Default is 'cuda' if available, otherwise 'cpu'
predictor = Predictor(model_path="best_model.ckpt")

# Force CPU inference
predictor = Predictor(model_path="best_model.ckpt", device="cpu")

# Specify a specific GPU
predictor = Predictor(model_path="best_model.ckpt", device="cuda:1")
```

> **Tip**: During training, PyTorch Lightning will print the hardware being used at the start of the process (e.g., `GPU available: True, used: True`). For the `Predictor`, you can check the `predictor.device` attribute to verify the current device.

## Data Format

TugaSP is designed to work seamlessly with [Pymatgen](https://pymatgen.org/) `Structure` objects.
- **Nodes**: Features are automatically derived from atomic numbers.
- **Edges**: Bonds are determined via distance cutoffs (default **5.0 Å**).
- **Angles**: Triplets are formed between all bonds sharing a common atom.

### Site Properties as Node Features

You can enrich node features with any per-site scalar or vector quantity stored in a pymatgen `Structure` (e.g. magnetic moments, forces, charges, spin vectors). Pass the property name(s) via `site_properties`:

```python
# Add site properties to your pymatgen structures
structure.add_site_property("mag_moment", [0.0, 2.2, ...])       # scalar per site
structure.add_site_property("force", [[0.1, 0.2, 0.3], ...]) # 3-vector per site

model = train_model(
    train_structures=structures,
    train_targets=targets,
    site_properties=["mag_moment"],              # single scalar property
    # site_properties=["mag_moment", "force"],   # multiple / vector properties
)
```

Scalars become `(N, 1)` and vectors become `(N, D)` tensors; multiple properties are concatenated to `(N, total_dim)` per batch. Missing properties on a structure are zero-filled automatically. The combined vector is projected via a 2-layer MLP to `d_model` and added to the atom embedding at the first layer.

### Global State Properties

You can condition predictions on graph-level scalar state variables stored in `Structure.properties`, such as pressure or temperature. Pass the property name(s) via `state_properties`:

```python
for structure in structures:
    structure.properties["pressure"] = 10.0

model = train_model(
    train_structures=structures,
    train_targets=targets,
    state_properties="pressure",
)
```

```python
for structure in structures:
    structure.properties["temperature"] = 300.0
    structure.properties["pressure"] = 1.0

model = train_model(
    train_structures=structures,
    train_targets=targets,
    state_properties=["temperature", "pressure"],
)
```

State properties are read as raw scalar floats and concatenated to `(B, state_property_dim)` per batch. Requested keys must exist on every structure. The state vector is projected to `d_model`, broadcast into node and edge updates, and concatenated into the final graph readout.

## On-the-Fly Data Loading & Adapters

For large datasets (e.g. millions of structures) where keeping all graphs in memory is impossible, TugaSP supports **on-the-fly graph construction** using an **Adapter Pattern**. Graphs are constructed in parallel inside CPU worker processes as training progresses.

You can specify the number of parallel CPU worker processes using the `num_workers` parameter in `train_model`.

### 1. Using Built-in Adapters (e.g. ASE LMDB/SQLite)

Built-in adapters simplify loading from standard files. For example, to train directly from an ASE `.aselmdb` database:

```python
from tugasp.train import train_model
from tugasp.data.adapters import AseLmdbAdapter

# Instantiate adapters pointing to ASE DB/LMDB files
train_adapter = AseLmdbAdapter("path/to/train.aselmdb")
val_adapter = AseLmdbAdapter("path/to/val.aselmdb")

model = train_model(
    train_structures=train_adapter,
    val_structures=val_adapter,
    num_workers=4,  # Spawns 4 background CPU processes to build graphs in parallel
    batch_size=64,
    max_epochs=100
)
```

Available built-in adapters:
- `ListStoreAdapter(structures, targets)`: Wraps raw in-memory structures and target lists.
- `PickleAdapter(pickle_path)`: Loads a list of structures/atoms from a `.pkl` file.
- `AseLmdbAdapter(path)`: Reads from ASE database/LMDB files or folders (uses `fairchem` if available, otherwise falls back to standard `ase.db`).

### 2. Creating a Custom Adapter

You can easily train on any database (SQLite, MySQL, MongoDB, JSON, etc.) by implementing a custom adapter that inherits from `BaseStoreAdapter` and implements `__len__` and `__getitem__`:

```python
from tugasp.train import train_model
from tugasp.data.adapters import BaseStoreAdapter

class CustomSqlAdapter(BaseStoreAdapter):
    def __init__(self, db_path):
        self.db_path = db_path
        self.length = self._count_rows()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        # 1. Fetch record from your database at index `idx`
        # 2. Convert to an ase.Atoms or pymatgen.Structure object
        # 3. Return a dictionary with the structure and potential target
        return {
            "structure": structure,  # pymatgen.Structure or ase.Atoms
            "y": target_value,       # Float or array target property (optional)
            "mat_id": f"id-{idx}"    # String ID for tracking (optional)
        }

# Pass the custom adapter directly to train_model
model = train_model(
    train_structures=CustomSqlAdapter("database.db"),
    num_workers=8,
    batch_size=128
)
```

For **streaming-only** stores that cannot support random access, implement `__iter__` (and `__len__` if known) instead of `__getitem__`, and set the class attribute `is_iterable = True` so the data module routes the adapter through the streaming pipeline (per-worker sharding and shuffle-buffering are handled for you).

### 3. Train/Validation Splitting

You can either provide independent `train_structures` and `val_structures` (each used in full), or carve a deterministic validation split out of a single dataset. To split one dataset, pass the **same** structures/adapter to both arguments and set `train_ratio < 1.0`:

```python
model = train_model(
    train_structures=adapter,   # same object (or same underlying data)
    val_structures=adapter,     # passed to both
    train_ratio=0.9,            # 90% train / 10% validation
    seed=42,                    # controls the split (and shuffling)
)
```

The split is assigned per-record via a deterministic hash of the index, so train and validation partitions are disjoint and reproducible across runs. With the default `train_ratio=1.0` no split is applied: `train_structures` is used fully and a separately-provided `val_structures` is used fully.

### 4. Dynamic-Cost Batching

To guard against GPU Out-of-Memory (OOM) errors when structures vary significantly in size, you can specify maximum node, edge, or triplet budgets per batch. Batches are built dynamically until these budgets are met:

```python
model = train_model(
    train_structures=train_adapter,
    val_structures=val_adapter,
    max_nodes_per_batch=10000,    # Max total atoms in a single batch
    max_edges_per_batch=50000,    # Max total bonds in a single batch
    max_triplets_per_batch=150000, # Max total angles in a single batch
)
```

## Config script for easy training - coming soon

Detailed configurations for Matbench and Materials Project benchmarks can be found in the `benchmark/` directory. These experiments use [Hydra](https://hydra.cc/) for configuration management.

```bash
python benchmark/matbench.py experiment=mp_e_form
```

## TODO
- [x] Add global state embedding (pressure, temperature, ...)
- [x] Add node embeddings from pymatgen site properties
- ... please make an issue/PR for other ideas!

## Contributing

TugaSP is meant to be a community project, and we are happy to receive feedback, ideas, issues, and pull requests! Feel free to open an issue or submit a PR if you have suggestions or would like to contribute.

## License
MIT License.

## Author
This software is written by [Pierre-Paul De Breuck](mailto:pierre-paul.debreuck@rub.de) with contributions from Paulo Pires and Miguel Marques.
For an up-to-date list, see the [Contributors on GitHub](https://github.com/ppdebreuck/tuga-sp/graphs/contributors).
