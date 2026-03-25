<p align="center">
  <img src="assets/tuga.svg" alt="Tuga Logo" width="120">
  <img src="assets/graph_icon.svg" alt="GNN Icon" width="100">
</p>

# Tuga-SP
**TugaSP** is a lightweight, easy-to-use graph model designed to make graph work feel effortless.  
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
git clone git@gitlab.ruhr-uni-bochum.de:aiims/tuga-sp.git
cd tuga-sp
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Using `pip`

```bash
pip install -e .
```

To install with benchmarking dependencies (Hydra, WandB, Matminer): -> this coming soon, future commit
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

| Dataset | TugaSP (MAE) | ALIGNN (MAE) | Unit |
|---------|--------------|--------------|------|
| Matbench E-form | 21 | 17 | meV/atom |
| Matbench Perovskites | 32 | 27 | meV/atom |
| Matbench Band Gaps | 158 | 156 | meV |

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

## Config script for easy training - coming soon

Detailed configurations for Matbench and Materials Project benchmarks can be found in the `benchmark/` directory. These experiments use [Hydra](https://hydra.cc/) for configuration management.

```bash
python benchmark/matbench.py experiment=mp_e_form
```

## TODO
- [ ] Add global state embedding (pressure, temperature, ...)
- [x] Add node embeddings from pymatgen site properties
- ... please make an issue/PR for other ideas!

## License
MIT License.

## Author
Pierre-Paul De Breuck
