from typing import Callable, List, Optional, Union

import pytorch_lightning as L
import torch

from .data.dataset import TugaGraphBuilder
from .data.loader import TugaDataModule
from .models.lit_model import LitTugaSP


def train_model(
    train_structures,
    train_targets: Optional[Union[List, torch.Tensor]] = None,
    val_structures=None,
    val_targets: Optional[Union[List, torch.Tensor]] = None,
    test_structures=None,
    test_targets: Optional[Union[List, torch.Tensor]] = None,
    target_mapper: Optional[Callable] = None,
    # Model Hyperparameters (Standardized)
    d_model=128,
    d_out=1,
    num_layers=3,
    nhead=4,
    dff_ratio=4,
    dropout=0.1,
    lr=1e-4,
    loss_type="l1",
    # SOTA improvements
    use_swiglu=False,
    use_lattice_encoding=True,
    use_dihedrals=False,
    # Site properties
    site_properties=None,
    # Structure-level state properties
    state_properties=None,
    # Training Params
    batch_size=32,
    max_epochs=100,
    accelerator="auto",
    devices="auto",
    logger=True,
    callbacks=None,
    # Data processing parallelism
    n_jobs: int = 1,
    **kwargs,
):
    """
    Easy training function supporting both explicit targets and mapper.

    Args:
        train_structures: List of pymatgen Structures.
        train_targets: List or Tensor of targets corresponding to train_structures.
        target_mapper: Function mapping Structure -> target (alternative to explicit targets).
        n_jobs: Number of parallel jobs for graph building. Follows joblib conventions:
                - 1 (default): Sequential processing.
                - -1: Use all available CPUs.
                - Positive integer: Use exactly n_jobs workers.
    """

    # 1. Process Data
    graph_builder = TugaGraphBuilder(
        site_properties=site_properties,
        state_properties=state_properties,
    )

    def process_data(structures, targets=None):
        if not structures:
            return None

        # If already graphs, return as-is
        if hasattr(structures[0], "edge_index"):
            return list(structures)

        # Validation checks
        if targets is not None:
            if len(structures) != len(targets):
                raise ValueError(
                    f"Mismatch: {len(structures)} structures vs {len(targets)} targets"
                )
            # Build properties list for get_graphs
            properties = [{"y": y} for y in targets]
        elif target_mapper is not None:
            # Use mapper to generate targets
            properties = [{"y": target_mapper(s)} for s in structures]
        else:
            raise ValueError(
                "Must provide either `targets` (list) or `target_mapper` (func)."
            )

        # Use parallel graph building
        return graph_builder.get_graphs(
            structures, properties=properties, n_jobs=n_jobs
        )

    train_data = process_data(train_structures, train_targets)
    val_data = process_data(val_structures, val_targets)
    test_data = process_data(test_structures, test_targets)

    # Infer site_property_dim from the first graph
    site_property_dim = 0
    if site_properties and train_data and hasattr(train_data[0], "site_feat") and train_data[0].site_feat is not None:
        site_property_dim = train_data[0].site_feat.shape[-1]

    # Infer state_property_dim from the first graph
    state_property_dim = 0
    if state_properties and train_data and hasattr(train_data[0], "state_feat") and train_data[0].state_feat is not None:
        state_property_dim = train_data[0].state_feat.shape[-1]

    # 2. Setup DataModule
    dm = TugaDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=batch_size,
    )

    # 3. Setup Model
    model = LitTugaSP(
        d_model=d_model,
        d_out=d_out,
        num_layers=num_layers,
        nhead=nhead,
        dff_ratio=dff_ratio,
        dropout=dropout,
        lr=lr,
        loss_type=loss_type,
        use_swiglu=use_swiglu,
        use_lattice_encoding=use_lattice_encoding,
        use_dihedrals=use_dihedrals,
        site_property_dim=site_property_dim,
        site_properties=site_properties,
        state_property_dim=state_property_dim,
        state_properties=state_properties,
        **kwargs,
    )

    # 4. Trainer
    # Handle missing validation data to avoid Trainer crashing on None dataloader
    if not val_data:
        if "limit_val_batches" not in kwargs:
            kwargs["limit_val_batches"] = 0.0
        if "num_sanity_val_steps" not in kwargs:
            kwargs["num_sanity_val_steps"] = 0

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        **kwargs,  # Pass kwargs like limit_val_batches if explicitly set
    )

    trainer.fit(model, datamodule=dm)

    if test_data:
        trainer.test(model, datamodule=dm)

    return model
