from typing import Callable, List, Optional, Union, Any
import pytorch_lightning as L
import torch

from .data.dataset import TugaGraphBuilder
from .data.adapters import BaseStoreAdapter, ListStoreAdapter
from .data.loader import OnTheFlyDataModule
from .models.lit_model import LitTugaSP


def train_model(
    train_structures,
    train_targets: Optional[Union[List, torch.Tensor, Any]] = None,
    val_structures=None,
    val_targets: Optional[Union[List, torch.Tensor, Any]] = None,
    test_structures=None,
    test_targets: Optional[Union[List, torch.Tensor, Any]] = None,
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
    num_workers: int = 0,
    train_ratio: float = 1.0,
    max_nodes_per_batch: Optional[int] = None,
    max_edges_per_batch: Optional[int] = None,
    max_triplets_per_batch: Optional[int] = None,
    **kwargs,
):
    """
    Easy training function supporting lists of structures/targets and on-the-fly store adapters.

    Args:
        train_structures: List of pymatgen Structures/Atoms or a BaseStoreAdapter.
        train_targets: List or Tensor of targets corresponding to train_structures.
        target_mapper: Function mapping Structure -> target (alternative to explicit targets).
        n_jobs: Number of parallel jobs for offline/dummy graph building.
        num_workers: Number of CPU worker processes used by PyTorch DataLoader for on-the-fly graph construction.
        train_ratio: Fraction of data used for training (rest is val split) if using shared adapter.
    """
    # 1. Process Data & Wrap in Adapters
    def get_adapter(structs, targets):
        if structs is None:
            return None
        if isinstance(structs, BaseStoreAdapter) or hasattr(structs, "__getitem__") or hasattr(structs, "__iter__"):
            # If it's a standard list, wrap in ListStoreAdapter (unless it already contains dicts with structures)
            if isinstance(structs, list) and len(structs) > 0:
                first = structs[0]
                if not isinstance(first, dict) or "structure" not in first:
                    # Apply target mapper if provided
                    if targets is None and target_mapper is not None:
                        resolved_targets = [target_mapper(s) for s in structs]
                    else:
                        resolved_targets = targets
                    return ListStoreAdapter(structs, resolved_targets)
            return structs
        # Apply target mapper if provided
        resolved_targets = targets
        if targets is None and target_mapper is not None:
            resolved_targets = [target_mapper(s) for s in structs]
        return ListStoreAdapter(structs, resolved_targets)

    train_adapter = get_adapter(train_structures, train_targets)
    val_adapter = get_adapter(val_structures, val_targets)
    test_adapter = get_adapter(test_structures, test_targets)

    graph_builder = TugaGraphBuilder(
        site_properties=site_properties,
        state_properties=state_properties,
    )

    # Infer site_property_dim and state_property_dim from the first graph
    site_property_dim = 0
    state_property_dim = 0

    if train_adapter is not None:
        try:
            if hasattr(train_adapter, "__getitem__"):
                first_item = train_adapter[0]
            else:
                first_item = next(iter(train_adapter))
            
            if isinstance(first_item, dict):
                first_struct = first_item["structure"]
            else:
                first_struct = first_item

            if hasattr(first_struct, "get_atomic_numbers"):
                dummy_graph = graph_builder.get_graph_from_atoms(first_struct)
            else:
                dummy_graph = graph_builder.get_graph(first_struct)

            if dummy_graph is not None:
                if getattr(dummy_graph, "site_feat", None) is not None:
                    site_property_dim = dummy_graph.site_feat.shape[-1]
                if getattr(dummy_graph, "state_feat", None) is not None:
                    state_property_dim = dummy_graph.state_feat.shape[-1]
        except Exception as e:
            print(f"Warning: Could not infer site/state property dimensions from dataset: {e}")

    # 2. Setup DataModule
    dm = OnTheFlyDataModule(
        train_adapter=train_adapter,
        val_adapter=val_adapter,
        test_adapter=test_adapter,
        builder=graph_builder,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        seed=kwargs.get("seed", 42),
        max_nodes_per_batch=max_nodes_per_batch,
        max_edges_per_batch=max_edges_per_batch,
        max_triplets_per_batch=max_triplets_per_batch,
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
    if not val_adapter:
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
        **kwargs,
    )

    trainer.fit(model, datamodule=dm)

    if test_adapter:
        trainer.test(model, datamodule=dm)

    return model
