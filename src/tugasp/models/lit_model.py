import pytorch_lightning as L
import torch
import torchmetrics

from .model import TugaGraphTransformer


class LitTugaSP(L.LightningModule):
    """
    Lightning module wrapper for TugaGraphTransformer.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_type: str = "l1",  # "l1", "mse", "huber", "bce", "cross_entropy"
        task_type: str = "regression",  # "regression" or "classification"
        num_atom_types: int = 100,
        d_model: int = 128,
        edge_rbf_start: float = 0.0,
        edge_rbf_stop: float = 5.0,
        num_edge_basis: int = 80,
        num_angle_basis: int = 40,
        d_out: int = 1,
        num_layers: int = 3,
        nhead: int = 4,
        dff_ratio: int = 4,
        activation: str = "silu",
        dropout: float = 0.0,
        # SOTA improvements
        use_swiglu: bool = False,
        use_lattice_encoding: bool = True,
        use_dihedrals: bool = False,
        # Site properties
        site_property_dim: int = 0,
        site_properties: list = None,
        # Structure-level state properties
        state_property_dim: int = 0,
        state_properties: list = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task_type = task_type.lower()
        self.model = TugaGraphTransformer(
            num_atom_types=num_atom_types,
            d_model=d_model,
            edge_rbf_start=edge_rbf_start,
            edge_rbf_stop=edge_rbf_stop,
            num_edge_basis=num_edge_basis,
            num_angle_basis=num_angle_basis,
            d_out=d_out,
            num_layers=num_layers,
            nhead=nhead,
            dff_ratio=dff_ratio,
            activation=activation,
            dropout=dropout,
            use_swiglu=use_swiglu,
            use_lattice_encoding=use_lattice_encoding,
            use_dihedrals=use_dihedrals,
            site_property_dim=site_property_dim,
            state_property_dim=state_property_dim,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type.lower()

        # Metrics
        if self.task_type == "regression":
            self.mae = torchmetrics.MeanAbsoluteError()
            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.rmse = torchmetrics.MeanSquaredError(squared=False)
            if self.loss_type == "mse":
                self.loss_fn = torch.nn.MSELoss()
            elif self.loss_type == "huber":
                self.loss_fn = torch.nn.HuberLoss(delta=1.0)
            else:
                self.loss_fn = torch.nn.L1Loss()
        else:  # Classification
            self.acc = torchmetrics.Accuracy(task="binary")
            self.auroc = torchmetrics.AUROC(task="binary")

            if self.loss_type == "bce":
                self.loss_fn = torch.nn.BCEWithLogitsLoss()
            elif self.loss_type == "cross_entropy":
                self.loss_fn = torch.nn.CrossEntropyLoss()
            else:
                self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        target = batch.y

        # Ensure dimensions match
        if output.shape != target.shape:
            output = output.view_as(target)

        loss = self.loss_fn(output, target)

        self.log(
            "train_loss",
            loss,
            batch_size=batch.num_graphs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # Log Train MAE
        if self.task_type == "regression":
            self.train_mae(output, target)
            self.log(
                "train_mae",
                self.train_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        target = batch.y

        if output.shape != target.shape:
            output = output.view_as(target)

        loss = self.loss_fn(output, target)
        self.log(
            "val_loss", loss, batch_size=batch.num_graphs, on_epoch=True, prog_bar=True
        )

        # Log Metrics
        if self.task_type == "regression":
            self.mae(output, target)
            self.rmse(output, target)
            self.log("val_mae", self.mae, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_rmse", self.rmse, on_step=False, on_epoch=True, prog_bar=True)
        else:
            # Classification Metrics
            probs = torch.sigmoid(output)

            self.acc(probs, target.long())
            self.auroc(probs, target.long())

            self.log("val_acc", self.acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val_auroc", self.auroc, on_step=False, on_epoch=True, prog_bar=True
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # OneCycleLR is used for stable convergence, as seen in similar architectures.
        if self.trainer is None:
            # Fallback for manual testing
            return optimizer

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.15,
            final_div_factor=1e3,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, batch):
        return self.model(batch)
