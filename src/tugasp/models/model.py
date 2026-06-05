import os

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax as pyg_softmax

from ..utils.features import GaussianSmearing, cosine_cutoff
from .layers import MLP, InteractionBlock


class LatticeEncoder(nn.Module):
    """
    Encodes lattice parameters (a, b, c, α, β, γ) into a d_model vector.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 6 lattice params: a, b, c, alpha, beta, gamma
        self.proj = nn.Sequential(
            nn.Linear(6, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lattice_params: (B, 6) tensor of [a, b, c, α, β, γ]

        Returns:
            (B, d_model) lattice representation
        """
        return self.proj(lattice_params)


class HybridEmbedding(nn.Module):
    """
    Embedding module using Pettifor embeddings or custom features.
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        d_model: int = 128,
        activation: str = "silu",
        properties_path: str = None,
        custom_embedding_dict: dict = None,
    ):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.d_model = d_model

        # 1. Custom / User-provided Embedding (Optional)
        self.use_custom = False
        if custom_embedding_dict is not None:
            self.use_custom = True
            # Infer dimension
            first_key = next(iter(custom_embedding_dict))
            custom_dim = len(custom_embedding_dict[first_key])

            # Create buffer
            custom_matrix = torch.zeros((num_atom_types + 1, custom_dim))
            for z, feats in custom_embedding_dict.items():
                if z <= num_atom_types:
                    # Handle if z is string (symbol) or int
                    # Assuming input ensures int keys or we'd need conversion.
                    # Usually custom dicts are {1: [...], 2: [...]}
                    custom_matrix[z] = torch.tensor(feats, dtype=torch.float32)

            self.register_buffer("custom_features", custom_matrix)

            # Projection for custom features
            self.proj = MLP(
                d_in=custom_dim,
                d_hidden=d_model,
                d_out=d_model,
                num_layers=2,
                activation=activation,
            )
        else:
            # Pettifor Embedding
            if properties_path is None:
                # Default to relative path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                properties_path = os.path.join(
                    current_dir, "..", "data", "petiffor_embedding.csv"
                )

            if not os.path.exists(properties_path):
                raise FileNotFoundError(
                    f"Pettifor embedding CSV not found at {properties_path}. "
                )

            import csv

            # Load CSV

            emb_values = []
            with open(properties_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    # Row[0] is symbol, rest are values
                    try:
                        vals = [float(x) for x in row[1:]]
                        emb_values.append(vals)
                    except ValueError:
                        # Handle potential empty strings or errors
                        vals = [0.0 if x == "" else float(x) for x in row[1:]]
                        emb_values.append(vals)

            # Feature dimension
            feature_dim = len(emb_values[0]) if emb_values else 0

            # Map Z=i+1 to values

            phys_matrix = torch.zeros((num_atom_types + 1, feature_dim))

            # Copy values
            limit = min(len(emb_values), num_atom_types)
            for i in range(limit):
                z = i + 1
                phys_matrix[z] = torch.tensor(emb_values[i], dtype=torch.float32)

            self.register_buffer("phys_features", phys_matrix)

            # Projection for physical features
            self.proj = MLP(
                d_in=feature_dim,
                d_hidden=d_model,
                d_out=d_model,
                num_layers=2,
                activation=activation,
            )

        # 3. Layer Norm
        self.norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # MLP is already initialized, but we can re-init if needed.
        # Generally model.apply() calls this, so it's good to have.
        pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N,) tensor of atomic numbers/indices

        Returns:
            (N, d_model) embedded features
        """
        if self.use_custom:
            features = self.custom_features[z]
        else:
            features = self.phys_features[z]

        out = self.proj(features)
        return self.norm(out)


class TugaGraphTransformer(nn.Module):
    """
    Graph Transformer for property prediction.
    """

    def __init__(
        self,
        # Node inputs
        num_atom_types: int = 100,
        d_model: int = 128,
        # Edge inputs (RBF)
        edge_rbf_start: float = 0.0,
        edge_rbf_stop: float = 5.0,
        num_edge_basis: int = 80,
        # Angle inputs (RBF)
        num_angle_basis: int = 40,
        # Model architecture
        d_out: int = 1,
        num_layers: int = 3,
        nhead: int = 4,
        dff_ratio: int = 4,
        activation: str = "silu",
        dropout: float = 0.0,
        # New SOTA features
        use_swiglu: bool = False,
        use_lattice_encoding: bool = True,
        use_dihedrals: bool = False,
        num_dihedral_basis: int = 20,
        custom_embedding_dict: dict = None,
        # Site properties
        site_property_dim: int = 0,
        # Structure-level state properties
        state_property_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_lattice_encoding = use_lattice_encoding
        self.use_dihedrals = use_dihedrals
        self.edge_cutoff = edge_rbf_stop
        self.site_property_dim = site_property_dim
        self.state_property_dim = state_property_dim

        # Calculate feedforward dimension
        dim_feedforward = int(d_model * dff_ratio)

        # Embeddings
        self.atom_embedding = HybridEmbedding(
            num_atom_types=num_atom_types,
            d_model=d_model,
            activation=activation,
            custom_embedding_dict=custom_embedding_dict,
        )

        # Project RBF to d_model
        self.edge_rbf = GaussianSmearing(
            start=edge_rbf_start, stop=edge_rbf_stop, num_gaussians=num_edge_basis
        )
        self.edge_embedding = nn.Linear(num_edge_basis, d_model)

        # Angle RBF to d_model
        self.angle_rbf = GaussianSmearing(
            start=-1.0, stop=1.0, num_gaussians=num_angle_basis
        )
        self.angle_embedding = nn.Linear(num_angle_basis, d_model)

        # Dihedral RBF to d_model (-pi to pi)
        if use_dihedrals:
            self.dihedral_rbf = GaussianSmearing(
                start=-3.14159, stop=3.14159, num_gaussians=num_dihedral_basis
            )
            self.dihedral_embedding = nn.Linear(num_dihedral_basis, d_model)

        # --- Lattice Encoder (optional) ---
        if use_lattice_encoding:
            self.lattice_encoder = LatticeEncoder(d_model)

        # --- Site Property Projector (optional) ---
        if site_property_dim > 0:
            self.site_proj = MLP(
                d_in=site_property_dim,
                d_hidden=d_model,
                d_out=d_model,
                num_layers=2,
                activation=activation,
            )

        # --- Global State Property Projector (optional) ---
        if state_property_dim > 0:
            self.state_proj = MLP(
                d_in=state_property_dim,
                d_hidden=d_model,
                d_out=d_model,
                num_layers=2,
                activation=activation,
            )

        # --- Encoder ---
        self.encoder = InteractionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            activation=activation,
            use_swiglu=use_swiglu,
            use_dihedrals=use_dihedrals,
        )

        # Pooling
        pool_hidden = max(d_model // 2, 16)

        # Pooling Attention
        self.pool_gate = MLP(
            d_in=d_model,
            d_hidden=pool_hidden,
            d_out=1,
            num_layers=1,
            activation=activation,
        )

        # Edge Pooling
        self.edge_pool_gate = MLP(
            d_in=d_model,
            d_hidden=pool_hidden,
            d_out=1,
            num_layers=1,
            activation=activation,
        )

        # Angle Pooling
        self.angle_pool_gate = MLP(
            d_in=d_model,
            d_hidden=pool_hidden,
            d_out=1,
            num_layers=1,
            activation=activation,
        )

        # Pooling Attention for Dihedrals (New Multi-Level Pooling)
        if use_dihedrals:
            self.dihedral_pool_gate = MLP(
                d_in=d_model,
                d_hidden=pool_hidden,
                d_out=1,
                num_layers=1,
                activation=activation,
            )

        # Final Projection
        readout_dim = 3 * d_model
        if use_dihedrals:
            readout_dim += d_model
        if use_lattice_encoding:
            readout_dim += d_model
        if state_property_dim > 0:
            readout_dim += d_model

        self.output_head = MLP(
            d_in=readout_dim,
            d_hidden=dim_feedforward,
            d_out=d_out,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        triplet_indices, angle_attr = batch.triplet_index, batch.angle_attr

        dihedral_index, dihedral_attr = None, None
        _use_dihedrals = self.use_dihedrals
        if _use_dihedrals:
            if hasattr(batch, "dihedral_index") and batch.dihedral_index is not None:
                dihedral_index = batch.dihedral_index
                dihedral_attr = batch.dihedral_attr
            else:
                _use_dihedrals = False

        # 1. Embeddings
        if x.dim() > 1 and x.size(1) == 1:
            x = x.squeeze(1)
        x = self.atom_embedding(x)

        # Inject site property features (additive, at first layer)
        if self.site_property_dim > 0 and hasattr(batch, "site_feat") and batch.site_feat is not None:
            x = x + self.site_proj(batch.site_feat.float())

        state_feats = None
        if self.state_property_dim > 0:
            if not hasattr(batch, "state_feat") or batch.state_feat is None:
                raise ValueError("Batch is missing required state_feat")
            state_feats = self.state_proj(batch.state_feat.float())

        # Edge RBF + Cutoff
        raw_dist = edge_attr
        edge_attr = self.edge_rbf(edge_attr)
        cutoff_envelope = cosine_cutoff(raw_dist, self.edge_cutoff)
        if cutoff_envelope.dim() == 1:
            cutoff_envelope = cutoff_envelope.unsqueeze(-1)
        edge_attr = edge_attr * cutoff_envelope
        edge_attr = self.edge_embedding(edge_attr)

        angle_attr = self.angle_rbf(angle_attr)
        angle_attr = self.angle_embedding(angle_attr)

        if _use_dihedrals and dihedral_attr is not None and dihedral_attr.numel() > 0:
            dihedral_attr = self.dihedral_rbf(dihedral_attr)
            dihedral_attr = self.dihedral_embedding(dihedral_attr)
        else:
            dihedral_attr = None  # Explicit None if empty or unused

        # 2. Encoder
        if hasattr(batch, "edge_index_batch") and batch.edge_index_batch is not None:
            edge_batch = batch.edge_index_batch
        else:
            edge_batch = batch_idx[edge_index[0]]

        x, edge_attr_out, angle_attr_out, dihedral_attr_out = self.encoder(
            node_feats=x,
            node_batch=batch_idx,
            edge_index=edge_index,
            edge_feats=edge_attr,
            edge_batch=edge_batch,
            triplet_indices=triplet_indices,
            angle_feats=angle_attr,
            state_feats=state_feats,
            dihedral_index=dihedral_index,
            dihedral_feats=dihedral_attr,
        )

        # 3. Readout
        # Node Pooling
        scores = self.pool_gate(x)
        att_weights = pyg_softmax(scores, batch_idx)
        node_repr = global_add_pool(x * att_weights, batch_idx)

        # Edge Pooling
        edge_scores = self.edge_pool_gate(edge_attr_out)
        edge_att = pyg_softmax(edge_scores, edge_batch)
        edge_repr = global_add_pool(
            edge_attr_out * edge_att, edge_batch, size=node_repr.size(0)
        )

        # Angle Pooling
        if angle_attr_out.size(0) > 0:
            angle_batch = edge_batch[triplet_indices[0]]

            angle_scores = self.angle_pool_gate(angle_attr_out)
            angle_att = pyg_softmax(angle_scores, angle_batch)
            angle_repr = global_add_pool(
                angle_attr_out * angle_att, angle_batch, size=node_repr.size(0)
            )
        else:
            angle_repr = torch.zeros_like(node_repr)

        # Dihedral Pooling
        if _use_dihedrals:
            if dihedral_attr_out is not None and dihedral_attr_out.size(0) > 0:
                # Recalculate angle batch for dihedral pooling
                angle_batch = edge_batch[triplet_indices[0]]
                dihedral_batch = angle_batch[dihedral_index[0]]

                dihedral_scores = self.dihedral_pool_gate(dihedral_attr_out)
                dihedral_att = pyg_softmax(dihedral_scores, dihedral_batch)
                dihedral_repr = global_add_pool(
                    dihedral_attr_out * dihedral_att,
                    dihedral_batch,
                    size=node_repr.size(0),
                )
            else:
                dihedral_repr = torch.zeros_like(node_repr)

            graph_repr = torch.cat(
                [node_repr, edge_repr, angle_repr, dihedral_repr], dim=-1
            )
        else:
            graph_repr = torch.cat([node_repr, edge_repr, angle_repr], dim=-1)

        # 4. Lattice Encoding (if enabled and available)
        if self.use_lattice_encoding and hasattr(batch, "lattice_params"):
            lattice_repr = self.lattice_encoder(batch.lattice_params)
            graph_repr = torch.cat([graph_repr, lattice_repr], dim=-1)
        elif self.use_lattice_encoding:
            # Fallback: pad with zeros if lattice not provided
            zeros = torch.zeros(
                graph_repr.size(0), self.d_model, device=graph_repr.device
            )
            graph_repr = torch.cat([graph_repr, zeros], dim=-1)

        # 5. Global State Readout
        if state_feats is not None:
            graph_repr = torch.cat([graph_repr, state_feats], dim=-1)

        # 6. Projection
        out = self.output_head(graph_repr)

        return out
