import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, TransformerConv


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Efficient alternative to LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    FFN(x) = (Swish(xW1) ⊙ xW3) W2
    """

    def __init__(self, d_model: int, dim_feedforward: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=bias)  # Gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Sequential):
    """
    Multi-Layer Perceptron.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        layers = []
        in_dim = d_in
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, d_hidden))
            layers.append(self._get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = d_hidden

        layers.append(nn.Linear(d_hidden, d_out))
        super().__init__(*layers)

    def _get_activation(self, activation: str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU()
        else:
            return nn.ReLU()


class GraphTransformerLayer(nn.Module):
    """
    Transformer layer for graph data.

    Args:
        d_model (int): Hidden dimension size.
        nhead (int): Number of attention heads.
        d_edge (int): Edge feature dimension.
        dim_feedforward (int): Dimension of FFN hidden layer.
        activation (str): Activation function (for non-SwiGLU FFN).
        dropout (float): Dropout probability.
        use_swiglu (bool): If True, use SwiGLU FFN instead of standard FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_edge: int,
        dim_feedforward: int = 512,
        activation: str = "relu",
        dropout: float = 0.0,
        use_swiglu: bool = False,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )

        self.norm1 = LayerNorm(d_model, mode="node")
        self.conv = TransformerConv(
            in_channels=d_model,
            out_channels=d_model // nhead,
            heads=nhead,
            concat=True,
            edge_dim=d_edge,
            dropout=dropout,
        )

        self.norm2 = LayerNorm(d_model, mode="node")
        self.use_swiglu = use_swiglu
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if use_swiglu:
            self.ffn = SwiGLU(d_model, dim_feedforward)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )

    def _get_activation(self, activation: str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        elif activation.lower() == "silu":
            return nn.SiLU()
        return nn.ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        # Pre-Norm Attention
        residual = x
        x_norm = self.norm1(x, batch)
        x_attn = self.conv(x_norm, edge_index, edge_attr)
        x = residual + x_attn

        # Pre-Norm FFN
        residual = x
        x_norm = self.norm2(x, batch)
        x_ffn = self.ffn(x_norm)
        x = residual + x_ffn

        return x


class InteractionBlock(nn.Module):
    """Interaction between Line Graph and Atom Graph."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int = 1,
        activation: str = "relu",
        use_swiglu: bool = False,
        use_dihedrals: bool = False,
    ):
        super().__init__()
        self.use_dihedrals = use_dihedrals

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                InteractionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    use_swiglu=use_swiglu,
                    use_dihedrals=use_dihedrals,
                )
            )

    def forward(
        self,
        node_feats,
        node_batch,
        edge_index,
        edge_feats,
        edge_batch,
        triplet_indices,
        angle_feats,
        state_feats=None,
        dihedral_index=None,
        dihedral_feats=None,
    ):
        x = node_feats
        e = edge_feats
        a = angle_feats
        d = dihedral_feats

        # Derive Angle Batch for Normalization in Dihedral Conv
        angle_batch = None
        if self.use_dihedrals and angle_feats.size(0) > 0:
            angle_batch = edge_batch[triplet_indices[0]]

        for layer in self.layers:
            x, e, a, d = layer(
                x,
                node_batch,
                edge_index,
                e,
                edge_batch,
                triplet_indices,
                a,
                state_feats,
                angle_batch,
                dihedral_index,
                d,
            )

        return x, e, a, d


class InteractionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: str,
        use_swiglu: bool = False,
        use_dihedrals: bool = False,
    ):
        super().__init__()
        self.use_dihedrals = use_dihedrals

        # Mix nodes -> edges
        self.node_edge_mixer = MLP(
            d_in=3 * d_model,
            d_hidden=d_model,
            d_out=d_model,
            num_layers=1,
            activation=activation,
        )
        # Norm
        self.edge_norm = RMSNorm(d_model)

        # Mix edges -> angles
        self.edge_angle_mixer = MLP(
            d_in=3 * d_model,
            d_hidden=d_model,
            d_out=d_model,
            num_layers=1,
            activation=activation,
        )
        self.angle_norm = RMSNorm(d_model)

        # Convolve Angles -> Edges
        self.line_graph_conv = GraphTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_edge=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            use_swiglu=use_swiglu,
        )

        # Convolve Edges -> Nodes
        self.atom_graph_conv = GraphTransformerLayer(
            d_model=d_model,
            nhead=nhead,
            d_edge=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            use_swiglu=use_swiglu,
        )

        # Dihedrals
        if use_dihedrals:
            # Mix angles -> dihedrals
            self.angle_dihedral_mixer = MLP(
                d_in=3 * d_model,
                d_hidden=d_model,
                d_out=d_model,
                num_layers=1,
                activation=activation,
            )
            self.dihedral_norm = RMSNorm(d_model)

            # Convolve Dihedrals -> Angles
            self.dihedral_graph_conv = GraphTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                d_edge=d_model,
                dim_feedforward=dim_feedforward,
                activation=activation,
                use_swiglu=use_swiglu,
            )

    def forward(
        self,
        x,
        node_batch,
        edge_index,
        e,
        edge_batch,
        triplet_indices,
        a,
        state_feats=None,
        angle_batch=None,
        dihedral_index=None,
        d=None,
    ):
        # Update Dihedrals
        if self.use_dihedrals and d is not None and dihedral_index is not None:
            if dihedral_index.size(1) > 0:
                # Update Dihedrals
                src_t, dst_t = dihedral_index
                d_input = torch.cat([a[src_t], a[dst_t], d], dim=-1)
                d_upd = self.angle_dihedral_mixer(d_input)
                d = d + d_upd
                d = self.dihedral_norm(d)

                # Update Angles from Dihedrals
                a = self.dihedral_graph_conv(a, dihedral_index, d, angle_batch)

        # Update Edges
        src, dst = edge_index
        if state_feats is not None:
            e = e + state_feats[edge_batch]
        edge_input = torch.cat([x[src], x[dst], e], dim=-1)
        e_upd = self.node_edge_mixer(edge_input)
        e = e + e_upd
        e = self.edge_norm(e)

        # Update Nodes
        if state_feats is not None:
            x = x + state_feats[node_batch]

        # Update Angles
        e_j, e_k = triplet_indices
        angle_input = torch.cat([e[e_j], e[e_k], a], dim=-1)
        a_upd = self.edge_angle_mixer(angle_input)
        a = a + a_upd
        a = self.angle_norm(a)

        # Convolve Angles -> Edges
        e = self.line_graph_conv(e, triplet_indices, a, edge_batch)

        # Convolve Edges -> Nodes
        x = self.atom_graph_conv(x, edge_index, e, node_batch)

        return x, e, a, d
