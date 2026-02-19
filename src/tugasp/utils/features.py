import torch


def get_atom_features(species):
    """
    Get atomic number for an atom.

    Args:
        species (pymatgen.core.Species or Element): The atomic species.

    Returns:
        torch.Tensor: Tensor containing atomic number (1-based index).
    """
    if hasattr(species, "element"):
        element = species.element
    else:
        element = species

    # Return atomic number (Z) directly
    # Can be used with nn.Embedding(num_embeddings=100)
    return torch.tensor([element.Z], dtype=torch.long)


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Smooth cosine cutoff function that goes to 0 at the cutoff distance.

    This provides smooth gradients at the cutoff boundary, improving
    training stability compared to hard cutoffs.

    Args:
        distances: Tensor of distances.
        cutoff: Cutoff distance.

    Returns:
        Tensor of cutoff values in [0, 1].
    """
    return 0.5 * (torch.cos(distances * torch.pi / cutoff) + 1) * (distances < cutoff)


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, sigma2=None):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        if sigma2 is None:
            # Heuristic: match the spacing
            step = offset[1] - offset[0]
            sigma2 = (step) ** 2
        self.register_buffer("offset", offset)
        self.register_buffer(
            "limit", torch.as_tensor(sigma2)
        )  # Use as_tensor to avoid copy warning

    def forward(self, dist):
        # dist: (N, ) or (N, 1)
        # offset: (G, )
        if dist.dim() == 1:
            dist = dist.unsqueeze(-1)
        diff = dist - self.offset
        return torch.exp(-(diff**2) / (2 * self.limit))
