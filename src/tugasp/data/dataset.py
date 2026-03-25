from typing import Dict, List, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from pymatgen.core import Structure
from torch_geometric.data import Data

from ..utils.features import get_atom_features


class CrystalGraph(Data):
    """
    PyG Data object representing a crystal structure as a graph.

    This is a hierarchical graph with:
    - **Nodes**: Atoms in the crystal
    - **Edges**: Bonds between atoms (distance-based cutoff)
    - **Triplets**: Angle relationships between pairs of edges sharing a node
      (used for 3-body interactions in the line graph)

    Attributes:
        x (Tensor): Node features, shape (N, F) where N=num_atoms.
                    Currently contains atomic numbers as indices.
        edge_index (Tensor): Edge connectivity, shape (2, E) where E=num_edges.
                             Format: [src_nodes, dst_nodes].
        edge_attr (Tensor): Edge features, shape (E, 1). Contains bond distances.
        triplet_index (Tensor): Triplet (angle) connectivity, shape (2, T).
                                Each column [i, j] means edges i and j share a node.
        angle_attr (Tensor): Angle features, shape (T, 1). Contains cos(θ) values.
        lattice_params (Tensor): Unit cell parameters, shape (1, 6).
                                 Contains [a, b, c, α, β, γ].
        y (Tensor, optional): Target property, shape varies.
        mat_id (str, optional): Material ID for tracking.
    """

    def __inc__(self, key, value, *args, **kwargs):
        """
        Defines how indices should be incremented during batching.

        PyG batches multiple graphs by concatenating node/edge tensors.
        Index tensors must be offset to point to the correct nodes/edges
        in the concatenated tensor.
        """
        if key == "edge_index":
            return self.x.size(0)  # Offset by number of nodes
        if key == "triplet_index":
            return self.edge_attr.size(0)  # Offset by number of edges
        if key == "dihedral_index":
            return self.angle_attr.size(0)  # Offset by number of triplets/angles
        return super().__inc__(key, value, *args, **kwargs)


class TugaGraphBuilder:
    """
    Converts pymatgen Structure objects to CrystalGraph objects.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        max_neighbors: int = 12,
        max_neighbors_dihedral: int = 4,
        site_properties: Optional[Union[str, List[str]]] = None,
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.max_neighbors_dihedral = max_neighbors_dihedral
        # this is till experimental....
        self.compute_dihedrals = False
        # Site properties to extract as node features
        if isinstance(site_properties, str):
            site_properties = [site_properties]
        self.site_properties = site_properties

    def get_site_feat_dim(self, structure: Structure) -> int:
        """
        Compute the total dimension of site features for a given structure.
        Useful to determine site_property_dim before building all graphs.
        """
        if not self.site_properties:
            return 0
        total = 0
        for name in self.site_properties:
            vals = structure.site_properties.get(name)
            if vals is None:
                total += 1
            else:
                v = np.asarray(vals[0])
                total += 1 if v.ndim == 0 else v.shape[0]
        return total

    def get_graph(self, structure, properties=None, mat_id=None) -> "CrystalGraph":
        """
        Convert a pymatgen Structure to a CrystalGraph.

        Args:
            structure: pymatgen Structure object.
            properties: Optional dict of properties to attach (e.g., {"y": 1.5}).
            mat_id: Optional material ID string for tracking.

        Returns:
            CrystalGraph object ready for model input.
        """
        # 1. Node features (Atomic Numbers)
        self_feats = [get_atom_features(site) for site in structure.species]
        x = torch.tensor(np.vstack(self_feats), dtype=torch.long)

        # 1b. Site property features (optional)
        site_feat = None
        if self.site_properties:
            n_sites = len(structure)
            columns = []
            for name in self.site_properties:
                vals = structure.site_properties.get(name)
                if vals is None:
                    # Missing property: fill with zeros; infer dim from 1 (scalar fallback)
                    columns.append(np.zeros((n_sites, 1), dtype=np.float32))
                else:
                    arr = np.asarray(vals, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]  # (N,) -> (N, 1)
                    columns.append(arr)
            site_feat = torch.tensor(np.concatenate(columns, axis=1), dtype=torch.float32)

        # 2. Edges (bonds) with Cutoff + Max Neighbors strategy
        center_indices, neighbor_indices, images, distances = (
            structure.get_neighbor_list(r=self.cutoff)
        )

        # Calculate difference vectors manually
        cart_coords = structure.cart_coords
        lattice_matrix = structure.lattice.matrix

        # Vectorized calculations images are (N, 3), lattice is (3, 3)
        center_coords = cart_coords[center_indices]
        image_offsets = np.dot(images, lattice_matrix)
        neighbor_coords = cart_coords[neighbor_indices] + image_offsets
        diff_vecs = neighbor_coords - center_coords

        # Apply sorting
        sort_idx = np.lexsort((distances, center_indices))
        center_indices = center_indices[sort_idx]
        neighbor_indices = neighbor_indices[sort_idx]
        diff_vecs = diff_vecs[sort_idx]
        distances = distances[sort_idx]

        # Create ranks
        counts = np.bincount(center_indices, minlength=len(structure))
        group_starts = np.zeros(len(counts), dtype=np.int64)
        group_starts[1:] = np.cumsum(counts)[:-1]
        starts_mapped = group_starts[center_indices]
        ranks = np.arange(len(center_indices)) - starts_mapped

        # Filter by max_neighbors
        if self.max_neighbors:
            mask = ranks < self.max_neighbors
            center_indices = center_indices[mask]
            neighbor_indices = neighbor_indices[mask]
            diff_vecs = diff_vecs[mask]
            distances = distances[mask]
            ranks = ranks[mask]

        src_indices = center_indices
        dst_indices = neighbor_indices
        edge_vecs = torch.tensor(diff_vecs, dtype=torch.float32)
        edge_ranks = torch.tensor(ranks, dtype=torch.long)
        edge_dists = torch.tensor(distances, dtype=torch.float32)

        edge_index = torch.stack(
            [
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(dst_indices, dtype=torch.long),
            ]
        )

        # Edge features: bond distances, shape (E, 1)
        edge_attr = edge_dists.unsqueeze(-1)

        # 3. Triplets (for 3-body / angle interactions)

        # first occurrence of each source node and count of repetitions
        _, start_indices, counts = np.unique(
            src_indices, return_index=True, return_counts=True
        )

        triplets_left = []
        triplets_right = []

        # Filter for atoms with at least 2 edges
        mask_triplets = counts >= 2
        valid_starts = start_indices[mask_triplets]
        valid_counts = counts[mask_triplets]

        for start, count in zip(valid_starts, valid_counts):
            # Edges for this atom are in range [start, start + count)
            edges = np.arange(start, start + count)

            # Broadcast to pairs (count, count)
            e_left = np.repeat(edges, count)
            e_right = np.tile(edges, count)

            # Filter diagonal (self-loop in angle)
            mask = e_left != e_right

            triplets_left.append(e_left[mask])
            triplets_right.append(e_right[mask])

        if not triplets_left:
            triplet_index = torch.empty((2, 0), dtype=torch.long)
            angle_attr = torch.empty((0, 1), dtype=torch.float32)
            dihedral_index = torch.empty((2, 0), dtype=torch.long)
            dihedral_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            # Concatenate all at once
            t_left = np.concatenate(triplets_left)
            t_right = np.concatenate(triplets_right)
            triplet_index = torch.tensor(np.stack([t_left, t_right]), dtype=torch.long)
            vec_j = edge_vecs[triplet_index[0]]
            vec_k = edge_vecs[triplet_index[1]]
            cos_theta = torch.cosine_similarity(vec_j, vec_k)
            angle_attr = cos_theta.unsqueeze(-1)

            # 3.5 Dihedrals (4-body) - Interactions between triplets sharing an edge
            # Triplet t1: (i, j, k) -> edges (j->i), (j->k).
            # Triplet t2: (j, k, l) -> edges (k->j), (k->l).
            # They share the j-k bond.

            if not self.compute_dihedrals:
                dihedral_index = torch.empty((2, 0), dtype=torch.long)
                dihedral_attr = torch.empty((0, 1), dtype=torch.float32)
            else:
                # THIS PORTION IS 4 BODY INTERACTIONS - OPTIONAL AND NOT WELL TESTED - DISABLED BY DEFAULT
                # Fully Vectorized Dihedral Construction
                # Determine connected triplets (e1, e2) and (e3, e4) where rev(e2) == e3

                triplet_idx_np = triplet_index.numpy()
                t_left = triplet_idx_np[0]  # e1 (src -> center)
                t_right = triplet_idx_np[1]  # e2 (center -> dst)
                num_triplets = triplet_index.shape[1]
                edge_ranks_np = edge_ranks.numpy()

                # 1. Build Reverse Map (O(E))
                edge_src = edge_index[0].numpy()
                edge_dst = edge_index[1].numpy()

                # Mapping (s, d) -> idx
                edge_map = {
                    (s, d): i
                    for i, (s, d) in enumerate(
                        zip(edge_src, edge_dst)
                    )  # Use numpy arrays directly
                }

                reverse_map = np.full(len(edge_vecs), -1, dtype=np.int64)
                for i in range(len(edge_vecs)):
                    s, d = edge_src[i], edge_dst[i]

                    if (d, s) in edge_map:
                        reverse_map[i] = edge_map[(d, s)]

                # 2. Prepare Join
                # Match T1 target edge with T2 source edge

                t1_targets = reverse_map[t_right]

                # Filter T1 based on validity and rank
                valid_t1 = t1_targets != -1
                if self.max_neighbors_dihedral is not None:
                    limit = self.max_neighbors_dihedral
                    # Check ranks for e1 and e2
                    valid_t1 &= (edge_ranks_np[t_left] < limit) & (
                        edge_ranks_np[t_right] < limit
                    )

                t1_indices = np.where(valid_t1)[0]
                t1_targets = t1_targets[valid_t1]

                # T2 candidates (Sources)
                # T2 starts with t_left.
                t2_sources = t_left
                valid_t2 = np.ones(num_triplets, dtype=bool)
                if self.max_neighbors_dihedral is not None:
                    limit = self.max_neighbors_dihedral
                    # Check ranks for e4 (t_right)
                    valid_t2 &= edge_ranks_np[t_right] < limit

                t2_indices = np.where(valid_t2)[0]
                t2_sources = t2_sources[valid_t2]

                # 3. Vectorized Join (Sort-Merge / Hash Join logic)
                # Sort T2 by source edge for binary search
                sort_idx = np.argsort(t2_sources)
                sorted_t2_sources = t2_sources[sort_idx]
                sorted_t2_indices = t2_indices[sort_idx]

                # Search for t1_targets in sorted_t2_sources
                l_bounds = np.searchsorted(sorted_t2_sources, t1_targets, side="left")
                r_bounds = np.searchsorted(sorted_t2_sources, t1_targets, side="right")

                counts = r_bounds - l_bounds

                # 4. Construct Result
                total_matches = counts.sum()

                if total_matches == 0:
                    dihedral_index = torch.empty((2, 0), dtype=torch.long)
                    dihedral_attr = torch.empty((0, 1), dtype=torch.float32)
                else:
                    # Repeat T1 indices matches
                    dh_t1 = np.repeat(t1_indices, counts)

                    # T2 indices: bulk copy from sorted array
                    # Construct T2 indices array
                    dh_t2 = np.empty(total_matches, dtype=np.int64)

                    # Loop over groups with matches
                    current_idx = 0
                    # Filter only relevant iterations
                    nz_indices = np.where(counts > 0)[0]

                    for i in nz_indices:
                        c = counts[i]
                        left_bound = l_bounds[i]
                        right_bound = r_bounds[i]
                        # Copy slice
                        dh_t2[current_idx : current_idx + c] = sorted_t2_indices[
                            left_bound:right_bound
                        ]
                        current_idx += c

                    # 5. Post-Filtering
                    # Filter self-loops: src(e1) != dst(e4)

                    n_i = edge_src[t_left[dh_t1]]
                    n_l = edge_dst[t_right[dh_t2]]

                    mask_loop = n_i != n_l

                    # Apply mask
                    dh_t1 = dh_t1[mask_loop]
                    dh_t2 = dh_t2[mask_loop]

                    if len(dh_t1) == 0:
                        dihedral_index = torch.empty((2, 0), dtype=torch.long)
                        dihedral_attr = torch.empty((0, 1), dtype=torch.float32)
                    else:
                        dihedral_index = torch.tensor(
                            np.stack([dh_t1, dh_t2]), dtype=torch.long
                        )

                    # Compute Torsion Angles for nodes i, j, k, l

                    t1_indices = dihedral_index[0]
                    t2_indices = dihedral_index[1]

                    e1_list = triplet_index[0, t1_indices]
                    e2_list = triplet_index[1, t1_indices]
                    e4_list = triplet_index[1, t2_indices]
                    # Rev edge for middle? e3 is reverse of e2.
                    e3_list = triplet_index[0, t2_indices]

                    v_ji = edge_vecs[e1_list]
                    v_jk = edge_vecs[e2_list]
                    v_kj = edge_vecs[e3_list]
                    v_kl = edge_vecs[e4_list]

                    # Cross products
                    # n1 = j->i x j->k
                    n1 = torch.cross(v_ji, v_jk, dim=1)
                    # n2 = k->j x k->l
                    n2 = torch.cross(v_kj, v_kl, dim=1)

                    # Normalize
                    n1 = torch.nn.functional.normalize(n1, dim=1)
                    n2 = torch.nn.functional.normalize(n2, dim=1)

                    # Cosine
                    # angle = atan2( (n1 x n2) . u_bond, n1 . n2 )
                    # u_bond = normalized unit vector along bond j-k.
                    u_jk = torch.nn.functional.normalize(v_jk, dim=1)

                    m1 = torch.sum(n1 * n2, dim=1)
                    m2 = torch.sum(torch.cross(n1, n2, dim=1) * u_jk, dim=1)

                    angle = torch.atan2(m2, m1)  # Radians in [-pi, pi]
                    dihedral_attr = angle.unsqueeze(-1)

        # 4. Lattice parameters: (a, b, c, α, β, γ)
        # Shape (1, 6) so PyG batching stacks to (B, 6)
        lattice = structure.lattice
        lattice_params = torch.tensor(
            [
                [
                    lattice.a,
                    lattice.b,
                    lattice.c,
                    lattice.alpha,
                    lattice.beta,
                    lattice.gamma,
                ]
            ],
            dtype=torch.float32,
        )

        # 5. Build the graph object
        graph = CrystalGraph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            triplet_index=triplet_index,
            angle_attr=angle_attr,
            dihedral_index=dihedral_index,
            dihedral_attr=dihedral_attr,
            lattice_params=lattice_params,
            site_feat=site_feat,
        )

        if mat_id:
            graph.mat_id = mat_id

        if properties:
            for k, v in properties.items():
                if k == "y":
                    # Handle Vector vs Scalar targets
                    val = torch.tensor(v, dtype=torch.float32)
                    if val.ndim == 0:
                        val = val.unsqueeze(0)  # Scalar -> (1,)
                    graph.y = val
                else:
                    setattr(graph, k, v)

        return graph

    def _process_batch(self, batch_data):
        """Helper to process a batch of structures in one go."""
        return [self.get_graph(s, p, m) for s, p, m in batch_data]

    def get_graphs(
        self,
        structures: List[Structure],
        properties: Optional[List[Optional[Dict]]] = None,
        mat_ids: Optional[List[Optional[str]]] = None,
        n_jobs: int = 1,
        show_progress: bool = True,
    ) -> List["CrystalGraph"]:
        """
        Convert a list of pymatgen Structures to CrystalGraph objects in parallel.

        Args:
            structures: List of pymatgen Structure objects.
            properties: Optional list of property dicts corresponding to each structure.
                        If None, all structures will have no properties attached.
            mat_ids: Optional list of material IDs corresponding to each structure.
            n_jobs: Number of parallel jobs. Follows joblib conventions:
                    - 1 (default): Sequential processing (no parallelism overhead).
                    - -1: Use all available CPUs.
                    - Positive integer: Use exactly n_jobs workers.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of CrystalGraph objects.
        """
        n = len(structures)

        # Handle None inputs by creating lists of None
        if properties is None:
            properties = [None] * n
        if mat_ids is None:
            mat_ids = [None] * n

        if len(properties) != n:
            raise ValueError(
                f"Length mismatch: {n} structures vs {len(properties)} properties"
            )
        if len(mat_ids) != n:
            raise ValueError(
                f"Length mismatch: {n} structures vs {len(mat_ids)} mat_ids"
            )

        if n_jobs == 1:
            # Sequential processing with optional progress bar
            iterable = zip(structures, properties, mat_ids)
            if show_progress:
                from tqdm import tqdm

                iterable = tqdm(
                    list(iterable),
                    desc="Building graphs (Serial)",
                    unit="struct",
                )
            return [self.get_graph(s, props, mid) for s, props, mid in iterable]

        # Parallel processing with Manual Batching

        # Determine effective n_jobs
        if n_jobs < 0:
            import multiprocessing

            n_jobs = multiprocessing.cpu_count()

        # Heuristic for chunk size based on benchmarks
        base_chunk = n // (n_jobs * 4)
        chunk_size = min(50, max(10, base_chunk))

        # Generator for chunks
        batches = []
        for i in range(0, n, chunk_size):
            batch = list(
                zip(
                    structures[i : i + chunk_size],
                    properties[i : i + chunk_size],
                    mat_ids[i : i + chunk_size],
                )
            )
            batches.append(batch)

        jobs = (delayed(self._process_batch)(batch) for batch in batches)

        if show_progress:
            from tqdm import tqdm

            # Use generator to preserve order and assemble efficiently
            parallel = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")

            graphs = []
            with tqdm(
                total=n, desc=f"Building graphs ({n_jobs} workers)", unit="struct"
            ) as pbar:
                for batch_result in parallel(jobs):
                    graphs.extend(batch_result)
                    pbar.update(len(batch_result))
        else:
            batch_results = Parallel(n_jobs=n_jobs, backend="loky")(jobs)
            graphs = [g for batch in batch_results for g in batch]

        return graphs
