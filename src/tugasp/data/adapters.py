import os
import pickle
from typing import Union, List, Dict, Optional, Any
import numpy as np

try:
    import torch
except ImportError:
    torch = None


class BaseStoreAdapter:
    """Base interface for all data store adapters.

    Subclasses that support random access implement ``__len__`` and
    ``__getitem__`` (map-style). Streaming-only stores that can only be
    iterated should instead implement ``__iter__`` (and ``__len__`` if known)
    and set the class attribute ``is_iterable = True`` so the data module
    routes them through the iterable pipeline rather than calling the
    unimplemented ``__getitem__``.
    """

    #: Whether this adapter is stream-only (no random access via __getitem__).
    is_iterable = False

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Should return a dict containing:
           - 'structure': pymatgen.Structure or ase.Atoms
           - 'y': float/tensor target property (optional)
           - 'mat_id': str (optional)
        """
        raise NotImplementedError


class ListStoreAdapter(BaseStoreAdapter):
    """Adapter for in-memory lists of structures and optional targets."""
    def __init__(self, structures: List, targets: Optional[Union[List, np.ndarray, Any]] = None):
        self.structures = list(structures)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "structure": self.structures[idx]
        }
        if self.targets is not None:
            if torch is not None and isinstance(self.targets, torch.Tensor):
                item["y"] = self.targets[idx].item() if self.targets[idx].numel() == 1 else self.targets[idx]
            else:
                item["y"] = self.targets[idx]
        
        struct = self.structures[idx]
        if hasattr(struct, "properties") and isinstance(struct.properties, dict) and "mat_id" in struct.properties:
            item["mat_id"] = struct.properties["mat_id"]
        elif hasattr(struct, "info") and isinstance(struct.info, dict) and "mat_id" in struct.info:
            item["mat_id"] = struct.info["mat_id"]
        return item


class PickleAdapter(BaseStoreAdapter):
    """Adapter that loads raw structures or dicts from a pickle file."""
    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Pickle file must contain a list of structures or dicts, got {type(data)}")
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        if isinstance(item, dict):
            if "structure" not in item:
                for k in ["structure", "atoms", "atoms_obj"]:
                    if k in item:
                        item = {"structure": item[k], **{key: val for key, val in item.items() if key != k}}
                        break
                else:
                    raise KeyError(f"Pickle dict item at index {idx} does not contain a 'structure' or 'atoms' key.")
            return item
        elif isinstance(item, (tuple, list)):
            return {
                "structure": item[0],
                "y": item[1]
            }
        else:
            return {
                "structure": item
            }


class AseLmdbAdapter(BaseStoreAdapter):
    """Adapter for reading from ASE LMDB/DB trajectory stores.
    
    If fairchem is installed, uses fairchem's AseDBDataset to index directories
    of shards and read rows. Otherwise, falls back to standard ase.db.connect.
    """
    def __init__(self, path: str):
        self.path = path
        self._dataset = None
        self._length = 0
        self._is_fairchem = False
        
        if not os.path.exists(path):
            import glob
            if not glob.glob(path):
                raise FileNotFoundError(f"ASE DB/LMDB path not found: {path}")

        try:
            from fairchem.core.datasets import AseDBDataset
            self._dataset = AseDBDataset(config=dict(src=path))
            self._length = len(self._dataset)
            self._is_fairchem = True
        except ImportError:
            if os.path.isdir(path):
                raise ImportError(
                    "To read a directory of shards, 'fairchem-core' is required. "
                    "Please install it or provide a single file path for ase.db."
                )
            
            import ase.db
            connect_args = {}
            if path.endswith(".aselmdb") or path.endswith(".lmdb"):
                connect_args = {"readonly": True, "use_lock_file": False}
            
            self._db = ase.db.connect(path, **connect_args)
            self._length = len(self._db)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._is_fairchem:
            atoms = self._dataset.get_atoms(idx)
            y = None
            try:
                y = float(atoms.get_potential_energy())
            except Exception:
                pass
            
            mat_id = ""
            try:
                row = self._dataset[idx]
                if hasattr(row, "sid"):
                    mat_id = str(row.sid)
            except Exception:
                pass
            
            return {
                "structure": atoms,
                "y": y,
                "mat_id": mat_id
            }
        else:
            row = self._db.get(idx + 1)
            atoms = row.toatoms()
            
            y = None
            if hasattr(row, "data") and isinstance(row.data, dict):
                for key in ["y", "energy", "target"]:
                    if key in row.data:
                        y = row.data[key]
                        break
            if y is None:
                try:
                    y = float(atoms.get_potential_energy())
                except Exception:
                    pass
            
            mat_id = ""
            if hasattr(row, "data") and isinstance(row.data, dict):
                mat_id = row.data.get("mat_id", "")
            if not mat_id and hasattr(row, "id"):
                mat_id = f"ase-{row.id}"
                
            return {
                "structure": atoms,
                "y": y,
                "mat_id": mat_id
            }
