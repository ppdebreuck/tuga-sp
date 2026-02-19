import torch

from .data.dataset import TugaGraphBuilder
from .data.loader import create_loader
from .models.lit_model import LitTugaSP


class Predictor:
    def __init__(
        self,
        model_path=None,
        model=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the predictor.

        Args:
            model_path (str): Path to the model checkpoint (.ckpt or .pt).
            model (LitTugaSP): Alternatively, pass a loaded model.
            device (str): Device to run inference on.
        """
        self.device = device

        if model:
            self.model = model
        elif model_path:
            # We assume it's a Lightning checkpoint for now.
            # If it's a raw state dict, we might need a different loading strategy or config.
            # Assuming standard lightning checkpoint.
            try:
                self.model = LitTugaSP.load_from_checkpoint(model_path)
            except Exception as e:
                # Fallback if it's just a state dict or different format
                print(
                    f"Failed to load as lightning checkpoint: {e}. Trying raw state dict."
                )
                # We need to know args to init model then load state dict.
                # For now assume checkpoint contains hparams.
                raise e
        else:
            raise ValueError("Must provide either model_path or model.")

        self.model.to(self.device)
        self.model.eval()
        self.graph_builder = TugaGraphBuilder()

    def predict(self, structure, n_jobs: int = 1):
        """
        Predict property for a single pymatgen Structure or list of structures.

        Args:
            structure (pymatgen.core.Structure or list): Input structure(s).
            n_jobs: Number of parallel jobs for graph building. Follows joblib conventions:
                    - 1 (default): Sequential processing.
                    - -1: Use all available CPUs.
                    - Positive integer: Use exactly n_jobs workers.

        Returns:
            torch.Tensor: Predictions.
        """
        is_list = isinstance(structure, list)
        if not is_list:
            structure = [structure]

        # Use parallel graph building for multiple structures
        graphs = self.graph_builder.get_graphs(structure, n_jobs=n_jobs)

        # Batch size can be len(graphs) for small inference
        loader = create_loader(graphs, batch_size=len(graphs), shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred = self.model(batch)
                preds.append(pred)

        result = torch.cat(preds, dim=0)

        if not is_list:
            return result[0]
        return result
