import os
import pandas as pd
import numpy as np
import joblib
from models.multimodal_model import MultiModalCTRModel
from utils.experiment_tracker import ExperimentTracker

class CreativeEffectivenessFacade:
    """
    Facade interface for the Creative Effectiveness Prediction system.
    Provides a simple API for training, evaluation, and prediction with experiment tracking.
    """
    def __init__(self, images_root: str, device: str = "cpu", mode: str = "multimodal"):
        self.images_root = images_root
        self.mode = mode
        self.device = device
        self.model = MultiModalCTRModel(device, mode)
        self.tracker = ExperimentTracker()

    def _resolve_paths(self, creative_ids: list, extensions: list) -> list[str]:
        paths = []
        for cid in creative_ids:
            found = False
            for ext in extensions:
                path = os.path.join(self.images_root, f"{cid}{ext}")
                if os.path.exists(path):
                    paths.append(path)
                    found = True
                    break
            if not found:
                # Fallback to first extension if not found
                paths.append(os.path.join(self.images_root, f"{cid}{extensions[0]}"))
        return paths

    def _get_image_paths(self, creative_ids: list) -> list[str]:
        return self._resolve_paths(creative_ids, [".jpg", ".jpeg", ".png", ".webp"])

    def _get_video_paths(self, creative_ids: list) -> list[str]:
        return self._resolve_paths(creative_ids, [".mp4", ".avi", ".mov"])

    def _get_texts(self, df: pd.DataFrame) -> list[str]:
        """Combines search tags and keywords for text processing."""
        return (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()

    def train_from_csv(self, csv_path: str, experiment_name: str = None):
        """Trains the model using a CSV file and logs the experiment."""
        df = pd.read_csv(csv_path)
        image_paths = self._get_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)
        video_paths = self._get_video_paths(df["creative_id"].tolist()) if self.mode == "enhanced" else None

        self.model.fit(df, image_paths, texts, video_paths)

        if experiment_name:
            metrics = self.model.evaluate(df, image_paths, texts, video_paths)
            self.tracker.log_experiment(experiment_name, {"mode": self.mode, "samples": len(df)}, metrics)

    def predict_dataframe(self, df: pd.DataFrame) -> dict:
        """Predicts CTR and returns insights for a dataframe of campaigns."""
        image_paths = self._get_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)
        video_paths = self._get_video_paths(df["creative_id"].tolist()) if self.mode == "enhanced" else None

        predictions = self.model.predict(df, image_paths, texts, video_paths)
        insights = self.model.get_feature_distribution(df, image_paths, texts, video_paths)

        return {
            "predictions": predictions.tolist(),
            "insights": insights
        }

    def predict_single(self, creative_id: int, metadata: dict) -> dict:
        """Predicts CTR for a single creative instance."""
        df = pd.DataFrame([metadata])
        df["creative_id"] = creative_id

        res = self.predict_dataframe(df)
        output = {
            "creative_id": creative_id,
            "predicted_ctr": res["predictions"][0]
        }

        # Add insights if available
        if res["insights"]:
            for k, v in res["insights"].items():
                output[k] = v[0]

        return output

    def evaluate_from_csv(self, csv_path: str) -> dict:
        """Evaluates the model on a test CSV."""
        df = pd.read_csv(csv_path)
        image_paths = self._get_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)
        video_paths = self._get_video_paths(df["creative_id"].tolist()) if self.mode == "enhanced" else None

        return self.model.evaluate(df, image_paths, texts, video_paths)

    def save_model(self, filepath: str):
        """Persists the trained model to disk."""
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath: str, images_root: str, device: str = "cpu"):
        """Loads a model and returns a new Facade instance."""
        model = MultiModalCTRModel.load(filepath)
        facade = cls(images_root, device, mode=model.mode)
        facade.model = model
        return facade
