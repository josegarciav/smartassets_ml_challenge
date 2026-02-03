import os
import pandas as pd
import numpy as np
import joblib
from models.multimodal_model import MultiModalCTRModel

class CreativeEffectivenessFacade:
    """
    Facade interface for the Creative Effectiveness Prediction system.
    Provides a simple API for training, evaluation, and prediction.
    """
    def __init__(self, images_root: str, device: str = "cpu"):
        self.images_root = images_root
        self.model = MultiModalCTRModel(device)

    def _resolve_image_paths(self, creative_ids: list) -> list[str]:
        """Resolves creative IDs to actual file paths."""
        paths = []
        for cid in creative_ids:
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                path = os.path.join(self.images_root, f"{cid}{ext}")
                if os.path.exists(path):
                    paths.append(path)
                    found = True
                    break
            if not found:
                # Fallback to .jpg even if it doesn't exist, model will handle dummy features
                paths.append(os.path.join(self.images_root, f"{cid}.jpg"))
        return paths

    def _get_texts(self, df: pd.DataFrame) -> list[str]:
        """Combines search tags and keywords for text processing."""
        return (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()

    def train_from_csv(self, csv_path: str):
        """Trains the model using a CSV file and images in images_root."""
        df = pd.read_csv(csv_path)
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)

        self.model.fit(df, image_paths, texts)
        print("Model trained successfully via Facade.")

    def predict_dataframe(self, df: pd.DataFrame) -> dict:
        """Predicts CTR and returns insights for a dataframe of campaigns."""
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)

        predictions = self.model.predict(df, image_paths, texts)
        insights = self.model.get_feature_distribution(df, image_paths, texts)

        return {
            "predictions": predictions.tolist(),
            "insights": insights
        }

    def predict_single(self, creative_id: int, metadata: dict) -> dict:
        """Predicts CTR for a single creative instance."""
        df = pd.DataFrame([metadata])
        df["creative_id"] = creative_id

        res = self.predict_dataframe(df)
        return {
            "creative_id": creative_id,
            "predicted_ctr": res["predictions"][0],
            "visual_sentiment": res["insights"]["visual_sentiment"][0],
            "textual_sentiment": res["insights"]["textual_sentiment"][0],
            "object_count": res["insights"]["object_counts"][0],
            "brightness": res["insights"]["brightness"][0],
            "colorfulness": res["insights"]["colorfulness"][0]
        }

    def evaluate_from_csv(self, csv_path: str) -> dict:
        """Evaluates the model on a test CSV."""
        df = pd.read_csv(csv_path)
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        texts = self._get_texts(df)

        return self.model.evaluate(df, image_paths, texts)

    def save_model(self, filepath: str):
        """Persists the trained model to disk."""
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath: str, images_root: str, device: str = "cpu"):
        """Loads a model and returns a new Facade instance."""
        facade = cls(images_root, device)
        facade.model = MultiModalCTRModel.load(filepath)
        return facade
