import os
import pandas as pd
import numpy as np
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
            # Try common extensions
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                path = os.path.join(self.images_root, f"{cid}{ext}")
                if os.path.exists(path):
                    paths.append(path)
                    found = True
                    break
            if not found:
                # Use a placeholder or handle missing
                print(f"Warning: Image for creative_id {cid} not found in {self.images_root}")
                # We'll still add a path to avoid alignment issues,
                # the processor will handle the missing file.
                paths.append(os.path.join(self.images_root, f"{cid}.jpg"))
        return paths

    def train_from_csv(self, csv_path: str):
        """Trains the model using a CSV file and images in images_root."""
        df = pd.read_csv(csv_path)
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        # Use search_tags and keywords for text
        texts = (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()

        self.model.fit(df, image_paths, texts)
        print("Model trained successfully via Facade.")

    def predict_dataframe(self, df: pd.DataFrame) -> dict:
        """Predicts CTR and returns insights for a dataframe of campaigns."""
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        texts = (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()

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
            "object_count": res["insights"]["object_counts"][0]
        }

    def evaluate_from_csv(self, csv_path: str) -> dict:
        """Evaluates the model on a test CSV."""
        df = pd.read_csv(csv_path)
        image_paths = self._resolve_image_paths(df["creative_id"].tolist())
        texts = (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()

        return self.model.evaluate(df, image_paths, texts)
