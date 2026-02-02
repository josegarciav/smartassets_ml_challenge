import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from features.image_processor import ImageProcessor
from features.text_processor import TextProcessor
from features.tabular_processor import TabularProcessor, calculate_ctr

class MultiModalCTRModel:
    """
    Multi-modal model for predicting Creative Effectiveness (CTR).
    Combines visual, textual, and tabular features.
    """
    def __init__(self, device: str = "cpu"):
        self.image_proc = ImageProcessor(device)
        self.text_proc = TextProcessor(device)
        self.tabular_proc = TabularProcessor()

        # Final regressor
        self.regressor = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def _extract_all_features(self, df: pd.DataFrame, image_paths: list[str], texts: list[str], fit_tabular: bool = False) -> np.ndarray:
        # 1. Tabular features
        if fit_tabular:
            self.tabular_proc.fit(df)
        X_tab = self.tabular_proc.transform(df)

        # 2. Image features
        img_results = self.image_proc.process_batch(image_paths)
        X_img_emb = np.array([r["embedding"] for r in img_results])
        X_img_sent = np.array([r["sentiment"] for r in img_results]).reshape(-1, 1)
        X_img_objs = np.array([r["num_objects"] for r in img_results]).reshape(-1, 1)
        X_img_color = np.array([r["avg_color"] for r in img_results])

        # 3. Text features
        txt_results = self.text_proc.process_batch(texts)
        X_txt_emb = np.array([r["embedding"] for r in txt_results])
        X_txt_sent = np.array([r["sentiment"] for r in txt_results]).reshape(-1, 1)

        # Concatenate all features
        X = np.hstack([
            X_tab,
            X_img_emb,
            X_img_sent,
            X_img_objs,
            X_img_color,
            X_txt_emb,
            X_txt_sent
        ])
        return X

    def fit(self, df: pd.DataFrame, image_paths: list[str], texts: list[str]):
        """Trains the model on multi-modal data."""
        X = self._extract_all_features(df, image_paths, texts, fit_tabular=True)
        y = calculate_ctr(df).values
        self.regressor.fit(X, y)
        print("Model training complete.")

    def predict(self, df: pd.DataFrame, image_paths: list[str], texts: list[str]) -> np.ndarray:
        """Predicts CTR for new data."""
        X = self._extract_all_features(df, image_paths, texts, fit_tabular=False)
        return self.regressor.predict(X)

    def evaluate(self, df: pd.DataFrame, image_paths: list[str], texts: list[str]):
        """Evaluates the model and returns metrics."""
        y_true = calculate_ctr(df).values
        y_pred = self.predict(df, image_paths, texts)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def get_feature_distribution(self, df: pd.DataFrame, image_paths: list[str], texts: list[str]):
        """Returns feature distributions (sentiment, object counts) for insights."""
        img_results = self.image_proc.process_batch(image_paths)
        txt_results = self.text_proc.process_batch(texts)

        return {
            "visual_sentiment": [r["sentiment"] for r in img_results],
            "textual_sentiment": [r["sentiment"] for r in txt_results],
            "object_counts": [r["num_objects"] for r in img_results]
        }
