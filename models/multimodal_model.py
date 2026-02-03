import numpy as np
import pandas as pd
import joblib
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
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self.feature_names = None

    def _extract_all_features(self, df: pd.DataFrame, image_paths: list[str], texts: list[str], fit_tabular: bool = False) -> np.ndarray:
        # 1. Tabular features
        if fit_tabular:
            self.tabular_proc.fit(df)
        X_tab = self.tabular_proc.transform(df)
        tab_names = self.tabular_proc.get_feature_names()

        # 2. Image features
        img_results = self.image_proc.process_batch(image_paths)
        X_img_emb = np.array([r["embedding"] for r in img_results])
        X_img_sent = np.array([r["sentiment"] for r in img_results]).reshape(-1, 1)
        X_img_objs = np.array([r["num_objects"] for r in img_results]).reshape(-1, 1)
        X_img_bright = np.array([r["brightness"] for r in img_results]).reshape(-1, 1)
        X_img_contrast = np.array([r["contrast"] for r in img_results]).reshape(-1, 1)
        X_img_colorf = np.array([r["colorfulness"] for r in img_results]).reshape(-1, 1)
        X_img_avg_c = np.array([r["avg_color"] for r in img_results]) # 3 columns (R,G,B)

        # 3. Text features
        txt_results = self.text_proc.process_batch(texts)
        X_txt_emb = np.array([r["embedding"] for r in txt_results])
        X_txt_sent = np.array([r["sentiment"] for r in txt_results]).reshape(-1, 1)
        X_txt_len = np.array([r["length"] for r in txt_results]).reshape(-1, 1)
        X_txt_words = np.array([r["word_count"] for r in txt_results]).reshape(-1, 1)

        # Concatenate all features
        features_list = [
            X_tab,
            X_img_emb,
            X_img_sent,
            X_img_objs,
            X_img_bright,
            X_img_contrast,
            X_img_colorf,
            X_img_avg_c,
            X_txt_emb,
            X_txt_sent,
            X_txt_len,
            X_txt_words
        ]
        X = np.hstack(features_list)

        if self.feature_names is None:
            # Build feature names list once
            names = list(tab_names)
            names += [f"img_emb_{i}" for i in range(X_img_emb.shape[1])]
            names += ["img_sentiment", "img_num_objects", "img_brightness", "img_contrast", "img_colorfulness"]
            names += ["img_avg_r", "img_avg_g", "img_avg_b"]
            names += [f"txt_emb_{i}" for i in range(X_txt_emb.shape[1])]
            names += ["txt_sentiment", "txt_length", "txt_word_count"]
            self.feature_names = names

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
        X = self._extract_all_features(df, image_paths, texts, fit_tabular=False)
        y_pred = self.regressor.predict(X)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def get_feature_distribution(self, df: pd.DataFrame, image_paths: list[str], texts: list[str]):
        """Returns feature distributions for insights."""
        img_results = self.image_proc.process_batch(image_paths)
        txt_results = self.text_proc.process_batch(texts)

        return {
            "visual_sentiment": [r["sentiment"] for r in img_results],
            "textual_sentiment": [r["sentiment"] for r in txt_results],
            "object_counts": [r["num_objects"] for r in img_results],
            "brightness": [r["brightness"] for r in img_results],
            "colorfulness": [r["colorfulness"] for r in img_results]
        }

    def save(self, filepath: str):
        """Saves the model to a file."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads the model from a file."""
        return joblib.load(filepath)
