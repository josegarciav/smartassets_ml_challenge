import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from features.image_processor import ImageProcessor
from features.text_processor import TextProcessor
from features.tabular_processor import TabularProcessor, calculate_ctr
from features.video_processor import VideoProcessor

class MultiModalCTRModel:
    """
    Enhanced Multi-modal model for predicting Creative Effectiveness (CTR).
    Supports multiple modes: 'baseline' (tabular), 'multimodal' (+image/text), 'enhanced' (+video).
    """
    def __init__(self, device: str = "cpu", mode: str = "multimodal"):
        self.device = device
        self.mode = mode
        self.tabular_proc = TabularProcessor()

        if mode != "baseline":
            self.image_proc = ImageProcessor(device)
            self.text_proc = TextProcessor(device)
            if mode == "enhanced":
                self.video_proc = VideoProcessor(device)

        self.regressor = HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        self.feature_names = None

    def _extract_all_features(self, df: pd.DataFrame, image_paths: list[str] = None, texts: list[str] = None, video_paths: list[str] = None, fit_tabular: bool = False) -> np.ndarray:
        # 1. Tabular features (Always used)
        if fit_tabular:
            self.tabular_proc.fit(df)
        X_tab = self.tabular_proc.transform(df)
        tab_names = self.tabular_proc.get_feature_names()

        if self.mode == "baseline":
            if self.feature_names is None:
                self.feature_names = list(tab_names)
            return X_tab

        # 2. Image features
        img_results = self.image_proc.process_batch(image_paths)
        X_img_emb = np.array([r["embedding"] for r in img_results])
        X_img_other = np.array([[r["sentiment"], r["num_objects"], r["brightness"], r["contrast"], r["colorfulness"]] for r in img_results])
        X_img_avg_c = np.array([r["avg_color"] for r in img_results])

        # 3. Text features
        txt_results = self.text_proc.process_batch(texts)
        X_txt_emb = np.array([r["embedding"] for r in txt_results])
        X_txt_other = np.array([[r["sentiment"], r["length"], r["word_count"]] for r in txt_results])

        features_list = [X_tab, X_img_emb, X_img_other, X_img_avg_c, X_txt_emb, X_txt_other]

        # 4. Video features (Enhanced mode only)
        if self.mode == "enhanced":
            if video_paths:
                vid_results = self.video_proc.process_batch(video_paths)
                X_vid = np.array([[r["avg_motion"], r["max_motion"], r["avg_complexity"], r["num_frames"]] for r in vid_results])
            else:
                X_vid = np.zeros((len(df), 4))
            features_list.append(X_vid)

        X = np.hstack(features_list)

        if self.feature_names is None:
            names = list(tab_names)
            names += [f"img_emb_{i}" for i in range(X_img_emb.shape[1])]
            names += ["img_sentiment", "img_num_objects", "img_brightness", "img_contrast", "img_colorfulness"]
            names += ["img_avg_r", "img_avg_g", "img_avg_b"]
            names += [f"txt_emb_{i}" for i in range(X_txt_emb.shape[1])]
            names += ["txt_sentiment", "txt_length", "txt_word_count"]
            if self.mode == "enhanced":
                names += ["vid_avg_motion", "vid_max_motion", "vid_avg_complexity", "vid_num_frames"]
            self.feature_names = names

        return X

    def fit(self, df: pd.DataFrame, image_paths: list[str] = None, texts: list[str] = None, video_paths: list[str] = None):
        """Trains the model."""
        X = self._extract_all_features(df, image_paths, texts, video_paths, fit_tabular=True)
        y = calculate_ctr(df).values
        self.regressor.fit(X, y)
        print(f"Model training complete (Mode: {self.mode}).")

    def predict(self, df: pd.DataFrame, image_paths: list[str] = None, texts: list[str] = None, video_paths: list[str] = None) -> np.ndarray:
        """Predicts CTR."""
        X = self._extract_all_features(df, image_paths, texts, video_paths, fit_tabular=False)
        return self.regressor.predict(X)

    def evaluate(self, df: pd.DataFrame, image_paths: list[str] = None, texts: list[str] = None, video_paths: list[str] = None) -> dict:
        """Evaluates metrics."""
        y_true = calculate_ctr(df).values
        X = self._extract_all_features(df, image_paths, texts, video_paths, fit_tabular=False)
        y_pred = self.regressor.predict(X)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": float(rmse),
            "r2": float(r2)
        }

    def get_feature_distribution(self, df: pd.DataFrame, image_paths: list[str] = None, texts: list[str] = None, video_paths: list[str] = None):
        """Returns insights."""
        res = {}
        if self.mode != "baseline":
            img_results = self.image_proc.process_batch(image_paths)
            txt_results = self.text_proc.process_batch(texts)
            res.update({
                "visual_sentiment": [r["sentiment"] for r in img_results],
                "textual_sentiment": [r["sentiment"] for r in txt_results],
                "object_counts": [r["num_objects"] for r in img_results],
                "brightness": [r["brightness"] for r in img_results],
                "colorfulness": [r["colorfulness"] for r in img_results]
            })
            if self.mode == "enhanced" and video_paths:
                vid_results = self.video_proc.process_batch(video_paths)
                res["avg_motion"] = [r["avg_motion"] for r in vid_results]
        return res

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str):
        return joblib.load(filepath)
