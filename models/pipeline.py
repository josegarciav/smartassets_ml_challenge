import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from features.image import ImageProcessor, creative_id_to_path
from features.text import TextProcessor
from features.tabular import TabularProcessor
from tqdm import tqdm

class MultiModalModel:
    """
    Integrates visual, textual, and tabular features into a single regressor.
    """
    def __init__(self, images_root: str, device: str = None):
        self.images_root = images_root
        self.image_processor = ImageProcessor(device=device)
        self.text_processor = TextProcessor(device=device)
        self.tabular_processor = TabularProcessor()
        self.model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

    def _extract_features(self, df: pd.DataFrame, is_training: bool = False) -> np.ndarray:
        print("Extracting tabular features...")
        if is_training:
            self.tabular_processor.fit(df)
        X_tab = self.tabular_processor.transform(df)

        print("Extracting image features and sentiment...")
        creative_ids = df["creative_id"].astype(str).tolist()
        image_paths = [creative_id_to_path(cid, self.images_root) for cid in creative_ids]

        X_img_emb = self.image_processor.get_embeddings(image_paths)
        X_img_sent = self.image_processor.get_sentiment_scores(image_paths)
        X_img_colors = self.image_processor.get_color_stats(image_paths)

        print("Extracting text features...")
        # Combine search tags and keywords for text embedding
        texts = (df["search_tags"].fillna("") + " " + df["keywords"].fillna("")).tolist()
        X_txt = self.text_processor.get_embeddings(texts)

        # Concatenate all features
        X = np.concatenate([X_tab, X_img_emb, X_img_sent, X_img_colors, X_txt], axis=1)
        return X

    def fit(self, df: pd.DataFrame):
        """Fits the multi-modal model."""
        # Calculate CTR as target
        y = df["clicks"] / df["impressions"].clip(lower=1)
        X = self._extract_features(df, is_training=True)
        print(f"Fitting model with feature matrix shape: {X.shape}")
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts CTR for the given dataframe."""
        X = self._extract_features(df, is_training=False)
        return self.model.predict(X)
