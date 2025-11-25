# models/multimodal_ctr.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features.tabular_features import add_ctr_target, get_feature_and_target_matrices
from features.image_embedder import ImageEmbedder, creative_id_to_path


class MultimodalCTRRegressor:
    """
    CTR regressor that combines tabular features with image embeddings.

    The model:
        1. Extracts tabular matrices (no one-hot here; we rely on the
           numeric columns only for simplicity).
        2. Extracts CLIP embeddings for each creative.
        3. Concatenates [tabular_num | image_emb] and fits a regressor.

    This avoids building a complex deep network while still providing
    a genuine multimodal baseline.
    """

    def __init__(self, images_root: str, device: str = "cpu") -> None:
        """
        Initializes the multimodal model and image embedder.

        Parameters
        ----------
        images_root : str
            Root folder where creative images are stored.
        device : str
            Device string for the image embedder.
        """
        self.images_root = images_root
        self.image_embedder = ImageEmbedder(device=device)
        self.tabular_scaler = StandardScaler()
        self.image_scaler = StandardScaler()
        self.model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        )

    def _build_image_paths(self, df: pd.DataFrame) -> list[str]:
        """
        Builds a list of image paths corresponding to the dataframe rows.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing a 'creative_id' column.

        Returns
        -------
        list[str]
            List of image file paths aligned with the dataframe order.
        """
        paths: list[str] = []
        for cid in df["creative_id"].astype(str).tolist():
            path = creative_id_to_path(cid, self.images_root)
            paths.append(path)
        return paths

    def _build_feature_matrix(self, df: pd.DataFrame, fit_scalers: bool) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the concatenated feature matrix and target vector.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with raw columns and 'creative_id'.
        fit_scalers : bool
            If True, scalers are fitted on the provided data; otherwise,
            existing scalers are used to transform the features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pair (X, y) where X is the combined feature matrix and y
            is the CTR target vector.
        """
        df_ctr = add_ctr_target(df)
        # Reuse the same helper but we will manually select numeric part
        X_df, y = get_feature_and_target_matrices(df_ctr)
        # Keep only numeric columns for simplicity
        numeric_cols = [
            "no_of_days",
            "approved_budget",
            "max_bid_cpm",
            "network_margin",
            "campaign_budget_usd",
            "impressions",
            "clicks",
            "media_cost_usd",
        ]
        X_tab_num = X_df[numeric_cols].values.astype("float32")

        image_paths = self._build_image_paths(df_ctr)
        X_img = self.image_embedder.encode_paths(image_paths)

        if fit_scalers:
            X_tab_scaled = self.tabular_scaler.fit_transform(X_tab_num)
            X_img_scaled = self.image_scaler.fit_transform(X_img)
        else:
            X_tab_scaled = self.tabular_scaler.transform(X_tab_num)
            X_img_scaled = self.image_scaler.transform(X_img)

        X = np.concatenate([X_tab_scaled, X_img_scaled], axis=1)
        return X, y.astype("float32")

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fits the multimodal regressor on the training dataframe.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training dataframe containing columns used for tabular
            features and a 'creative_id' column.
        """
        X_train, y_train = self._build_feature_matrix(df_train, fit_scalers=True)
        self.model.fit(X_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts CTR for a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the same structure as training data.

        Returns
        -------
        np.ndarray
            Predicted CTR values.
        """
        X, _ = self._build_feature_matrix(df, fit_scalers=False)
        preds = self.model.predict(X)
        return preds

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """
        Evaluates the model on a test dataframe.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test dataframe including 'creative_id'.

        Returns
        -------
        dict
            Dictionary with RMSE, MAE and R^2.
        """
        X_test, y_true = self._build_feature_matrix(df_test, fit_scalers=False)
        y_pred = self.model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {"rmse": rmse, "mae": mae, "r2": r2}
