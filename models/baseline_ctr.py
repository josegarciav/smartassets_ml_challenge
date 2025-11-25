# models/baseline_ctr.py

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features.tabular_features import add_ctr_target, build_tabular_preprocessor, get_feature_and_target_matrices


class BaselineCTRRegressor:
    """
    Baseline CTR regressor using only tabular features.

    The model is a scikit-learn pipeline that combines preprocessing
    (standardization and one-hot encoding) with a gradient boosting
    regressor. It is meant to serve as a simple benchmark before
    adding image embeddings.
    """

    def __init__(self) -> None:
        """
        Initializes the baseline model and its preprocessing pipeline.
        """
        self.preprocessor = build_tabular_preprocessor()
        gbr = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        )
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("model", gbr),
            ]
        )

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Fits the baseline pipeline on the training dataframe.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training dataframe containing all raw columns.
        """
        df_train_ctr = add_ctr_target(df_train)
        X_train_df, y_train = get_feature_and_target_matrices(df_train_ctr)
        self.pipeline.fit(X_train_df, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Produces CTR predictions for a dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the same structure as the training data.

        Returns
        -------
        np.ndarray
            Predicted CTR values.
        """
        df_ctr = add_ctr_target(df)
        X_df, _ = get_feature_and_target_matrices(df_ctr)
        preds = self.pipeline.predict(X_df)
        return preds

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """
        Evaluates the model on a test dataframe using RMSE, MAE and R^2.

        Parameters
        ----------
        df_test : pd.DataFrame
            Test dataframe with raw columns.

        Returns
        -------
        dict
            Dictionary with keys 'rmse', 'mae' and 'r2'.
        """
        df_test_ctr = add_ctr_target(df_test)
        X_test_df, y_true = get_feature_and_target_matrices(df_test_ctr)
        y_pred = self.pipeline.predict(X_test_df)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {"rmse": rmse, "mae": mae, "r2": r2}
