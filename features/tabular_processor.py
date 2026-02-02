import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TabularProcessor:
    """
    Handles preprocessing of campaign metadata using scikit-learn.
    Standardizes numeric features and one-hot encodes categorical features.
    """
    NUMERIC_COLS = [
        "no_of_days",
        "approved_budget",
        "max_bid_cpm",
        "network_margin",
        "campaign_budget_usd",
    ]

    CATEGORICAL_COLS = [
        "ext_service_name",
        "channel_name",
        "search_tag_cat",
        "weekday_cat",
    ]

    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.NUMERIC_COLS),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS),
            ]
        )

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessor on the provided dataframe."""
        # Ensure columns exist
        cols_to_use = self.NUMERIC_COLS + self.CATEGORICAL_COLS
        available_cols = [c for c in cols_to_use if c in df.columns]
        if len(available_cols) < len(cols_to_use):
            print(f"Warning: Some columns are missing. Using only: {available_cols}")

        self.preprocessor.fit(df[available_cols])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms the dataframe into a dense feature matrix."""
        cols_to_use = self.NUMERIC_COLS + self.CATEGORICAL_COLS
        return self.preprocessor.transform(df[cols_to_use])

def calculate_ctr(df: pd.DataFrame) -> pd.Series:
    """Calculates CTR (Clicks / Impressions) and handles division by zero."""
    impressions = df["impressions"].clip(lower=1)
    return df["clicks"] / impressions
