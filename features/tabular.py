import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TabularProcessor:
    """
    Handles tabular data preprocessing.
    Scales numeric features and one-hot encodes categorical features.
    """
    def __init__(self):
        self.numeric_features = [
            "no_of_days", "approved_budget", "max_bid_cpm",
            "network_margin", "campaign_budget_usd", "impressions",
            "media_cost_usd"
        ]
        self.categorical_features = [
            "ext_service_name", "advertiser_currency",
            "channel_name", "search_tag_cat", "weekday_cat"
        ]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features),
            ]
        )

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessor on the dataframe."""
        df_copy = df.copy()
        # Ensure numeric columns are numeric
        for col in self.numeric_features:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        self.preprocessor.fit(df_copy)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms the dataframe into a dense feature matrix."""
        df_copy = df.copy()
        for col in self.numeric_features:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        return self.preprocessor.transform(df_copy)
