import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class TabularProcessor:
    """
    Handles preprocessing of campaign metadata using scikit-learn.
    Standardizes numeric features and one-hot encodes categorical features.
    Now includes automated feature engineering.
    """
    BASE_NUMERIC_COLS = [
        "no_of_days",
        "approved_budget",
        "max_bid_cpm",
        "network_margin",
        "campaign_budget_usd",
    ]

    ENGINEERED_NUMERIC_COLS = [
        "budget_per_day",
        "url_len",
        "month",
        "day_of_week"
    ]

    CATEGORICAL_COLS = [
        "ext_service_name",
        "channel_name",
        "search_tag_cat",
        "weekday_cat",
    ]

    def __init__(self):
        self.numeric_cols = self.BASE_NUMERIC_COLS + self.ENGINEERED_NUMERIC_COLS
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.CATEGORICAL_COLS),
            ]
        )

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features from raw metadata."""
        df = df.copy()

        # 1. Time-based features
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            df["month"] = df["time"].dt.month
            df["day_of_week"] = df["time"].dt.dayofweek
        else:
            df["month"] = 0
            df["day_of_week"] = 0

        # 2. Budget efficiency
        # Handle division by zero
        df["budget_per_day"] = df["approved_budget"] / df["no_of_days"].clip(lower=1)

        # 3. URL context
        df["url_len"] = df["landing_page"].fillna("").apply(len)

        return df

    def fit(self, df: pd.DataFrame):
        """Fits the preprocessor on the provided dataframe."""
        df_eng = self.engineer_features(df)
        cols_to_use = self.numeric_cols + self.CATEGORICAL_COLS
        self.preprocessor.fit(df_eng[cols_to_use])
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms the dataframe into a dense feature matrix."""
        df_eng = self.engineer_features(df)
        cols_to_use = self.numeric_cols + self.CATEGORICAL_COLS
        return self.preprocessor.transform(df_eng[cols_to_use])

    def get_feature_names(self):
        """Returns the names of the features after transformation."""
        # Helper to get names from preprocessor
        num_names = self.numeric_cols
        cat_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.CATEGORICAL_COLS).tolist()
        return num_names + cat_names

def calculate_ctr(df: pd.DataFrame) -> pd.Series:
    """Calculates CTR (Clicks / Impressions) and handles division by zero."""
    impressions = df["impressions"].clip(lower=1)
    return df["clicks"] / impressions
