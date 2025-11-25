# features/tabular_features.py

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS = [
    "no_of_days",
    "approved_budget",
    "max_bid_cpm",
    "network_margin",
    "campaign_budget_usd",
    "impressions",
    "clicks",
    "media_cost_usd",
]

CATEGORICAL_COLS = [
    "ext_service_name",
    "channel_name",
    "search_tag_cat",
    "weekday_cat",
    "timezone",
    "advertiser_currency",
    "stats_currency",
]


def add_ctr_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a CTR target column (clicks / impressions) to the dataframe.

    This function creates a new column 'ctr' while protecting against
    division by zero. It does not modify the input dataframe in place:
    a copy is returned.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe containing 'clicks' and 'impressions' columns.

    Returns
    -------
    pd.DataFrame
        New dataframe with a 'ctr' float column appended.
    """
    df_copy = df.copy()
    impressions_safe = df_copy["impressions"].clip(lower=1)
    df_copy["ctr"] = df_copy["clicks"] / impressions_safe
    return df_copy


def build_tabular_preprocessor() -> ColumnTransformer:
    """
    Builds a ColumnTransformer for tabular preprocessing.

    Numeric columns are standardized and categorical columns are
    one-hot encoded. The transformer is intended to be included
    inside a scikit-learn Pipeline used by the baseline model.

    Returns
    -------
    ColumnTransformer
        Configured transformer for tabular data.
    """
    numeric_processor = StandardScaler()
    categorical_processor = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_processor, NUMERIC_COLS),
            ("cat", categorical_processor, CATEGORICAL_COLS),
        ]
    )

    return preprocessor


def get_feature_and_target_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Returns the design matrix (tabular columns only) and target vector.

    This helper expects that 'ctr' has already been added as a column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with all raw columns and a 'ctr' column.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        A pair (X_df, y) where X_df contains only the tabular feature
        columns and y is the numeric CTR target.
    """
    if "ctr" not in df.columns:
        raise ValueError("Dataframe must contain a 'ctr' column. Call add_ctr_target first.")

    X_df = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["ctr"].values.astype("float32")
    return X_df, y
