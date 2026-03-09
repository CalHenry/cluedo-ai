import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.preprocessing import StandardScaler

# 1. Read data
lf = pl.scan_parquet(
    "logs/data/processed/*.parquet",
    cast_options=pl.ScanCastOptions(integer_cast="upcast"),
)


def preprocess(lf: pl.LazyFrame) -> tuple[np.ndarray, list]:
    """
    Prepare the variables from the aggregated log dataset for Isolation Forest model.
    - Fill any nulls (we only have int and bool variables)
    - Convert Boolean variables to Integer
    - Remove the ID var (process_pid)

    Return:
        X_scaled: numpy array of the features
        features_cols: list of the features name
    """
    # --- Fill nulls
    lf = lf.with_columns(
        cs.numeric().fill_null(
            cs.numeric().mean().over("process_pid")
        ),  # mean is computed by process_pid
        cs.boolean().fill_null(False),
    )

    # --- Bool vars to Int
    lf = lf.cast({cs.Boolean(): pl.Int32})

    # --- Remove id var
    df_clean = lf.drop("process_pid").collect()

    # --- Convert to numpy and StandardScaler
    feature_cols = df_clean.columns
    X = df_clean.to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Scaled {X_scaled.shape[1]} features, {X_scaled.shape[0]} samples\n")

    return X_scaled, feature_cols
