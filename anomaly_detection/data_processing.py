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
    """"""
    # --- Fill nulls - we only have int and bool
    lf = lf.with_columns(
        cs.numeric().fill_null(
            cs.numeric().mean().over("process_pid")
        ),  # mean is computed by process_pid
        cs.boolean().fill_null(False),
    )

    # remove id var
    df_clean = lf.drop("process_pis").collect()

    # --- Convert to numpy and StandardScaler
    feature_cols = df_clean.columns
    X = df_clean.to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Scaled {X_scaled.shape[1]} features, {X_scaled.shape[0]} samples\n")
    return X_scaled, feature_cols


X_scaled, feature_cols = preprocess(lf)

print(X_scaled)
print("----")
print(feature_cols)
