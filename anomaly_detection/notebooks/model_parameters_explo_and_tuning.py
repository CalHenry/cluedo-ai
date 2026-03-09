import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from sklearn.ensemble import IsolationForest
    import matplotlib.pyplot as plt
    from pathlib import Path
    import polars as pl

    from anomaly_detection.data_processing import preprocess

    return IsolationForest, Path, mo, np, pl, plt, preprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Unsupervised learning model

    Isolation Forest hyperparameters:
        - n_estimators
    """)
    return


@app.cell
def _(Path, pl):
    # Input data comes froms the aggregated logs. Prepare dataset in parquet is found in 'logs/data/processed/'

    data_path = Path("logs/data/processed/")
    lf = pl.scan_parquet(data_path.glob("*parquet"))
    return


@app.cell
def _(preprocess):
    X_scaled, feature_cols = preprocess()
    return (X_scaled,)


@app.cell
def _(IsolationForest, X_scaled, np, plt):
    # n_estimator: This cell look at the correct value for n_estimator

    estimator_range = [50, 100, 150, 200, 250, 300, 400, 500]
    score_variance = []

    for n in estimator_range:
        # Run 10 times with different seeds, measure score std
        run_scores = []
        for seed in range(10):
            iso = IsolationForest(
                n_estimators=n, contamination=0.05, random_state=seed
            )
            iso.fit(X_scaled)
            scores = iso.decision_function(X_scaled)
            run_scores.append(scores)
        # Mean std across all samples across all runs
        score_variance.append(np.std(run_scores, axis=0).mean())

    plt.plot(estimator_range, score_variance, marker="o")
    plt.xlabel("n_estimators")
    plt.ylabel("Score variance across seeds")
    plt.title("Score stability vs n_estimators")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
