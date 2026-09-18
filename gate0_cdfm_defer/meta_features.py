"""仅从观测矩阵 X 提取 Gate-0 Router 的简单 meta-features。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from generate_data import MANIFEST_PATH, ROOT


RESULTS_DIR = ROOT / "results"
FEATURES_PATH = RESULTS_DIR / "meta_features.csv"
TOP_PAIR_COUNT = 10
CV_SPLITS = 3
CV_SEED = 20260916
GAIN_EPSILON = 1e-12
FEATURE_COLUMNS = [
    "mean_abs_skewness",
    "mean_abs_excess_kurtosis",
    "mean_abs_pearson_correlation",
    "mean_abs_spearman_correlation",
    "nonlinearity_gain_median",
    "nonlinearity_gain_mean",
    "nonlinearity_gain_max",
]


def _off_diagonal_upper_mean_abs(matrix: np.ndarray) -> float:
    indices = np.triu_indices_from(matrix, k=1)
    return float(np.mean(np.abs(matrix[indices])))


def _top_correlated_pairs(x: np.ndarray, count: int) -> list[tuple[int, int]]:
    correlation = np.corrcoef(x, rowvar=False)
    pairs = [
        (left, right)
        for left in range(x.shape[1])
        for right in range(left + 1, x.shape[1])
    ]
    pairs.sort(key=lambda pair: abs(float(correlation[pair])), reverse=True)
    return pairs[:count]


def _directional_gain(predictor: np.ndarray, outcome: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    linear_losses: list[float] = []
    nonlinear_losses: list[float] = []
    predictor_2d = predictor.reshape(-1, 1)
    for train_indices, test_indices in folds:
        linear = LinearRegression()
        nonlinear = make_pipeline(
            SplineTransformer(n_knots=7, degree=3, include_bias=False),
            Ridge(alpha=1e-3),
        )
        linear.fit(predictor_2d[train_indices], outcome[train_indices])
        nonlinear.fit(predictor_2d[train_indices], outcome[train_indices])
        linear_losses.append(mean_squared_error(outcome[test_indices], linear.predict(predictor_2d[test_indices])))
        nonlinear_losses.append(mean_squared_error(outcome[test_indices], nonlinear.predict(predictor_2d[test_indices])))
    mse_linear = float(np.mean(linear_losses))
    mse_nonlinear = float(np.mean(nonlinear_losses))
    return (mse_linear - mse_nonlinear) / (mse_linear + GAIN_EPSILON)


def extract_meta_features(x: np.ndarray) -> dict[str, float]:
    """提取固定的 7 个特征；不接收 seed、lambda、图或算法输出。"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("X must be a finite two-dimensional matrix")
    if np.any(x.std(axis=0) <= 0):
        raise ValueError("X contains a constant column")

    pearson = np.corrcoef(x, rowvar=False)
    ranked = np.column_stack([rankdata(x[:, column]) for column in range(x.shape[1])])
    spearman = np.corrcoef(ranked, rowvar=False)
    folds = list(KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED).split(x))
    gains: list[float] = []
    for left, right in _top_correlated_pairs(x, TOP_PAIR_COUNT):
        gains.append(_directional_gain(x[:, left], x[:, right], folds))
        gains.append(_directional_gain(x[:, right], x[:, left], folds))

    return {
        "mean_abs_skewness": float(np.mean(np.abs(skew(x, axis=0, bias=False)))),
        "mean_abs_excess_kurtosis": float(np.mean(np.abs(kurtosis(x, axis=0, fisher=True, bias=False)))),
        "mean_abs_pearson_correlation": _off_diagonal_upper_mean_abs(pearson),
        "mean_abs_spearman_correlation": _off_diagonal_upper_mean_abs(spearman),
        "nonlinearity_gain_median": float(np.median(gains)),
        "nonlinearity_gain_mean": float(np.mean(gains)),
        "nonlinearity_gain_max": float(np.max(gains)),
    }


def _atomic_write(rows: list[dict[str, object]]) -> None:
    columns = ["dataset_id", "base_seed", "lambda", *FEATURE_COLUMNS]
    temp_path = FEATURES_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, FEATURES_PATH)


def run(*, limit: int | None = None, rerun: bool = False) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    if limit is not None:
        manifest = manifest.iloc[:limit]
    existing = [] if rerun or not FEATURES_PATH.exists() else pd.read_csv(FEATURES_PATH).to_dict(orient="records")
    rows_by_id = {str(row["dataset_id"]): row for row in existing}
    total = len(manifest)
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        dataset_id = str(row["dataset_id"])
        if dataset_id in rows_by_id:
            print(f"[{ordinal}/{total}] skip {dataset_id}")
            continue
        with np.load(ROOT / str(row["dataset_path"])) as payload:
            x = np.asarray(payload["X"], dtype=float)
        features = extract_meta_features(x)
        rows_by_id[dataset_id] = {
            "dataset_id": dataset_id,
            "base_seed": int(row["base_seed"]),
            "lambda": float(row["lambda"]),
            **features,
        }
        _atomic_write([rows_by_id[key] for key in sorted(rows_by_id)])
        print(f"[{ordinal}/{total}] {dataset_id}: gain={features['nonlinearity_gain_mean']:.4f}")
    print(f"Meta-feature extraction complete: {len(rows_by_id)}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, rerun=args.rerun)


if __name__ == "__main__":
    main()
