"""统一的 source->target 图指标和 bootstrap 工具。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def validate_adjacency(adjacency: np.ndarray, *, name: str) -> np.ndarray:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.isfinite(adjacency).all():
        raise ValueError(f"{name} contains non-finite values")
    binary = (adjacency != 0).astype(np.int8)
    if np.any(np.diag(binary)):
        raise ValueError(f"{name} contains self loops")
    return binary


def directed_graph_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    """计算有向边 precision/recall/F1 和 reversal-as-one 的 SHD。"""
    predicted = validate_adjacency(predicted, name="predicted")
    truth = validate_adjacency(truth, name="truth")
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth shapes differ")

    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    pred_flat = predicted[off_diagonal].astype(bool)
    truth_flat = truth[off_diagonal].astype(bool)
    tp = int(np.sum(pred_flat & truth_flat))
    fp = int(np.sum(pred_flat & ~truth_flat))
    fn = int(np.sum(~pred_flat & truth_flat))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    shd = 0
    for left in range(truth.shape[0]):
        for right in range(left + 1, truth.shape[0]):
            pred_pair = (predicted[left, right], predicted[right, left])
            truth_pair = (truth[left, right], truth[right, left])
            shd += int(pred_pair != truth_pair)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "shd": int(shd),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def lingam_target_source_to_source_target(
    coefficient_matrix: np.ndarray,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """把 B[target,source] 的非零系数结构转为 A[source,target]。"""
    coefficients = np.asarray(coefficient_matrix, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
        raise ValueError("DirectLiNGAM coefficient matrix must be square")
    if not np.isfinite(coefficients).all():
        raise ValueError("DirectLiNGAM coefficient matrix contains non-finite values")
    adjacency = (np.abs(coefficients).T > tolerance).astype(np.int8)
    np.fill_diagonal(adjacency, 0)
    return adjacency


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_statistic(
    sample_size: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 20260916,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        selected = rng.integers(0, sample_size, size=sample_size)
        estimates[index] = statistic(selected)
    return estimates

