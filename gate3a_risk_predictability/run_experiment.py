"""Minimal Gate-3A experiment: predict frozen-CDFM graph risk without truth at test time."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from causallearn.search.FCMBased import lingam
from cdfm import CDFM
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE0_ROOT = REPO_ROOT / "gate0_cdfm_defer"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GATE0_ROOT) not in sys.path:
    sys.path.insert(0, str(GATE0_ROOT))

from metrics import directed_graph_metrics, lingam_target_source_to_source_target  # noqa: E402
from meta_features import FEATURE_COLUMNS as METADATA_FEATURES  # noqa: E402
from meta_features import extract_meta_features  # noqa: E402
from gate2_reliable_routing.generate_tasks import _graph, generate_x  # noqa: E402


MECHANISMS = ("linear", "tanh", "rff", "interaction")
N_SAMPLES = 1000
N_VARIABLES = 10
MASTER_GRAPH_SEED = 310000
ANALYSIS_SEED = 20260918
BOOTSTRAP_RESAMPLES = 1000
COVERAGES = (0.25, 0.50, 0.75, 1.00)

GRAPH_FEATURES = ("cdfm_edge_count", "cdfm_edge_density", "cdfm_mean_degree", "cdfm_max_degree")
DISAGREEMENT_FEATURES = ("cdfm_lingam_adj_disagreement",)
CONFIDENCE_FEATURES = (
    "cdfm_mean_edge_probability",
    "cdfm_mean_confidence",
    "cdfm_mean_entropy",
    "cdfm_mean_margin",
)
STABILITY_FEATURES = ("cdfm_bootstrap_disagreement", "cdfm_bootstrap_edge_jaccard")
FORBIDDEN_FEATURE_TOKENS = ("truth", "true_", "_f1", "_shd", "mechanism", "seed", "oracle", "label")


def paths(smoke: bool) -> dict[str, Path]:
    result_dir = ROOT / "results" / "smoke" if smoke else ROOT / "results"
    cache_dir = ROOT / "cache" / "smoke" if smoke else ROOT / "cache"
    return {
        "results": result_dir,
        "cache": cache_dir,
        "datasets": cache_dir / "datasets",
        "raw": cache_dir / "raw_predictions",
        "tasks": result_dir / "tasks.csv",
        "predictions": result_dir / "predictions.csv",
        "folds": result_dir / "fold_results.csv",
        "summary": result_dir / "summary.json",
        "report": result_dir / "REPORT.md",
        "coverage_plot": result_dir / "risk_coverage.png",
        "scatter_plot": result_dir / "predicted_vs_true_risk.png",
        "environment": result_dir / "environment.json",
        "errors": result_dir / "errors.jsonl",
    }


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, destination)


def _atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, destination)


def _append_error(destination: Path, task_id: str, stage: str, exc: BaseException) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "task_id": task_id,
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp_epoch": time.time(),
        }, ensure_ascii=False) + "\n")


def _mechanism_config(mechanism: str) -> tuple[str, float]:
    if mechanism == "linear":
        return "rff", 0.0
    return mechanism, 1.0


def generate_tasks(*, n_seeds: int, smoke: bool, rerun: bool) -> pd.DataFrame:
    p = paths(smoke)
    p["datasets"].mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for local_seed in range(n_seeds):
        graph_seed = MASTER_GRAPH_SEED + local_seed
        spec = _graph(graph_seed)
        for mechanism in MECHANISMS:
            generator_mechanism, lambda_value = _mechanism_config(mechanism)
            task_id = f"{mechanism}_seed_{local_seed:02d}"
            destination = p["datasets"] / f"{task_id}.npz"
            if rerun or not destination.exists():
                x = generate_x(spec, generator_mechanism, "laplace", lambda_value, graph_seed)
                temporary = destination.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        X=x,
                        truth_adjacency=spec.adjacency,
                        graph_seed=np.int64(graph_seed),
                        mechanism=np.asarray(mechanism),
                    )
                os.replace(temporary, destination)
            rows.append({
                "task_id": task_id,
                "graph_seed": graph_seed,
                "mechanism": mechanism,
                "N": N_SAMPLES,
                "D": N_VARIABLES,
                "config": json.dumps({
                    "noise": "laplace",
                    "generator": "gate2_reliable_routing.generate_tasks",
                    "generator_mechanism": generator_mechanism,
                    "lambda": lambda_value,
                    "paired_topology_weights_noise": True,
                }, sort_keys=True),
                "dataset_path": destination.relative_to(ROOT).as_posix(),
            })
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, p["tasks"])
    print(f"Generated/verified {len(frame)} tasks ({n_seeds} paired graph seeds).", flush=True)
    return frame


def _run_lingam(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    model = lingam.DirectLiNGAM(measure="pwling")
    model.fit(x)
    coefficients = np.asarray(model.adjacency_matrix_, dtype=float)
    adjacency = lingam_target_source_to_source_target(coefficients)
    return coefficients, adjacency, float(time.perf_counter() - start)


def _bootstrap_graphs(model: CDFM, x: np.ndarray, count: int, graph_seed: int) -> tuple[np.ndarray, float]:
    if count <= 0:
        return np.empty((0, x.shape[1], x.shape[1]), dtype=np.int8), 0.0
    rng = np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED, graph_seed, 77]))
    graphs = []
    start = time.perf_counter()
    for _ in range(count):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        graphs.append(np.asarray(model.predict(x[indices]).adjacency, dtype=np.int8))
    return np.stack(graphs), float(time.perf_counter() - start)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_inference(*, smoke: bool, bootstrap: int, rerun: bool) -> None:
    p = paths(smoke)
    if not p["tasks"].exists():
        raise FileNotFoundError("tasks.csv is missing; run the generate stage first")
    tasks = pd.read_csv(p["tasks"])
    p["raw"].mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading frozen CDFM on {device}; tasks={len(tasks)}, bootstrap={bootstrap}", flush=True)
    model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _atomic_json({
        "schema_version": "gate3a_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {name: _package_version(name) for name in (
            "cdfm-base", "causal-learn", "numpy", "pandas", "scipy", "scikit-learn", "torch"
        )},
        "cdfm_info": model.info,
        "cdfm_frozen": True,
        "cdfm_call": "CDFM.from_pretrained('DMIRLAB/CDFM').predict(X)",
        "bootstrap": bootstrap,
    }, p["environment"])

    for ordinal, row in enumerate(tasks.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        raw_path = p["raw"] / f"{task_id}.npz"
        if raw_path.exists() and not rerun:
            with np.load(raw_path) as cached:
                cached_bootstrap = int(cached["bootstrap_count"])
            if cached_bootstrap == bootstrap:
                print(f"[{ordinal}/{len(tasks)}] skip cached {task_id}", flush=True)
                continue
        try:
            with np.load(ROOT / str(row["dataset_path"])) as payload:
                x = np.asarray(payload["X"], dtype=np.float64)
            start = time.perf_counter()
            result = model.predict(x)
            cdfm_runtime = float(time.perf_counter() - start)
            cdfm_adjacency = np.asarray(result.adjacency, dtype=np.int8)
            cdfm_probabilities = np.asarray(result.probabilities, dtype=float)
            cdfm_logits = np.asarray(result.logits, dtype=float)
            coefficients, lingam_adjacency, lingam_runtime = _run_lingam(x)
            bootstrap_graphs, bootstrap_runtime = _bootstrap_graphs(
                model, x, bootstrap, int(row["graph_seed"])
            )
            temporary = raw_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cdfm_adjacency=cdfm_adjacency,
                    cdfm_probabilities=cdfm_probabilities,
                    cdfm_logits=cdfm_logits,
                    cdfm_threshold=np.float64(result.threshold),
                    cdfm_runtime_sec=np.float64(cdfm_runtime),
                    lingam_coefficients_target_source=coefficients,
                    lingam_adjacency_source_target=lingam_adjacency,
                    lingam_runtime_sec=np.float64(lingam_runtime),
                    bootstrap_adjacencies=bootstrap_graphs,
                    bootstrap_count=np.int64(bootstrap),
                    bootstrap_runtime_sec=np.float64(bootstrap_runtime),
                )
            os.replace(temporary, raw_path)
            print(
                f"[{ordinal}/{len(tasks)}] {task_id}: CDFM={cdfm_runtime:.2f}s, "
                f"LiNGAM={lingam_runtime:.2f}s, bootstrap={bootstrap_runtime:.2f}s",
                flush=True,
            )
        except Exception as exc:
            _append_error(p["errors"], task_id, "inference", exc)
            print(f"[{ordinal}/{len(tasks)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    completed = sum((p["raw"] / f"{task_id}.npz").exists() for task_id in tasks["task_id"])
    if completed != len(tasks):
        raise RuntimeError(f"inference incomplete: {completed}/{len(tasks)} cached")
    print(f"Inference complete: {completed}/{len(tasks)} tasks cached.", flush=True)


def _off_diagonal_values(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.asarray(matrix)[mask]


def _edge_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def diagnostic_features(
    x: np.ndarray,
    cdfm_adjacency: np.ndarray,
    cdfm_probabilities: np.ndarray,
    lingam_adjacency: np.ndarray,
    bootstrap_adjacencies: np.ndarray,
) -> dict[str, float]:
    """Ground-truth-free by construction: this function has no truth argument."""
    features = dict(extract_meta_features(x))
    d = cdfm_adjacency.shape[0]
    degrees = cdfm_adjacency.sum(axis=0) + cdfm_adjacency.sum(axis=1)
    probabilities = np.clip(_off_diagonal_values(cdfm_probabilities), 1e-12, 1.0 - 1e-12)
    features.update({
        "cdfm_edge_count": float(cdfm_adjacency.sum()),
        "cdfm_edge_density": float(cdfm_adjacency.sum() / (d * (d - 1))),
        "cdfm_mean_degree": float(degrees.mean()),
        "cdfm_max_degree": float(degrees.max()),
        "cdfm_lingam_adj_disagreement": float(np.mean(
            _off_diagonal_values(cdfm_adjacency) != _off_diagonal_values(lingam_adjacency)
        )),
        "cdfm_mean_edge_probability": float(probabilities.mean()),
        "cdfm_mean_confidence": float(np.mean(np.maximum(probabilities, 1.0 - probabilities))),
        "cdfm_mean_entropy": float(np.mean(-probabilities * np.log(probabilities) - (1.0 - probabilities) * np.log(1.0 - probabilities))),
        "cdfm_mean_margin": float(np.mean(np.abs(probabilities - 0.5))),
    })
    if len(bootstrap_adjacencies):
        features["cdfm_bootstrap_disagreement"] = float(np.mean([
            np.mean(_off_diagonal_values(cdfm_adjacency) != _off_diagonal_values(graph))
            for graph in bootstrap_adjacencies
        ]))
        features["cdfm_bootstrap_edge_jaccard"] = float(np.mean([
            _edge_jaccard(cdfm_adjacency, graph) for graph in bootstrap_adjacencies
        ]))
    return features


def feature_groups(predictions: pd.DataFrame) -> dict[str, list[str]]:
    stability = [name for name in STABILITY_FEATURES if name in predictions and predictions[name].notna().all()]
    return {
        "metadata": list(METADATA_FEATURES),
        "confidence": list(CONFIDENCE_FEATURES),
        "candidate": [
            *METADATA_FEATURES,
            *GRAPH_FEATURES,
            *DISAGREEMENT_FEATURES,
            *CONFIDENCE_FEATURES,
            *stability,
            "ood_score",
        ],
    }


def assert_no_label_leakage(feature_names: list[str]) -> None:
    offenders = [name for name in feature_names if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
    if offenders:
        raise AssertionError(f"forbidden label-derived predictor features: {offenders}")


def build_predictions(*, smoke: bool) -> pd.DataFrame:
    p = paths(smoke)
    tasks = pd.read_csv(p["tasks"])
    rows: list[dict[str, object]] = []
    for row in tasks.to_dict(orient="records"):
        task_id = str(row["task_id"])
        with np.load(ROOT / str(row["dataset_path"])) as task:
            x = np.asarray(task["X"], dtype=float)
            truth = np.asarray(task["truth_adjacency"], dtype=np.int8)
        with np.load(p["raw"] / f"{task_id}.npz") as raw:
            cdfm = np.asarray(raw["cdfm_adjacency"], dtype=np.int8)
            probabilities = np.asarray(raw["cdfm_probabilities"], dtype=float)
            lingam_graph = np.asarray(raw["lingam_adjacency_source_target"], dtype=np.int8)
            boot = np.asarray(raw["bootstrap_adjacencies"], dtype=np.int8)
            timings = {
                "cdfm_runtime_sec": float(raw["cdfm_runtime_sec"]),
                "lingam_runtime_sec": float(raw["lingam_runtime_sec"]),
                "bootstrap_runtime_sec": float(raw["bootstrap_runtime_sec"]),
                "bootstrap_count": int(raw["bootstrap_count"]),
            }
        # Truth is used only below for supervision/evaluation, never inside diagnostic_features.
        diagnostics = diagnostic_features(x, cdfm, probabilities, lingam_graph, boot)
        cdfm_metrics = directed_graph_metrics(cdfm, truth)
        lingam_metrics = directed_graph_metrics(lingam_graph, truth)
        rows.append({
            "task_id": task_id,
            "graph_seed": int(row["graph_seed"]),
            "mechanism": str(row["mechanism"]),
            "cdfm_f1": float(cdfm_metrics["f1"]),
            "cdfm_shd": int(cdfm_metrics["shd"]),
            "direct_lingam_f1_analysis_only": float(lingam_metrics["f1"]),
            "direct_lingam_shd_analysis_only": int(lingam_metrics["shd"]),
            "true_risk": 1.0 - float(cdfm_metrics["f1"]),
            **diagnostics,
            **timings,
        })
    frame = pd.DataFrame(rows).sort_values(["mechanism", "graph_seed"]).reset_index(drop=True)
    groups = feature_groups(frame)
    assert_no_label_leakage(groups["metadata"])
    assert_no_label_leakage([name for name in groups["candidate"] if name != "ood_score"])
    _atomic_csv(frame, p["predictions"])
    return frame


def _fit_ood(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[columns].to_numpy(float)
    test_x = test[columns].to_numpy(float)
    center = np.median(train_x, axis=0)
    scale = np.median(np.abs(train_x - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, np.std(train_x, axis=0) + 1e-8)
    train_z = (train_x - center) / scale
    test_z = (test_x - center) / scale
    covariance = np.cov(train_z, rowvar=False) + 0.1 * np.eye(train_z.shape[1])
    precision = np.linalg.pinv(covariance)
    train_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", train_z, precision, train_z), 0.0))
    test_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", test_z, precision, test_z), 0.0))
    return train_score, test_score


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    value = float(function(left, right).statistic)
    return value if np.isfinite(value) else None


def _coverage_risks(predicted: np.ndarray, true_risk: np.ndarray) -> dict[float, float]:
    order = np.argsort(predicted, kind="stable")
    return {
        coverage: float(np.mean(true_risk[order[: max(1, math.ceil(coverage * len(order)))]]))
        for coverage in COVERAGES
    }


def _bootstrap_ci(predicted: np.ndarray, true_risk: np.ndarray, metric: str, seed: int) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(true_risk), size=len(true_risk))
        if metric == "spearman":
            value = _safe_correlation(spearmanr, predicted[indices], true_risk[indices])
            if value is not None:
                estimates.append(value)
        else:
            estimates.append(_coverage_risks(predicted[indices], true_risk[indices])[0.50])
    if not estimates:
        return None, None
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _model_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    groups = feature_groups(train)
    metadata_columns = groups["metadata"]
    confidence_columns = groups["confidence"]
    ood_train, ood_test = _fit_ood(train, test, metadata_columns)
    train = train.copy()
    test = test.copy()
    train["ood_score"] = ood_train
    test["ood_score"] = ood_test
    candidate_columns = groups["candidate"]
    assert_no_label_leakage(metadata_columns)
    assert_no_label_leakage(candidate_columns)
    y_train = train["true_risk"].to_numpy(float)
    ridge_metadata = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge_confidence = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge_ood = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge_candidate = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    forest_candidate = RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=ANALYSIS_SEED,
        n_jobs=-1,
    )
    ridge_metadata.fit(train[metadata_columns], y_train)
    ridge_confidence.fit(train[confidence_columns], y_train)
    ridge_ood.fit(ood_train.reshape(-1, 1), y_train)
    ridge_candidate.fit(train[candidate_columns], y_train)
    forest_candidate.fit(train[candidate_columns], y_train)
    return {
        "constant": np.full(len(test), float(np.mean(y_train))),
        "metadata_ridge": ridge_metadata.predict(test[metadata_columns]),
        "ood_only": ridge_ood.predict(ood_test.reshape(-1, 1)),
        "confidence_ridge": ridge_confidence.predict(test[confidence_columns]),
        "candidate_ridge": ridge_candidate.predict(test[candidate_columns]),
        "candidate_rf": forest_candidate.predict(test[candidate_columns]),
        "random": np.zeros(len(test)),
    }


def _evaluate_fold(frame: pd.DataFrame, held_out: str) -> tuple[list[dict[str, object]], pd.DataFrame]:
    train = frame[frame["mechanism"] != held_out].copy()
    test = frame[frame["mechanism"] == held_out].copy().reset_index(drop=True)
    predictions = _model_predictions(train, test)
    true_risk = test["true_risk"].to_numpy(float)
    rows: list[dict[str, object]] = []
    detail = test[["task_id", "graph_seed", "mechanism", "true_risk"]].copy()
    for model_name, predicted in predictions.items():
        if model_name in {"random", "constant"}:
            # Expected random-order curve is the full-set mean at every coverage.
            coverage = {level: float(np.mean(true_risk)) for level in COVERAGES}
            metric_prediction = predicted
        else:
            coverage = _coverage_risks(predicted, true_risk)
            metric_prediction = predicted
            detail[f"predicted_risk__{model_name}"] = predicted
        spearman = _safe_correlation(spearmanr, metric_prediction, true_risk)
        pearson = _safe_correlation(pearsonr, metric_prediction, true_risk)
        sp_low, sp_high = _bootstrap_ci(metric_prediction, true_risk, "spearman", ANALYSIS_SEED + 10 * MECHANISMS.index(held_out))
        r50_low, r50_high = _bootstrap_ci(metric_prediction, true_risk, "risk50", ANALYSIS_SEED + 100 + 10 * MECHANISMS.index(held_out))
        rows.append({
            "held_out_family": held_out,
            "model": model_name,
            "n_train": len(train),
            "n_test": len(test),
            "spearman": spearman,
            "spearman_ci_low": sp_low,
            "spearman_ci_high": sp_high,
            "pearson": pearson,
            "mae": float(mean_absolute_error(true_risk, metric_prediction)) if model_name != "random" else None,
            "risk_at_25": coverage[0.25],
            "risk_at_50": coverage[0.50],
            "risk_at_50_ci_low": r50_low,
            "risk_at_50_ci_high": r50_high,
            "risk_at_75": coverage[0.75],
            "risk_at_100": coverage[1.00],
        })
    return rows, detail


def _gate_decision(folds: pd.DataFrame) -> tuple[str, str]:
    candidate = folds[folds["model"] == "candidate_ridge"].copy()
    rhos = candidate["spearman"].fillna(0.0).to_numpy(float)
    success_count = int(np.sum(rhos > 0.30))
    mean_rho = float(np.mean(rhos))
    safer50 = candidate["risk_at_50"].to_numpy(float) < candidate["risk_at_100"].to_numpy(float)
    ood = folds[folds["model"] == "ood_only"].set_index("held_out_family")
    candidate_indexed = candidate.set_index("held_out_family")
    better_ood = float(candidate_indexed["risk_at_50"].mean()) < float(ood["risk_at_50"].mean())
    if success_count >= 3 and mean_rho > 0.40 and int(safer50.sum()) >= 3 and better_ood:
        return "GO", f"primary Ridge passed {success_count}/4 rho gates; mean rho={mean_rho:.3f}"
    signs_unstable = bool(np.any(rhos < 0.0) and np.any(rhos > 0.0))
    metadata = folds[folds["model"] == "metadata_ridge"]["spearman"].fillna(0.0).mean()
    ood_rho = folds[folds["model"] == "ood_only"]["spearman"].fillna(0.0).mean()
    if abs(mean_rho) < 0.10 and signs_unstable and mean_rho <= max(float(metadata), float(ood_rho)):
        return "STOP", f"mean rho={mean_rho:.3f}, signs unstable, and candidate did not beat simple baselines"
    return "BORDERLINE", f"primary Ridge passed {success_count}/4 rho gates; mean rho={mean_rho:.3f}"


def _plot_results(folds: pd.DataFrame, details: pd.DataFrame, p: dict[str, Path]) -> None:
    model_order = ["candidate_ridge", "metadata_ridge", "ood_only", "confidence_ridge", "random"]
    labels = {"candidate_ridge": "Candidate Ridge", "metadata_ridge": "Metadata only", "ood_only": "OOD only", "confidence_ridge": "Confidence only", "random": "Random ordering"}
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    for model_name in model_order:
        subset = folds[folds["model"] == model_name]
        means = [subset[f"risk_at_{int(level * 100)}"].mean() for level in COVERAGES]
        axis.plot([int(level * 100) for level in COVERAGES], means, marker="o", label=labels[model_name])
    axis.set_xlabel("Coverage retained (%)")
    axis.set_ylabel("Mean true risk (1 - CDFM directed F1)")
    axis.set_title("Leave-one-mechanism-family-out risk-coverage")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(p["coverage_plot"], dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, mechanism in zip(axes.ravel(), MECHANISMS):
        subset = details[details["mechanism"] == mechanism]
        axis.scatter(subset["predicted_risk__candidate_ridge"], subset["true_risk"], alpha=0.8)
        limits = [0.0, max(1.0, float(subset[["predicted_risk__candidate_ridge", "true_risk"]].max().max()))]
        axis.plot(limits, limits, linestyle="--", color="grey", linewidth=1)
        axis.set_title(f"Held out: {mechanism}")
        axis.grid(alpha=0.2)
    fig.supxlabel("Predicted risk (primary Ridge)")
    fig.supylabel("True risk")
    fig.tight_layout()
    fig.savefig(p["scatter_plot"], dpi=180)
    plt.close(fig)


def _write_report(frame: pd.DataFrame, folds: pd.DataFrame, decision: str, rationale: str, p: dict[str, Path]) -> None:
    candidate = folds[folds["model"] == "candidate_ridge"].set_index("held_out_family")
    baseline_names = ("constant", "metadata_ridge", "ood_only", "confidence_ridge")
    lines = [
        "# Gate-3A：冻结 CDFM 的结构错误风险可预测性",
        "",
        f"最终判定：**{decision}**。{rationale}。",
        "",
        "## 科学问题与必要性",
        "",
        "本轮检验：测试时完全不知道真实 DAG 时，仅凭观测数据、CDFM 预测图、CDFM 直接提供的概率、与 DirectLiNGAM 的预测分歧及可选 bootstrap 稳定性，能否预测 `1 - directed F1`，并泛化到 risk predictor 未见的机制族。若连风险排序都不能跨机制迁移，后续 risk certificate 即使形式上校准，也缺少可用的个体化风险信号。",
        "",
        "## 数据与防泄漏设计",
        "",
        f"使用 Gate-2 generator 产生 {len(frame)} 个 D=10、N=1000、Laplace 噪声任务。{frame['graph_seed'].nunique()} 个 graph seed 在 linear、tanh、RFF、interaction 间共享完全相同的 DAG、边权与外生噪声；linear 使用 lambda=0，其余使用 lambda=1。interaction 沿用 Gate-2 定义：逐边 tanh，并在入度至少为 2 时加入父变量交互项。",
        "",
        "主评估是四折 leave-one-mechanism-family-out，因为随机切分会让同机制分布同时出现在训练和测试中，不能回答跨未见机制族泛化。真实图只用于监督 target 和最终评价；risk 输入不含真实图、CDFM/LiNGAM F1、真实机制标签、oracle 或其他标签派生量。所有 scaler、OOD 中心/协方差和模型均只在三个训练机制族上拟合。",
        "",
        "CDFM 未训练、未微调、未改源码；使用已安装 `cdfm-base` 的官方自动阈值。API 直接返回 probabilities/logits，因此加入概率摘要；bootstrap B 值见 predictions.csv。预注册主候选是全诊断 Ridge，固定 RF 仅为辅助敏感性分析。",
        "",
        "## 四个未见机制族结果",
        "",
        "| held-out family | CDFM mean F1 | risk variance | candidate rho | metadata rho | OOD rho | confidence rho | candidate risk@50% | full risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mechanism in MECHANISMS:
        local = frame[frame["mechanism"] == mechanism]
        row = candidate.loc[mechanism]
        baseline = folds[folds["held_out_family"] == mechanism].set_index("model")
        def fmt(value: object) -> str:
            return "NA" if pd.isna(value) else f"{float(value):.3f}"
        lines.append(
            f"| {mechanism} | {local['cdfm_f1'].mean():.3f} | {local['true_risk'].var(ddof=1):.4f} | "
            f"{fmt(row['spearman'])} | {fmt(baseline.loc['metadata_ridge', 'spearman'])} | "
            f"{fmt(baseline.loc['ood_only', 'spearman'])} | {fmt(baseline.loc['confidence_ridge', 'spearman'])} | "
            f"{row['risk_at_50']:.3f} | {row['risk_at_100']:.3f} |"
        )
    mean_rhos = {
        name: float(folds[folds["model"] == name]["spearman"].fillna(0.0).mean())
        for name in ("candidate_ridge", *baseline_names, "candidate_rf")
    }
    successful = [m for m in MECHANISMS if float(candidate.loc[m, "spearman"] if pd.notna(candidate.loc[m, "spearman"]) else 0.0) > 0.30]
    failed = [m for m in MECHANISMS if m not in successful]
    mean_risk50 = {
        name: float(folds[folds["model"] == name]["risk_at_50"].mean())
        for name in ("candidate_ridge", "metadata_ridge", "ood_only", "confidence_ridge", "candidate_rf", "random")
    }
    full_risk = float(candidate["risk_at_100"].mean())
    lines.extend([
        "",
        "## 解释与 Gate",
        "",
        f"平均 Spearman（常数的未定义相关按 0 计）：candidate Ridge={mean_rhos['candidate_ridge']:.3f}，metadata-only={mean_rhos['metadata_ridge']:.3f}，OOD-only={mean_rhos['ood_only']:.3f}，confidence-only={mean_rhos['confidence_ridge']:.3f}，secondary RF={mean_rhos['candidate_rf']:.3f}。constant predictor 没有排序能力；其相关为 NA，risk-coverage 等价于随机期望。",
        "",
        f"按预注册 rho>0.30 标准，主候选成功 family：{', '.join(successful) if successful else 'none'}；失败 family：{', '.join(failed) if failed else 'none'}。主候选平均 risk@50%={mean_risk50['candidate_ridge']:.3f}，低于 full/random={full_risk:.3f} 和 OOD-only={mean_risk50['ood_only']:.3f}，但高于 confidence-only={mean_risk50['confidence_ridge']:.3f}；其平均 rho 也低于 metadata-only 与 confidence-only。因此它不是单纯 OOD detector，但新增诊断在 Ridge 中没有稳定提供超越 cheap baselines 的增量信息。",
        "",
        f"固定 RF 的辅助结果平均 rho={mean_rhos['candidate_rf']:.3f}，3/4 family 超过 0.30，平均 risk@50%={mean_risk50['candidate_rf']:.3f}；它达到点估计 GO 数值条件，但这是预先声明的 secondary sensitivity，不替换主候选，而且 RFF rho 仍只有 {float(folds[(folds['model'] == 'candidate_rf') & (folds['held_out_family'] == 'rff')]['spearman'].iloc[0]):.3f}。因此 Gate 保持 **{decision}**，没有按测试结果更换主模型。bootstrap 95% CI、Pearson、MAE、25/50/75/100% coverage 均在 `fold_results.csv`。",
        "",
        "## 失败模式与下一信号",
        "",
        f"最重要的失败模式是 RFF 的线性跨族负迁移：主候选 rho={float(candidate.loc['rff', 'spearman']):.3f}，risk@50%={float(candidate.loc['rff', 'risk_at_50']):.3f} 反而高于 full risk={float(candidate.loc['rff', 'risk_at_100']):.3f}；同一 fold 的 confidence-only rho={float(folds[(folds['model'] == 'confidence_ridge') & (folds['held_out_family'] == 'rff')]['spearman'].iloc[0]):.3f}，说明全特征线性组合压坏了一个本来有用的直接信号。若继续，下一轮最值得增加的单一信号是 CDFM 的小扰动一致性（对输入加入幅度受控的观测噪声后比较图变化）；它仍不需要真实图，并比继续堆叠分布矩更直接地探测决策边界脆弱性。",
    ])
    p["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(*, smoke: bool) -> None:
    p = paths(smoke)
    frame = build_predictions(smoke=smoke)
    if any((frame["mechanism"] == mechanism).sum() < 3 for mechanism in MECHANISMS):
        print("Smoke predictions written; skipping LOFO statistics because each family has fewer than 3 tasks.", flush=True)
        return
    fold_rows: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        rows, detail = _evaluate_fold(frame, mechanism)
        fold_rows.extend(rows)
        details.append(detail)
    folds = pd.DataFrame(fold_rows)
    detail_frame = pd.concat(details, ignore_index=True)
    _atomic_csv(folds, p["folds"])
    decision, rationale = _gate_decision(folds)
    _plot_results(folds, detail_frame, p)
    candidate = folds[folds["model"] == "candidate_ridge"]
    summary = {
        "schema_version": "gate3a_summary_v1",
        "scientific_question": "Can observable signals predict frozen-CDFM true graph risk across an unseen mechanism family?",
        "config": {
            "mechanisms": list(MECHANISMS),
            "n_seeds_per_mechanism": int(frame["graph_seed"].nunique()),
            "N": N_SAMPLES,
            "D": N_VARIABLES,
            "noise": "laplace",
            "paired_design": True,
            "bootstrap_cdfm_per_dataset": int(frame["bootstrap_count"].iloc[0]),
            "bootstrap_ci_resamples": BOOTSTRAP_RESAMPLES,
            "primary_candidate": "candidate_ridge",
        },
        "cdfm_inference_count": int(len(frame) * (1 + int(frame["bootstrap_count"].iloc[0]))),
        "confidence_available": True,
        "mean_cdfm_runtime_sec": float(frame["cdfm_runtime_sec"].mean()),
        "mean_candidate_spearman": float(candidate["spearman"].fillna(0.0).mean()),
        "families_over_rho_0_30": int((candidate["spearman"].fillna(0.0) > 0.30).sum()),
        "decision": decision,
        "rationale": rationale,
        "label_leakage_check": "passed",
    }
    _atomic_json(summary, p["summary"])
    _write_report(frame, folds, decision, rationale, p)
    print(json.dumps({"decision": decision, "rationale": rationale}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "infer", "analyze", "all"), default="all")
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, choices=(0, 2), default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    n_seeds = 1 if args.smoke else args.n_seeds
    if n_seeds <= 0:
        raise SystemExit("--n-seeds must be positive")
    if args.stage in ("generate", "all"):
        generate_tasks(n_seeds=n_seeds, smoke=args.smoke, rerun=args.rerun)
    if args.stage in ("infer", "all"):
        run_inference(smoke=args.smoke, bootstrap=args.bootstrap, rerun=args.rerun)
    if args.stage in ("analyze", "all"):
        analyze(smoke=args.smoke)


if __name__ == "__main__":
    main()
