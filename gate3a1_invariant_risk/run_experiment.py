"""Gate-3A.1: frozen, graph-instance-independent CDFM risk confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from itertools import combinations
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE0_ROOT = REPO_ROOT / "gate0_cdfm_defer"
for import_root in (REPO_ROOT, GATE0_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gate2_reliable_routing.generate_tasks import _graph, generate_x  # noqa: E402
from metrics import directed_graph_metrics, lingam_target_source_to_source_target  # noqa: E402
from meta_features import FEATURE_COLUMNS as METADATA_FEATURES  # noqa: E402
from meta_features import extract_meta_features  # noqa: E402


OLD_ROOT = REPO_ROOT / "gate3a_risk_predictability"
OLD_PREDICTIONS = OLD_ROOT / "results" / "predictions.csv"
OLD_RAW = OLD_ROOT / "cache" / "raw_predictions"
PROTOCOL_PATH = ROOT / "frozen_protocol.json"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASETS = CACHE / "datasets"
RAW = CACHE / "raw_predictions"

ANALYSIS_SEED = 20260918
BOOTSTRAP_RESAMPLES = 1000
COVERAGES = (0.25, 0.50, 0.75, 1.00)
TEST_SPECS = {
    "softsign": range(320000, 320025),
    "sine": range(321000, 321025),
}

PRIMARY_FEATURES = [
    "cdfm_edge_density",
    "cdfm_max_degree",
    "cdfm_mean_entropy",
    "cdfm_bootstrap_disagreement",
    "cdfm_bootstrap_edge_jaccard",
]
THRESHOLD_FEATURES = [
    "cdfm_auto_threshold",
    "cdfm_threshold_margin_mean",
    "cdfm_threshold_margin_p10",
    "cdfm_threshold_margin_p25",
    "cdfm_threshold_near_frac_002",
    "cdfm_threshold_near_frac_005",
]
SECONDARY_FEATURES = [*PRIMARY_FEATURES, *THRESHOLD_FEATURES]
GRAPH_FEATURES = ["cdfm_edge_count", "cdfm_edge_density", "cdfm_mean_degree", "cdfm_max_degree"]
DISAGREEMENT_FEATURES = ["cdfm_lingam_adj_disagreement"]
CONFIDENCE_FEATURES = [
    "cdfm_mean_edge_probability",
    "cdfm_mean_confidence",
    "cdfm_mean_entropy",
    "cdfm_mean_margin",
]
STABILITY_FEATURES = ["cdfm_bootstrap_disagreement", "cdfm_bootstrap_edge_jaccard"]
OLD_CANDIDATE_BASE_FEATURES = [
    *METADATA_FEATURES,
    *GRAPH_FEATURES,
    *DISAGREEMENT_FEATURES,
    *CONFIDENCE_FEATURES,
    *STABILITY_FEATURES,
]
MODEL_ORDER = [
    "stable_ridge_v1",
    "confidence_ridge",
    "metadata_ridge",
    "ood_only",
    "old_candidate_ridge",
    "threshold_aware_ridge",
    "constant",
]


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, destination)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _append_error(task_id: str, stage: str, exc: BaseException) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "task_id": task_id,
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "timestamp_epoch": time.time(),
        }, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["frozen_before_confirmatory_truth_evaluation"] is True
    assert protocol["primary"]["features"] == PRIMARY_FEATURES
    assert protocol["secondary_threshold_aware"]["features"] == SECONDARY_FEATURES
    assert protocol["training"]["task_count"] == 60
    assert protocol["confirmatory_test"]["task_count"] == 50
    return protocol


def _verify_audited_sources(protocol: dict[str, object]) -> None:
    for relative, expected in protocol["audited_source_sha256"].items():
        actual = _sha256(REPO_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"audited Gate-3A source changed: {relative}: {actual} != {expected}")


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.asarray(matrix)[mask]


def _edge_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    union = np.logical_or(left_bool, right_bool).sum()
    return float(np.logical_and(left_bool, right_bool).sum() / union) if union else 1.0


def threshold_features(probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(_off_diagonal(probabilities), dtype=float)
    margins = np.abs(values - float(threshold))
    return {
        "cdfm_auto_threshold": float(threshold),
        "cdfm_threshold_margin_mean": float(np.mean(margins)),
        "cdfm_threshold_margin_p10": float(np.quantile(margins, 0.10)),
        "cdfm_threshold_margin_p25": float(np.quantile(margins, 0.25)),
        "cdfm_threshold_near_frac_002": float(np.mean(margins <= 0.02)),
        "cdfm_threshold_near_frac_005": float(np.mean(margins <= 0.05)),
    }


def diagnostic_features(
    x: np.ndarray,
    cdfm_adjacency: np.ndarray,
    probabilities: np.ndarray,
    lingam_adjacency: np.ndarray,
    bootstrap_adjacencies: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Return ground-truth-free diagnostics; deliberately accepts no truth graph."""
    features = dict(extract_meta_features(x))
    d = int(cdfm_adjacency.shape[0])
    degrees = cdfm_adjacency.sum(axis=0) + cdfm_adjacency.sum(axis=1)
    clipped = np.clip(_off_diagonal(probabilities), 1e-12, 1.0 - 1e-12)
    features.update({
        "cdfm_edge_count": float(cdfm_adjacency.sum()),
        "cdfm_edge_density": float(cdfm_adjacency.sum() / (d * (d - 1))),
        "cdfm_mean_degree": float(degrees.mean()),
        "cdfm_max_degree": float(degrees.max()),
        "cdfm_lingam_adj_disagreement": float(np.mean(
            _off_diagonal(cdfm_adjacency) != _off_diagonal(lingam_adjacency)
        )),
        "cdfm_mean_edge_probability": float(clipped.mean()),
        "cdfm_mean_confidence": float(np.mean(np.maximum(clipped, 1.0 - clipped))),
        "cdfm_mean_entropy": float(np.mean(
            -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)
        )),
        "cdfm_mean_margin": float(np.mean(np.abs(clipped - 0.5))),
        "cdfm_bootstrap_disagreement": float(np.mean([
            np.mean(_off_diagonal(cdfm_adjacency) != _off_diagonal(graph))
            for graph in bootstrap_adjacencies
        ])),
        "cdfm_bootstrap_edge_jaccard": float(np.mean([
            _edge_jaccard(cdfm_adjacency, graph) for graph in bootstrap_adjacencies
        ])),
    })
    features.update(threshold_features(probabilities, threshold))
    return features


def _old_cache_audit() -> pd.DataFrame:
    old = pd.read_csv(OLD_PREDICTIONS)
    if len(old) != 60 or set(old["mechanism"]) != {"linear", "tanh", "rff", "interaction"}:
        raise RuntimeError("Gate-3A predictions do not contain the frozen 60-task training set")
    rows: list[dict[str, object]] = []
    for record in old.to_dict(orient="records"):
        path = OLD_RAW / f"{record['task_id']}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing old Gate-3A cache: {path}")
        with np.load(path) as raw:
            required = {"cdfm_probabilities", "cdfm_threshold", "bootstrap_adjacencies", "bootstrap_count"}
            missing = required.difference(raw.files)
            if missing:
                raise RuntimeError(f"old cache {path.name} lacks {sorted(missing)}")
            if int(raw["bootstrap_count"]) != 2:
                raise RuntimeError(f"old cache {path.name} does not have bootstrap B=2")
        rows.append({
            "task_id": record["task_id"],
            "graph_seed": int(record["graph_seed"]),
            "mechanism": record["mechanism"],
            "relative_path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        })
    return pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)


def _verify_old_caches_unchanged() -> None:
    recorded_path = RESULTS / "old_cache_audit.csv"
    if not recorded_path.exists():
        raise FileNotFoundError("old_cache_audit.csv is missing; run prepare first")
    recorded = pd.read_csv(recorded_path)
    current = _old_cache_audit()
    if not recorded[["task_id", "sha256"]].equals(current[["task_id", "sha256"]]):
        raise RuntimeError("one or more old Gate-3A caches changed after the prepare audit")


def _augment_old_training() -> pd.DataFrame:
    frame = pd.read_csv(OLD_PREDICTIONS).sort_values(["mechanism", "graph_seed"]).reset_index(drop=True)
    threshold_rows: list[dict[str, float]] = []
    for task_id in frame["task_id"]:
        with np.load(OLD_RAW / f"{task_id}.npz") as raw:
            threshold_rows.append(threshold_features(
                np.asarray(raw["cdfm_probabilities"], dtype=float),
                float(raw["cdfm_threshold"]),
            ))
    return pd.concat([frame, pd.DataFrame(threshold_rows)], axis=1)


def _ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    result = float(function(left, right).statistic)
    return result if np.isfinite(result) else None


def _coverage_risks(predicted: np.ndarray, true_risk: np.ndarray) -> dict[float, float]:
    order = np.argsort(predicted, kind="stable")
    return {
        coverage: float(np.mean(true_risk[order[: max(1, math.ceil(coverage * len(order)))]]))
        for coverage in COVERAGES
    }


def _development_audit(training: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for held_out in ("linear", "tanh", "rff", "interaction"):
        train = training[training["mechanism"] != held_out]
        test = training[training["mechanism"] == held_out]
        model = _ridge().fit(train[PRIMARY_FEATURES], train["true_risk"])
        predicted = model.predict(test[PRIMARY_FEATURES])
        risk = test["true_risk"].to_numpy(float)
        coverage = _coverage_risks(predicted, risk)
        rows.append({
            "evidence_status": "post_hoc_exploratory_only",
            "held_out_family": held_out,
            "model": "stable_ridge_v1",
            "n_train": len(train),
            "n_test": len(test),
            "spearman": _safe_correlation(spearmanr, predicted, risk),
            "pearson": _safe_correlation(pearsonr, predicted, risk),
            "mae": float(mean_absolute_error(risk, predicted)),
            "risk_at_25": coverage[0.25],
            "risk_at_50": coverage[0.50],
            "risk_at_75": coverage[0.75],
            "risk_at_100": coverage[1.00],
        })
    return pd.DataFrame(rows)


def _same_seed_risk_correlations(training: pd.DataFrame) -> list[dict[str, object]]:
    pivot = training.pivot(index="graph_seed", columns="mechanism", values="true_risk")
    rows = []
    for left, right in combinations(sorted(pivot.columns), 2):
        rows.append({
            "left_family": left,
            "right_family": right,
            "n_paired_graph_seeds": len(pivot),
            "spearman": _safe_correlation(spearmanr, pivot[left].to_numpy(), pivot[right].to_numpy()),
        })
    return rows


def _generate_new_tasks() -> pd.DataFrame:
    DATASETS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for mechanism, graph_seeds in TEST_SPECS.items():
        for index, graph_seed in enumerate(graph_seeds):
            task_id = f"{mechanism}_seed_{graph_seed}"
            path = DATASETS / f"{task_id}.npz"
            if not path.exists():
                spec = _graph(graph_seed)
                x = generate_x(spec, mechanism, "laplace", 1.0, graph_seed)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        X=x,
                        truth_adjacency=spec.adjacency,
                        graph_seed=np.int64(graph_seed),
                        mechanism=np.asarray(mechanism),
                    )
                os.replace(temporary, path)
            rows.append({
                "task_id": task_id,
                "family": mechanism,
                "family_index": index,
                "graph_seed": graph_seed,
                "N": 1000,
                "D": 10,
                "noise": "laplace",
                "lambda": 1.0,
                "dataset_path": path.relative_to(ROOT).as_posix(),
            })
    frame = pd.DataFrame(rows)
    if frame["graph_seed"].duplicated().any() or set(frame["graph_seed"]).intersection(range(310000, 310015)):
        raise AssertionError("confirmatory graph seeds are not independent from one another and Gate-3A")
    _atomic_csv(frame, CACHE / "tasks.csv")
    return frame


def prepare() -> None:
    protocol = _protocol()
    _verify_audited_sources(protocol)
    RESULTS.mkdir(parents=True, exist_ok=True)
    cache_audit = _old_cache_audit()
    _atomic_csv(cache_audit, RESULTS / "old_cache_audit.csv")
    training = _augment_old_training()
    _atomic_csv(_development_audit(training), RESULTS / "development_audit.csv")
    _atomic_csv(pd.DataFrame(_same_seed_risk_correlations(training)), RESULTS / "same_seed_risk_correlations.csv")
    tasks = _generate_new_tasks()
    _atomic_json({
        "schema_version": "gate3a1_prepare_receipt_v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "old_predictions_sha256": _sha256(OLD_PREDICTIONS),
        "old_cache_count": len(cache_audit),
        "old_cdfm_inference_count": 0,
        "new_task_count": len(tasks),
        "new_seed_ranges": {key: [min(value), max(value)] for key, value in TEST_SPECS.items()},
        "primary_frozen": True,
    }, RESULTS / "prepare_receipt.json")
    print(f"Prepared: old caches audited={len(cache_audit)}, new tasks={len(tasks)}", flush=True)


def _bootstrap_graphs(model: CDFM, x: np.ndarray, graph_seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED, graph_seed, 77]))
    graphs = []
    start = time.perf_counter()
    for _ in range(2):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        graphs.append(np.asarray(model.predict(x[indices]).adjacency, dtype=np.int8))
    return np.stack(graphs), float(time.perf_counter() - start)


def _run_lingam(x: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    model = lingam.DirectLiNGAM(measure="pwling")
    model.fit(x)
    graph = lingam_target_source_to_source_target(np.asarray(model.adjacency_matrix_, dtype=float))
    return graph, float(time.perf_counter() - start)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def infer() -> None:
    _protocol()
    _verify_old_caches_unchanged()
    tasks_path = CACHE / "tasks.csv"
    if not tasks_path.exists():
        raise FileNotFoundError("tasks.csv missing; run prepare first")
    tasks = pd.read_csv(tasks_path)
    RAW.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _atomic_json({
        "schema_version": "gate3a1_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {name: _package_version(name) for name in (
            "cdfm-base", "causal-learn", "numpy", "pandas", "scipy", "scikit-learn", "torch"
        )},
        "cdfm_info": model.info,
        "cdfm_frozen": True,
        "bootstrap": 2,
    }, RESULTS / "environment.json")
    for ordinal, row in enumerate(tasks.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = RAW / f"{task_id}.npz"
        if destination.exists():
            with np.load(destination) as cached:
                if int(cached["bootstrap_count"]) != 2:
                    raise RuntimeError(f"unexpected cached bootstrap count for {task_id}")
            print(f"[{ordinal}/{len(tasks)}] skip cached {task_id}", flush=True)
            continue
        try:
            with np.load(ROOT / str(row["dataset_path"])) as dataset:
                x = np.asarray(dataset["X"], dtype=np.float64)
            start = time.perf_counter()
            result = model.predict(x)
            cdfm_runtime = float(time.perf_counter() - start)
            bootstrap_adjacencies, bootstrap_runtime = _bootstrap_graphs(model, x, int(row["graph_seed"]))
            lingam_adjacency, lingam_runtime = _run_lingam(x)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cdfm_adjacency=np.asarray(result.adjacency, dtype=np.int8),
                    cdfm_probabilities=np.asarray(result.probabilities, dtype=float),
                    cdfm_logits=np.asarray(result.logits, dtype=float),
                    cdfm_threshold=np.float64(result.threshold),
                    cdfm_runtime_sec=np.float64(cdfm_runtime),
                    bootstrap_adjacencies=bootstrap_adjacencies,
                    bootstrap_count=np.int64(2),
                    bootstrap_runtime_sec=np.float64(bootstrap_runtime),
                    lingam_adjacency_source_target=np.asarray(lingam_adjacency, dtype=np.int8),
                    lingam_runtime_sec=np.float64(lingam_runtime),
                )
            os.replace(temporary, destination)
            print(
                f"[{ordinal}/{len(tasks)}] {task_id}: CDFM={cdfm_runtime:.2f}s, "
                f"bootstrap={bootstrap_runtime:.2f}s, LiNGAM={lingam_runtime:.2f}s",
                flush=True,
            )
        except Exception as exc:
            _append_error(task_id, "inference", exc)
            print(f"[{ordinal}/{len(tasks)}] ERROR {type(exc).__name__}: {exc}", flush=True)
    completed = sum((RAW / f"{task_id}.npz").exists() for task_id in tasks["task_id"])
    _verify_old_caches_unchanged()
    if completed != len(tasks):
        raise RuntimeError(f"new inference incomplete: {completed}/{len(tasks)}")
    print(f"Inference complete: {completed}/{len(tasks)} new tasks cached; old tasks rerun=0", flush=True)


def _build_confirmatory_features() -> pd.DataFrame:
    tasks = pd.read_csv(CACHE / "tasks.csv")
    rows: list[dict[str, object]] = []
    for row in tasks.to_dict(orient="records"):
        task_id = str(row["task_id"])
        with np.load(ROOT / str(row["dataset_path"])) as dataset:
            x = np.asarray(dataset["X"], dtype=float)
        with np.load(RAW / f"{task_id}.npz") as raw:
            diagnostics = diagnostic_features(
                x,
                np.asarray(raw["cdfm_adjacency"], dtype=np.int8),
                np.asarray(raw["cdfm_probabilities"], dtype=float),
                np.asarray(raw["lingam_adjacency_source_target"], dtype=np.int8),
                np.asarray(raw["bootstrap_adjacencies"], dtype=np.int8),
                float(raw["cdfm_threshold"]),
            )
            timings = {
                "cdfm_runtime_sec": float(raw["cdfm_runtime_sec"]),
                "bootstrap_runtime_sec": float(raw["bootstrap_runtime_sec"]),
                "lingam_runtime_sec": float(raw["lingam_runtime_sec"]),
            }
        rows.append({
            "task_id": task_id,
            "graph_seed": int(row["graph_seed"]),
            "mechanism": str(row["family"]),
            **diagnostics,
            **timings,
        })
    return pd.DataFrame(rows).sort_values(["mechanism", "graph_seed"]).reset_index(drop=True)


def _fit_ood(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[METADATA_FEATURES].to_numpy(float)
    test_x = test[METADATA_FEATURES].to_numpy(float)
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


def _fit_frozen_models(training: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    train = training.copy()
    output = test[["task_id", "graph_seed", "mechanism"]].copy()
    ood_train, ood_test = _fit_ood(train, test)
    train["ood_score"] = ood_train
    old_candidate_features = [*OLD_CANDIDATE_BASE_FEATURES, "ood_score"]
    y = train["true_risk"].to_numpy(float)
    specifications = {
        "stable_ridge_v1": PRIMARY_FEATURES,
        "confidence_ridge": CONFIDENCE_FEATURES,
        "metadata_ridge": list(METADATA_FEATURES),
        "threshold_aware_ridge": SECONDARY_FEATURES,
    }
    for name, features in specifications.items():
        model = _ridge().fit(train[features], y)
        output[f"predicted_risk__{name}"] = model.predict(test[features])
    ood_model = _ridge().fit(ood_train.reshape(-1, 1), y)
    output["predicted_risk__ood_only"] = ood_model.predict(ood_test.reshape(-1, 1))
    test_with_ood = test.copy()
    test_with_ood["ood_score"] = ood_test
    old_model = _ridge().fit(train[old_candidate_features], y)
    output["predicted_risk__old_candidate_ridge"] = old_model.predict(test_with_ood[old_candidate_features])
    output["predicted_risk__constant"] = float(np.mean(y))
    _atomic_json({
        "schema_version": "gate3a1_model_fit_receipt_v1",
        "written_before_confirmatory_truth_was_loaded": True,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "training_predictions_sha256": _sha256(OLD_PREDICTIONS),
        "confirmatory_feature_table_sha256": _sha256(RESULTS / "confirmatory_features_ground_truth_free.csv"),
        "n_train": len(train),
        "n_test": len(test),
        "scalers_fit_on_training_only": True,
        "alpha": 1.0,
        "primary_features": PRIMARY_FEATURES,
        "secondary_features": SECONDARY_FEATURES,
        "test_results_used_for_model_selection": False,
    }, RESULTS / "model_fit_receipt.json")
    return output


def _load_confirmatory_truth(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    tasks = pd.read_csv(CACHE / "tasks.csv").set_index("task_id")
    truth_rows = []
    for task_id in predictions["task_id"]:
        task = tasks.loc[task_id]
        with np.load(ROOT / str(task["dataset_path"])) as dataset:
            truth = np.asarray(dataset["truth_adjacency"], dtype=np.int8)
        with np.load(RAW / f"{task_id}.npz") as raw:
            cdfm = np.asarray(raw["cdfm_adjacency"], dtype=np.int8)
        metrics = directed_graph_metrics(cdfm, truth)
        truth_rows.append({
            "task_id": task_id,
            "cdfm_f1": float(metrics["f1"]),
            "cdfm_shd": int(metrics["shd"]),
            "true_risk": 1.0 - float(metrics["f1"]),
        })
    output = predictions.merge(pd.DataFrame(truth_rows), on="task_id", validate="one_to_one")
    feature_columns = [column for column in features if column not in {"graph_seed", "mechanism"}]
    return output.merge(features[feature_columns], on="task_id", validate="one_to_one")


def _bootstrap_ci(
    predicted: np.ndarray,
    risk: np.ndarray,
    metric: str,
    seed: int,
    constant: bool = False,
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(risk), size=len(risk))
        if metric == "spearman":
            value = _safe_correlation(spearmanr, predicted[indices], risk[indices])
            if value is not None:
                estimates.append(value)
        elif constant:
            estimates.append(float(np.mean(risk[indices])))
        else:
            estimates.append(_coverage_risks(predicted[indices], risk[indices])[0.50])
    if not estimates:
        return None, None
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(("softsign", "sine")):
        local = predictions[predictions["mechanism"] == family].reset_index(drop=True)
        risk = local["true_risk"].to_numpy(float)
        full_risk = float(np.mean(risk))
        for model_index, name in enumerate(MODEL_ORDER):
            predicted = local[f"predicted_risk__{name}"].to_numpy(float)
            is_constant = name == "constant"
            coverage = (
                {level: full_risk for level in COVERAGES}
                if is_constant else _coverage_risks(predicted, risk)
            )
            rho_low, rho_high = _bootstrap_ci(
                predicted, risk, "spearman", ANALYSIS_SEED + 100 * family_index + model_index, is_constant
            )
            risk_low, risk_high = _bootstrap_ci(
                predicted, risk, "risk50", ANALYSIS_SEED + 1000 + 100 * family_index + model_index, is_constant
            )
            rows.append({
                "family": family,
                "model": name,
                "n_train": 60,
                "n_test": len(local),
                "spearman": _safe_correlation(spearmanr, predicted, risk),
                "spearman_ci_low": rho_low,
                "spearman_ci_high": rho_high,
                "pearson": _safe_correlation(pearsonr, predicted, risk),
                "mae": float(mean_absolute_error(risk, predicted)),
                "risk_at_25": coverage[0.25],
                "risk_at_50": coverage[0.50],
                "risk_at_50_ci_low": risk_low,
                "risk_at_50_ci_high": risk_high,
                "risk_at_75": coverage[0.75],
                "risk_at_100": coverage[1.00],
                "relative_risk_reduction_at_50": (
                    float((full_risk - coverage[0.50]) / full_risk) if full_risk > 0 else 0.0
                ),
            })
    return pd.DataFrame(rows)


def _gate_decision(results: pd.DataFrame) -> tuple[str, str]:
    primary = results[results["model"] == "stable_ridge_v1"].set_index("family")
    rhos = primary.loc[["softsign", "sine"], "spearman"].fillna(0.0).to_numpy(float)
    mean_rho = float(np.mean(rhos))
    both_rho = bool(np.all(rhos > 0.30))
    both_safer = bool(np.all(primary["risk_at_50"].to_numpy(float) < primary["risk_at_100"].to_numpy(float)))
    mean_reduction = float(primary["relative_risk_reduction_at_50"].mean())
    if both_rho and mean_rho > 0.40 and both_safer and mean_reduction >= 0.15:
        return "GO", "all frozen GO conditions passed"
    if mean_rho < 0.25 or not both_safer:
        return "STOP", f"STOP rule applied: mean rho={mean_rho:.3f}, both risk@50 improvements={both_safer}"
    one_over = int(np.sum(rhos > 0.30)) == 1
    if ((0.25 <= mean_rho <= 0.40) or one_over) and both_safer:
        return "BORDERLINE", f"confirmation retained signal but missed GO: mean rho={mean_rho:.3f}"
    return "BORDERLINE", f"GO missed without a STOP condition: mean rho={mean_rho:.3f}"


def _plot(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    labels = {
        "stable_ridge_v1": "stable_ridge_v1",
        "confidence_ridge": "confidence-only",
        "threshold_aware_ridge": "threshold-aware",
        "constant": "constant/full-risk",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for axis, family in zip(axes, ("softsign", "sine")):
        subset = results[results["family"] == family].set_index("model")
        for model in labels:
            row = subset.loc[model]
            axis.plot(
                [25, 50, 75, 100],
                [row["risk_at_25"], row["risk_at_50"], row["risk_at_75"], row["risk_at_100"]],
                marker="o",
                label=labels[model],
            )
        axis.set_title(family)
        axis.set_xlabel("Coverage retained (%)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean true risk (1 - directed F1)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Independent confirmatory risk-coverage")
    fig.tight_layout()
    fig.savefig(RESULTS / "risk_coverage_confirmatory.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharex=True, sharey=True)
    for axis, family in zip(axes, ("softsign", "sine")):
        subset = predictions[predictions["mechanism"] == family]
        axis.scatter(subset["predicted_risk__stable_ridge_v1"], subset["true_risk"], alpha=0.8)
        lower = min(0.0, float(subset["predicted_risk__stable_ridge_v1"].min()))
        upper = max(1.0, float(subset[["predicted_risk__stable_ridge_v1", "true_risk"]].max().max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="grey", linewidth=1)
        axis.set_title(family)
        axis.grid(alpha=0.25)
    fig.supxlabel("Predicted risk: stable_ridge_v1")
    fig.supylabel("True risk")
    fig.tight_layout()
    fig.savefig(RESULTS / "predicted_vs_true_confirmatory.png", dpi=180)
    plt.close(fig)


def _fmt(value: object) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def _write_report(
    development: pd.DataFrame,
    correlations: pd.DataFrame,
    confirmatory: pd.DataFrame,
    decision: str,
    rationale: str,
) -> None:
    primary = confirmatory[confirmatory["model"] == "stable_ridge_v1"].set_index("family")
    confidence = confirmatory[confirmatory["model"] == "confidence_ridge"].set_index("family")
    threshold = confirmatory[confirmatory["model"] == "threshold_aware_ridge"].set_index("family")
    mean_rho = float(primary["spearman"].fillna(0.0).mean())
    mean_reduction = float(primary["relative_risk_reduction_at_50"].mean())
    tanh_interaction = correlations[
        ((correlations["left_family"] == "interaction") & (correlations["right_family"] == "tanh"))
        | ((correlations["left_family"] == "tanh") & (correlations["right_family"] == "interaction"))
    ].iloc[0]
    lines = [
        "# Gate-3A.1：跨机制、跨图实例的 invariant-risk 独立确认",
        "",
        f"最终判定：**{decision}**。{rationale}。",
        "",
        "## Gate-3A 审计边界",
        "",
        "Gate-3A 有三个需要显式修正但不应称为数据泄漏的问题：",
        "",
        "1. 固定 D=10 时，`cdfm_edge_count`、`cdfm_edge_density`、`cdfm_mean_degree` 是线性重复信息。",
        "2. `cdfm_mean_confidence = 0.5 + cdfm_mean_margin`，二者也是线性重复信息。",
        "3. Gate-3A 的 train/test mechanism folds 共享相同 graph seeds；它是 mechanism holdout，但存在 train/test group dependence / repeated underlying graph instances，不是 graph-instance-independent confirmation。",
        "",
        "这些问题影响模型设定与独立性解释，但这里不将它们描述为数据泄漏。",
        "",
        "## 1. Post-hoc development evidence",
        "",
        "主候选 `stable_ridge_v1` 使用固定五特征与 `StandardScaler + Ridge(alpha=1.0)`，在旧 Gate-3A cache 上重放四个 LOFO folds。旧 60 个 CDFM tasks 没有重新 inference。",
        "",
        "这些 feature 是在观察 Gate-3A 后确定的，因此旧 60 tasks 上的结果是 post-hoc exploratory，不得作为最终验证。",
        "",
        "| held-out family | Spearman | risk@50 | full risk |",
        "|---|---:|---:|---:|",
    ]
    for row in development.to_dict(orient="records"):
        lines.append(
            f"| {row['held_out_family']} | {_fmt(row['spearman'])} | "
            f"{row['risk_at_50']:.3f} | {row['risk_at_100']:.3f} |"
        )
    lines.extend([
        "",
        "### 相同 graph seed 的旧风险相关 sanity check",
        "",
        f"四个机制共享 15 个底层 graph seeds。所有 pairwise Spearman 见 `same_seed_risk_correlations.csv`；尤其 tanh vs interaction 为 **{float(tanh_interaction['spearman']):.3f}**。该分析只量化旧实验的 graph-instance dependence，不进入任何 predictor feature。",
        "",
        "### Threshold-aware diagnostics",
        "",
        "原 Gate-3A 的 margin 以 0.5 为中心，而 CDFM 使用 official auto threshold。本轮从旧 raw cache 与新 raw cache 的 `cdfm_probabilities`、`cdfm_threshold` 计算 threshold、margin mean/p10/p25 与 near-fraction(0.02/0.05)。它们只进入预先冻结的 secondary Ridge；没有替换 primary。",
        "",
        "## 2. Pre-registered independent confirmation",
        "",
        "训练集仅为旧 Gate-3A 的 linear/tanh/RFF/interaction 60 tasks。测试集为 25 个 softsign（320000–320024）和 25 个 sine（321000–321024），均为 Laplace、D=10、N=1000；两个 family 不共享 seed，也不与旧 310000–310014 重叠。因此测试同时是 unseen mechanism family 与 unseen graph instances。CDFM 保持冻结；每个新任务运行一次原始 inference 和 B=2 bootstrap。",
        "",
        "所有 scaler、OOD 统计量和 Ridge 均只 fit 旧 60 tasks。`model_fit_receipt.json` 在加载 confirmatory truth 前写出。没有 hyperparameter search。",
        "",
        "### Primary stable_ridge_v1",
        "",
        "| family | rho (95% bootstrap CI) | Pearson | MAE | risk@50 (95% CI) | full risk | relative reduction@50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for family in ("softsign", "sine"):
        row = primary.loc[family]
        lines.append(
            f"| {family} | {_fmt(row['spearman'])} [{_fmt(row['spearman_ci_low'])}, {_fmt(row['spearman_ci_high'])}] | "
            f"{_fmt(row['pearson'])} | {row['mae']:.3f} | {row['risk_at_50']:.3f} "
            f"[{row['risk_at_50_ci_low']:.3f}, {row['risk_at_50_ci_high']:.3f}] | "
            f"{row['risk_at_100']:.3f} | {100 * row['relative_risk_reduction_at_50']:.1f}% |"
        )
    lines.extend([
        "",
        f"Primary mean rho={mean_rho:.3f}；平均 relative risk reduction@50={100 * mean_reduction:.1f}%。25/50/75/100% coverage 的完整结果见 `confirmatory_results.csv`。",
        "",
        "### Frozen baselines and secondary",
        "",
        "| family | confidence-only rho / risk@50 | threshold-aware rho / risk@50 |",
        "|---|---:|---:|",
    ])
    for family in ("softsign", "sine"):
        c_row = confidence.loc[family]
        t_row = threshold.loc[family]
        lines.append(
            f"| {family} | {_fmt(c_row['spearman'])} / {c_row['risk_at_50']:.3f} | "
            f"{_fmt(t_row['spearman'])} / {t_row['risk_at_50']:.3f} |"
        )
    primary_mean_rho = float(primary["spearman"].fillna(0.0).mean())
    confidence_mean_rho = float(confidence["spearman"].fillna(0.0).mean())
    confidence_mean_r50 = float(confidence["risk_at_50"].mean())
    primary_mean_r50 = float(primary["risk_at_50"].mean())
    caveat = ""
    if decision == "GO" and (confidence_mean_rho > primary_mean_rho + 0.05 or confidence_mean_r50 < primary_mean_r50 - 0.01):
        caveat = " 风险信号成立，但 composite diagnostics 的增量价值仍 BORDERLINE。"
    lines.extend([
        "",
        f"confidence-only 平均 rho={confidence_mean_rho:.3f}、平均 risk@50={confidence_mean_r50:.3f}；primary 平均 risk@50={primary_mean_r50:.3f}。{caveat}",
        "",
        "constant mean-risk、metadata-only、OOD-only、old all-feature candidate Ridge 的完整对照，以及 secondary threshold-aware Ridge，均在 `confirmatory_results.csv`。",
        "",
        "## Gate 与不可变性声明",
        "",
        f"按 `frozen_protocol.json` 的预注册判据，结果为 **{decision}**。测试结果产生后，feature set、model、alpha、threshold 与 normalization 的修改：**No**。",
    ])
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> None:
    _protocol()
    _verify_old_caches_unchanged()
    development = pd.read_csv(RESULTS / "development_audit.csv")
    correlations = pd.read_csv(RESULTS / "same_seed_risk_correlations.csv")
    training = _augment_old_training()
    features = _build_confirmatory_features()
    _atomic_csv(features, RESULTS / "confirmatory_features_ground_truth_free.csv")
    model_predictions = _fit_frozen_models(training, features)
    predictions = _load_confirmatory_truth(model_predictions, features)
    _atomic_csv(predictions, RESULTS / "confirmatory_predictions.csv")
    results = _evaluate(predictions)
    _atomic_csv(results, RESULTS / "confirmatory_results.csv")
    decision, rationale = _gate_decision(results)
    _plot(results, predictions)
    primary = results[results["model"] == "stable_ridge_v1"].set_index("family")
    confidence = results[results["model"] == "confidence_ridge"].set_index("family")
    threshold = results[results["model"] == "threshold_aware_ridge"].set_index("family")
    summary = {
        "schema_version": "gate3a1_summary_v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "primary": {
            family: {
                "spearman": primary.loc[family, "spearman"],
                "risk_at_50": primary.loc[family, "risk_at_50"],
                "full_risk": primary.loc[family, "risk_at_100"],
                "relative_risk_reduction_at_50": primary.loc[family, "relative_risk_reduction_at_50"],
            } for family in ("softsign", "sine")
        },
        "mean_primary_spearman": float(primary["spearman"].fillna(0.0).mean()),
        "mean_primary_relative_risk_reduction_at_50": float(primary["relative_risk_reduction_at_50"].mean()),
        "confidence_only": {
            family: {
                "spearman": confidence.loc[family, "spearman"],
                "risk_at_50": confidence.loc[family, "risk_at_50"],
                "full_risk": confidence.loc[family, "risk_at_100"],
            } for family in ("softsign", "sine")
        },
        "threshold_aware_secondary": {
            family: {
                "spearman": threshold.loc[family, "spearman"],
                "risk_at_50": threshold.loc[family, "risk_at_50"],
                "full_risk": threshold.loc[family, "risk_at_100"],
            } for family in ("softsign", "sine")
        },
        "same_seed_risk_pairwise_spearman": correlations.to_dict(orient="records"),
        "decision": decision,
        "rationale": rationale,
        "old_cdfm_tasks_rerun": 0,
        "new_original_cdfm_inferences": 50,
        "new_bootstrap_cdfm_inferences": 100,
        "post_test_feature_model_hyperparameter_modifications": False,
        "label_leakage_check": "ground-truth absent from feature builder and loaded only after model-fit receipt",
    }
    _atomic_json(summary, RESULTS / "summary.json")
    _write_report(development, correlations, results, decision, rationale)
    _verify_old_caches_unchanged()
    print(json.dumps(_json_ready({
        "decision": decision,
        "mean_rho": summary["mean_primary_spearman"],
        "mean_relative_reduction": summary["mean_primary_relative_risk_reduction_at_50"],
    }), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "analyze", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare()
    if args.stage in {"infer", "all"}:
        infer()
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
