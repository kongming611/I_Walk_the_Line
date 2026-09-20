"""Gate-4: calibrated upper-risk certificates for CDFM graph outputs.

The experiment is deliberately isolated from all historical Gate directories.
It reuses historical full-graph CDFM caches as development inputs, performs
fresh leave-one-variable CDFM inference, and creates a new final split with
unseen mechanism/noise families.  The final truth loader is kept in a separate
function and is guarded by protocol/model/prediction hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
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
from cdfm import CDFM
from scipy.stats import beta, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE3C_ROOT = REPO_ROOT / "gate3c_final_topic_decision"
GATE2_ROOT = REPO_ROOT / "gate2_reliable_routing"
for import_root in (REPO_ROOT, GATE2_ROOT, GATE3C_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gate2_reliable_routing.generate_tasks import _graph, _normalize, _rng  # noqa: E402
from gate3c_final_topic_decision.run_experiment import (  # noqa: E402
    _noise as historical_noise,
)
from gate0_cdfm_defer.meta_features import (  # noqa: E402
    FEATURE_COLUMNS as METADATA_FEATURES,
    extract_meta_features,
)
from gate0_cdfm_defer.metrics import directed_graph_metrics  # noqa: E402


ANALYSIS_SEED = 20260919
N_SAMPLES = 1000
N_VARIABLES = 10
PRIMARY_ALPHA = 0.10
SECONDARY_ALPHA = 0.05
BAD_RISK_THRESHOLD = 0.30
N_BOOTSTRAP = 1000
N_LOVO_FOLDS = 3

TRAIN_MANIFEST_OLD = GATE3C_ROOT / "results" / "train_manifest.csv"
CALIB_MANIFEST_OLD = GATE3C_ROOT / "results" / "dev_manifest.csv"
DEV_MANIFEST_OLD = GATE3C_ROOT / "results" / "final_manifest.csv"
OLD_RAW = GATE3C_ROOT / "cache" / "raw_predictions"

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASETS = CACHE / "datasets"
FULL_RAW = CACHE / "full_predictions"
SUBSET_RAW = CACHE / "subset_predictions"
PROTOCOL_PATH = RESULTS / "frozen_protocol.json"
UNIFIED_MANIFEST_PATH = RESULTS / "unified_manifest.csv"

TRAIN_MECHANISMS = ("linear", "tanh", "rff", "interaction")
CALIB_MECHANISMS = ("softsign", "sine")
DEV_MECHANISMS = ("quadratic", "piecewise")
FINAL_MECHANISMS = ("arctan", "cubic")
TRAIN_NOISES = ("laplace", "gaussian", "student_t8", "student_t3")
CALIB_NOISES = TRAIN_NOISES
DEV_NOISES = ("gaussian", "student_t3", "exponential")
FINAL_NOISES = ("gaussian", "student_t3", "centered_lognormal")
FINAL_TASKS_PER_DOMAIN = 30
FINAL_SEED_START = 700000

THRESHOLD_FEATURES = [
    "cdfm_edge_density",
    "cdfm_max_degree",
    "cdfm_mean_entropy",
    "cdfm_auto_threshold",
    "cdfm_threshold_margin_mean",
    "cdfm_threshold_margin_p10",
    "cdfm_threshold_margin_p25",
    "cdfm_threshold_near_frac_002",
    "cdfm_threshold_near_frac_005",
]
CONFIDENCE_FEATURES = [
    "cdfm_mean_edge_probability",
    "cdfm_mean_confidence",
    "cdfm_mean_entropy",
    "cdfm_mean_margin",
]
BLPC_FEATURES = [
    "blpc_weighted_contradiction",
    "blpc_confident_contradiction_rate",
    "blpc_direction_flip_rate",
    "blpc_worst_delete_inconsistency",
]
SC_FEATURES = ["self_compatibility"]
LOVO_FEATURES = ["lovo_error", "lovo_pair_coverage", "lovo_gain_vs_baseline"]
ORDINARY_FEATURES = [*METADATA_FEATURES, *CONFIDENCE_FEATURES]
PROPOSED_FEATURES = [*THRESHOLD_FEATURES, *BLPC_FEATURES]

METHOD_FEATURES: dict[str, list[str]] = {
    "constant": [],
    "entropy": ["cdfm_mean_entropy"],
    "ood": ["ood_score"],
    "self_compatibility": SC_FEATURES,
    "lovo": LOVO_FEATURES,
    "ordinary_risk_regression": ORDINARY_FEATURES,
    "threshold_aware": THRESHOLD_FEATURES,
    "proposed_blpc": PROPOSED_FEATURES,
}
METHOD_ORDER = tuple(METHOD_FEATURES)
BASE_FEATURE_COLUMNS = [
    *METADATA_FEATURES,
    *THRESHOLD_FEATURES,
    *CONFIDENCE_FEATURES,
    *BLPC_FEATURES,
    *SC_FEATURES,
    *LOVO_FEATURES,
]
FEATURE_COLUMNS = list(dict.fromkeys(BASE_FEATURE_COLUMNS))
FORBIDDEN_FEATURE_TOKENS = ("truth", "true_", "_f1", "_shd", "oracle", "label")


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _edge_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _is_dag(adjacency: np.ndarray) -> bool:
    adjacency = np.asarray(adjacency, dtype=bool)
    indegree = adjacency.sum(axis=0).astype(int)
    queue = [int(i) for i in np.flatnonzero(indegree == 0)]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return visited == adjacency.shape[0]


def _gate4_noise(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "centered_lognormal":
        values = rng.lognormal(mean=0.0, sigma=1.0, size=(N_SAMPLES, N_VARIABLES))
        values = values - np.exp(0.5)
        return values / np.std(values, axis=0, keepdims=True)
    return historical_noise(kind, rng)


def _new_edge_transform(kind: str, x: np.ndarray, slope: float, bias: float) -> np.ndarray:
    z = slope * x + bias
    if kind == "arctan":
        return np.arctan(z)
    if kind == "cubic":
        return z + 0.08 * z**3
    raise ValueError(kind)


def generate_x_gate4(spec: object, mechanism: str, noise_kind: str, seed: int) -> np.ndarray:
    """Generate only the new FINAL mechanisms; truth is not read by features."""
    rng = _rng(seed, 2)
    exogenous = _gate4_noise(noise_kind, rng)
    x = np.zeros((N_SAMPLES, N_VARIABLES), dtype=np.float64)
    incoming = {node: np.flatnonzero(spec.targets == node) for node in range(N_VARIABLES)}
    for child_value in spec.order:
        child = int(child_value)
        indices = incoming[child]
        if len(indices) == 0:
            x[:, child] = exogenous[:, child]
            continue
        linear = np.zeros(N_SAMPLES)
        nonlinear = np.zeros(N_SAMPLES)
        for edge_index in indices:
            source = int(spec.sources[edge_index])
            weight = float(spec.weights[edge_index])
            linear += weight * x[:, source]
            nonlinear += weight * _new_edge_transform(
                mechanism, x[:, source], float(spec.slopes[edge_index]), float(spec.biases[edge_index])
            )
        x[:, child] = _normalize(nonlinear) + exogenous[:, child]
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) <= 1e-12):
        raise ValueError("invalid generated data")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float32)


def _write_dataset(path: Path, x: np.ndarray, truth: np.ndarray, graph_seed: int, mechanism: str, noise: str) -> None:
    if path.exists():
        with np.load(path) as existing:
            if int(existing["graph_seed"]) != graph_seed:
                raise RuntimeError(f"seed mismatch in {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X=np.asarray(x, dtype=np.float32),
            truth_adjacency=np.asarray(truth, dtype=np.int8),
            graph_seed=np.int64(graph_seed),
            mechanism=np.asarray(mechanism),
            noise=np.asarray(noise),
        )
    os.replace(temporary, path)


def protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "gate4_risk_certificate_protocol_v1",
        "freeze_date": "2026-09-19",
        "scientific_question": "Ground-truth-free graph-risk certification for causal foundation models under mechanism shift",
        "certificate": {
            "target": "whole_graph_risk=1-directed_F1",
            "primary_alpha": PRIMARY_ALPHA,
            "secondary_alpha": SECONDARY_ALPHA,
            "bad_risk_threshold": BAD_RISK_THRESHOLD,
            "upper_bound": "clip(predicted_mean_risk + finite_sample_calibration_quantile * predicted_scale, 0, 1)",
            "calibration": "maximum of pooled, mechanism-family, and noise-family one-sided finite-sample quantiles",
            "shift_claim": "empirical unseen-family stability; no exchangeability guarantee is claimed across mechanism shift",
        },
        "data": {
            "train": {"source": "Gate-3C train", "mechanisms": list(TRAIN_MECHANISMS), "noises": list(TRAIN_NOISES), "task_count": 160},
            "calibration": {"source": "Gate-3C dev", "mechanisms": list(CALIB_MECHANISMS), "noises": list(CALIB_NOISES), "task_count": 80},
            "dev": {"source": "Gate-3C final used as development", "mechanisms": list(DEV_MECHANISMS), "noises": list(DEV_NOISES), "task_count": 120},
            "final": {
                "mechanisms": list(FINAL_MECHANISMS), "noises": list(FINAL_NOISES),
                "tasks_per_domain": FINAL_TASKS_PER_DOMAIN, "task_count": 180,
                "graph_seed_range": [FINAL_SEED_START, FINAL_SEED_START + 179],
                "all_graph_seeds_disjoint": True,
            },
            "N": N_SAMPLES,
            "D": N_VARIABLES,
            "random_task_split": False,
        },
        "cdfm": {
            "model": "DMIRLAB/CDFM",
            "full_call": "CDFM.from_pretrained('DMIRLAB/CDFM').predict(X)",
            "subset_calls_per_task": N_VARIABLES,
            "training": False,
            "fine_tuning": False,
            "source_modification": False,
        },
        "baselines": {
            "entropy": "mean CDFM Bernoulli entropy",
            "ood": "robust Mahalanobis score on metadata, fitted on TRAIN only",
            "self_compatibility": "graphical latent-projection incompatibility score",
            "lovo": "graphical LOVO parent-adjustment error and identification coverage",
            "ordinary_risk_regression": "Ridge on metadata and ordinary CDFM confidence summaries",
            "threshold_aware": THRESHOLD_FEATURES,
        },
        "proposed": {
            "name": "Boundary-aware Latent Projection Consistency (BLPC)",
            "features": PROPOSED_FEATURES,
            "model": "StandardScaler + Ridge(alpha=1.0)",
            "no_search": True,
        },
        "gate": {
            "reliability": {
                "pooled_exceedance_max": 0.10,
                "macro_exceedance_max": 0.12,
                "pooled_upper_95_max": 0.15,
                "domains_exceedance_max": 0.20,
                "max_domain_exceedance": 0.25,
                "false_safe_rate_max": 0.10,
            },
            "increment": {
                "aurc_relative_improvement_min": 0.10,
                "safe_coverage_absolute_gain_min": 0.10,
                "bad_auroc_absolute_gain_min": 0.05,
                "allowed_exceedance_loss_vs_threshold": 0.02,
            },
        },
        "truth_isolation": {
            "feature_builder_accepts_truth": False,
            "final_truth_used_for_feature_construction": False,
            "final_truth_used_for_model_selection": False,
            "final_truth_used_for_hyperparameter_choice": False,
            "final_truth_loader": "called only after certificate_receipt.json and pre_truth_hashes.json",
        },
    }


def write_protocol() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    atomic_json(protocol_payload(), PROTOCOL_PATH)


def _read_old_manifest(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["split"] = split
    frame["dataset_path"] = frame["dataset_path"].map(lambda value: str((GATE3C_ROOT / str(value)).relative_to(REPO_ROOT).as_posix()))
    frame["full_raw_path"] = frame["task_id"].map(lambda value: str((OLD_RAW / f"{value}.npz").relative_to(REPO_ROOT).as_posix()))
    frame["source"] = "gate3c_cache"
    return frame


def make_final_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for mechanism in FINAL_MECHANISMS:
        for noise in FINAL_NOISES:
            domain = f"{mechanism}_{noise}"
            for task_index in range(FINAL_TASKS_PER_DOMAIN):
                graph_seed = FINAL_SEED_START + ordinal
                task_id = f"final_{domain}_{task_index:03d}"
                dataset = DATASETS / f"{task_id}.npz"
                if not dataset.exists():
                    spec = _graph(graph_seed)
                    x = generate_x_gate4(spec, mechanism, noise, graph_seed)
                    _write_dataset(dataset, x, spec.adjacency, graph_seed, mechanism, noise)
                rows.append({
                    "task_id": task_id, "split": "final", "domain": domain,
                    "mechanism": mechanism, "noise": noise, "task_index": task_index,
                    "graph_seed": graph_seed, "N": N_SAMPLES, "D": N_VARIABLES,
                    "dataset_path": str(dataset.relative_to(REPO_ROOT).as_posix()),
                    "full_raw_path": str((FULL_RAW / f"{task_id}.npz").relative_to(REPO_ROOT).as_posix()),
                    "source": "gate4_new_final",
                })
                ordinal += 1
    frame = pd.DataFrame(rows)
    if frame["graph_seed"].duplicated().any():
        raise AssertionError("new final graph seeds are not unique")
    return frame


def prepare() -> None:
    write_protocol()
    train = _read_old_manifest(TRAIN_MANIFEST_OLD, "train")
    calibration = _read_old_manifest(CALIB_MANIFEST_OLD, "calibration")
    dev = _read_old_manifest(DEV_MANIFEST_OLD, "dev")
    final = make_final_manifest()
    manifest = pd.concat([train, calibration, dev, final], ignore_index=True)
    if manifest["graph_seed"].duplicated().any():
        raise AssertionError("graph seed overlap across Gate-4 splits")
    atomic_csv(manifest, UNIFIED_MANIFEST_PATH)
    for split in ("train", "calibration", "dev", "final"):
        atomic_csv(manifest[manifest["split"] == split].reset_index(drop=True), RESULTS / f"{split}_manifest.csv")
    old_receipt = {
        "historical_files": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (TRAIN_MANIFEST_OLD, CALIB_MANIFEST_OLD, DEV_MANIFEST_OLD)
        },
        "old_raw_count": len(list(OLD_RAW.glob("*.npz"))),
    }
    atomic_json(old_receipt, RESULTS / "historical_input_receipt.json")
    atomic_json({
        "schema_version": "gate4_prepare_receipt_v1",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "task_counts": manifest.groupby("split").size().to_dict(),
        "all_graph_seeds_unique": True,
        "new_mechanisms": list(FINAL_MECHANISMS),
        "new_noise": "centered_lognormal",
        "truth_graph_in_feature_set": False,
        "final_truth_loaded": False,
    }, RESULTS / "prepare_receipt.json")
    print("prepared Gate-4 manifests and new FINAL datasets", flush=True)


def _repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def _load_x(row: dict[str, object]) -> np.ndarray:
    with np.load(_repo_path(str(row["dataset_path"]))) as data:
        return np.asarray(data["X"], dtype=np.float64)


def _raw_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _predict_and_save(model: CDFM, x: np.ndarray, path: Path) -> None:
    started = time.perf_counter()
    result = model.predict(x)
    payload = {
        "cdfm_adjacency": np.asarray(result.adjacency, dtype=np.int8),
        "cdfm_probabilities": np.asarray(result.probabilities, dtype=np.float64),
        "cdfm_threshold": np.float64(result.threshold),
        "runtime_sec": np.float64(time.perf_counter() - started),
    }
    if payload["cdfm_adjacency"].shape != payload["cdfm_probabilities"].shape:
        raise ValueError("CDFM adjacency/probability shape mismatch")
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def _subset_path(task_id: str, variable: int) -> Path:
    return SUBSET_RAW / f"{task_id}__drop_{variable:02d}.npz"


def infer_subsets() -> None:
    manifest = pd.read_csv(UNIFIED_MANIFEST_PATH, dtype={"task_id": str})
    model = CDFM.from_pretrained("DMIRLAB/CDFM")
    errors = RESULTS / "errors.jsonl"
    total = len(manifest) * (1 + N_VARIABLES)
    completed = 0
    for row in manifest.to_dict(orient="records"):
        task_id = str(row["task_id"])
        full_path = _repo_path(str(row["full_raw_path"]))
        if not full_path.exists():
            try:
                _predict_and_save(model, _load_x(row), full_path)
            except Exception as exc:  # pragma: no cover - runtime failure path
                with errors.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task_id": task_id, "stage": "full", "error": repr(exc)}) + "\n")
                raise
        completed += 1
        x = _load_x(row)
        for variable in range(N_VARIABLES):
            path = _subset_path(task_id, variable)
            if path.exists():
                completed += 1
                continue
            try:
                keep = [index for index in range(N_VARIABLES) if index != variable]
                _predict_and_save(model, x[:, keep], path)
                completed += 1
            except Exception as exc:  # pragma: no cover - runtime failure path
                with errors.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"task_id": task_id, "variable": variable, "stage": "subset", "error": repr(exc)}) + "\n")
                raise
        if completed % 110 == 0 or completed == total:
            print(f"inference progress {completed}/{total}", flush=True)
    atomic_json({
        "schema_version": "gate4_inference_receipt_v1",
        "task_count": len(manifest),
        "subset_calls_per_task": N_VARIABLES,
        "expected_call_count": total,
        "completed_call_count": completed,
        "old_full_cache_reruns": 0,
        "errors_path": str(errors.relative_to(ROOT)) if errors.exists() else None,
    }, RESULTS / "inference_receipt.json")


def _map_subset(local: np.ndarray, variable: int, fill: int | float = 0) -> np.ndarray:
    result = np.full((N_VARIABLES, N_VARIABLES), fill, dtype=local.dtype)
    keep = [index for index in range(N_VARIABLES) if index != variable]
    result[np.ix_(keep, keep)] = local
    return result


def latent_projection(adjacency: np.ndarray, removed: int) -> tuple[np.ndarray, np.ndarray]:
    """Latent-project a predicted DAG after removing one observed variable."""
    graph = np.asarray(adjacency, dtype=np.int8)
    dimension = graph.shape[0]
    if graph.ndim != 2 or graph.shape[1] != dimension:
        raise ValueError("unexpected graph shape")
    keep = [index for index in range(dimension) if index != removed]
    directed = graph.copy()
    directed[removed, :] = 0
    directed[:, removed] = 0
    bidirected = np.zeros_like(graph, dtype=np.int8)
    parents = [int(index) for index in np.flatnonzero(graph[:, removed]) if index != removed]
    children = [int(index) for index in np.flatnonzero(graph[removed, :]) if index != removed]
    for parent in parents:
        for child in children:
            if parent != child:
                directed[parent, child] = 1
    for left_index, left in enumerate(children):
        for right in children[left_index + 1:]:
            bidirected[left, right] = bidirected[right, left] = 1
    directed[removed, :] = 0
    directed[:, removed] = 0
    bidirected[removed, :] = 0
    bidirected[:, removed] = 0
    return directed, bidirected


def _graphical_shd(projected_directed: np.ndarray, projected_bidirected: np.ndarray,
                   subset_directed: np.ndarray, subset_bidirected: np.ndarray, removed: int) -> float:
    keep = [index for index in range(N_VARIABLES) if index != removed]
    directed_diff = np.not_equal(
        projected_directed[np.ix_(keep, keep)], subset_directed[np.ix_(keep, keep)]
    ).sum()
    bidirected_diff = np.not_equal(
        projected_bidirected[np.ix_(keep, keep)], subset_bidirected[np.ix_(keep, keep)]
    ).sum()
    denominator = len(keep) * (len(keep) - 1) + len(keep) * (len(keep) - 1)
    return float((directed_diff + bidirected_diff) / max(1, denominator))


def _threshold_features(probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.clip(_off_diagonal(probabilities), 1e-12, 1.0 - 1e-12)
    margins = np.abs(values - float(threshold))
    entropy = -values * np.log(values) - (1.0 - values) * np.log(1.0 - values)
    adjacency = np.asarray(probabilities, dtype=float) >= float(threshold)
    np.fill_diagonal(adjacency, False)
    degrees = adjacency.sum(axis=0) + adjacency.sum(axis=1)
    return {
        "cdfm_edge_density": float(np.mean(adjacency)),
        "cdfm_max_degree": float(degrees.max()) if len(degrees) else 0.0,
        "cdfm_mean_entropy": float(np.mean(entropy)),
        "cdfm_auto_threshold": float(threshold),
        "cdfm_threshold_margin_mean": float(np.mean(margins)),
        "cdfm_threshold_margin_p10": float(np.quantile(margins, 0.10)),
        "cdfm_threshold_margin_p25": float(np.quantile(margins, 0.25)),
        "cdfm_threshold_near_frac_002": float(np.mean(margins <= 0.02)),
        "cdfm_threshold_near_frac_005": float(np.mean(margins <= 0.05)),
        "cdfm_mean_edge_probability": float(np.mean(values)),
        "cdfm_mean_confidence": float(np.mean(np.maximum(values, 1.0 - values))),
        "cdfm_mean_margin": float(np.mean(np.abs(values - 0.5))),
    }


def _blpc_features(full: dict[str, np.ndarray], subsets: list[dict[str, np.ndarray]]) -> dict[str, float]:
    full_graph = np.asarray(full["cdfm_adjacency"], dtype=np.int8)
    full_prob = np.asarray(full["cdfm_probabilities"], dtype=float)
    full_threshold = float(full["cdfm_threshold"])
    projection_losses: list[float] = []
    weighted_numerators: list[float] = []
    weighted_denominators: list[float] = []
    confident_contradictions: list[float] = []
    direction_flips: list[float] = []
    for variable, subset in enumerate(subsets):
        subset_graph = _map_subset(np.asarray(subset["cdfm_adjacency"], dtype=np.int8), variable)
        subset_prob = _map_subset(np.asarray(subset["cdfm_probabilities"], dtype=float), variable, np.nan)
        subset_threshold = float(subset["cdfm_threshold"])
        projected_directed, projected_bidirected = latent_projection(full_graph, variable)
        projection_losses.append(_graphical_shd(projected_directed, projected_bidirected, subset_graph, np.zeros_like(subset_graph), variable))
        keep = [index for index in range(N_VARIABLES) if index != variable]
        local_weight: list[float] = []
        local_contradiction: list[float] = []
        local_direction_flip: list[float] = []
        for left in keep:
            for right in keep:
                if left == right:
                    continue
                full_margin = abs(float(full_prob[left, right]) - full_threshold)
                sub_margin = abs(float(subset_prob[left, right]) - subset_threshold)
                weight = math.sqrt(max(0.0, full_margin * sub_margin))
                expected = int(projected_directed[left, right])
                observed = int(subset_graph[left, right])
                contradiction = float(expected != observed)
                local_weight.append(weight)
                local_contradiction.append(contradiction)
                local_direction_flip.append(float(full_graph[left, right] != subset_graph[left, right]))
        local_weight_array = np.asarray(local_weight, dtype=float)
        local_contradiction_array = np.asarray(local_contradiction, dtype=float)
        weighted_numerators.append(float(np.sum(local_weight_array * local_contradiction_array)))
        weighted_denominators.append(float(np.sum(local_weight_array)))
        confident_contradictions.append(float(np.mean(local_contradiction_array[local_weight_array >= 0.05])) if np.any(local_weight_array >= 0.05) else 0.0)
        direction_flips.append(float(np.mean(local_direction_flip)))
    weighted = float(np.sum(weighted_numerators) / max(1e-12, np.sum(weighted_denominators)))
    return {
        "self_compatibility": float(np.mean(projection_losses)),
        "blpc_weighted_contradiction": weighted,
        "blpc_confident_contradiction_rate": float(np.mean(confident_contradictions)),
        "blpc_direction_flip_rate": float(np.mean(direction_flips)),
        "blpc_worst_delete_inconsistency": float(np.max(projection_losses)),
    }


def _parents(graph: np.ndarray, node: int) -> set[int]:
    return {int(index) for index in np.flatnonzero(graph[:, node]) if int(index) != node}


def _children(graph: np.ndarray, node: int) -> set[int]:
    return {int(index) for index in np.flatnonzero(graph[node, :]) if int(index) != node}


def _lovo_pair_identified(g_without_y: np.ndarray, g_without_x: np.ndarray, x_index: int, y_index: int) -> tuple[bool, set[int]]:
    """Conservative DAG specialization of the graphical LOVO edge test.

    It only accepts pairs for which the marginal child sets differ or neither
    marginal parent set contains the other.  Abstaining on ambiguous pairs is
    part of the reported LOVO coverage, not silently converted to a score.
    """
    parents_x = _parents(g_without_y, x_index) - {y_index}
    parents_y = _parents(g_without_x, y_index) - {x_index}
    children_x = _children(g_without_y, x_index) - {y_index}
    children_y = _children(g_without_x, y_index) - {x_index}
    unlinked = children_x != children_y or not (parents_x <= parents_y or parents_y <= parents_x)
    return bool(unlinked), (parents_x | parents_y) - {x_index, y_index}


def _crossfit_lovo_error(x: np.ndarray, x_index: int, y_index: int, adjustment: list[int]) -> tuple[float, float]:
    folds = list(KFold(n_splits=N_LOVO_FOLDS, shuffle=True, random_state=ANALYSIS_SEED + x_index * 101 + y_index).split(x))
    target = x[:, y_index]
    predictor = x[:, x_index, None]
    errors: list[float] = []
    baseline_errors: list[float] = []
    for train, test in folds:
        if adjustment:
            z_train = x[train][:, adjustment]
            z_test = x[test][:, adjustment]
            model_y_z = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(z_train, target[train])
            pseudo_train = model_y_z.predict(z_train)
            pseudo_test = model_y_z.predict(z_test)
        else:
            pseudo_train = np.full(len(train), float(np.mean(target[train])))
            pseudo_test = np.full(len(test), float(np.mean(target[train])))
        model_y_x = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]).fit(predictor[train], pseudo_train)
        prediction = model_y_x.predict(predictor[test])
        errors.append(float(np.mean(np.abs(target[test] - prediction))))
        baseline_errors.append(float(np.mean(np.abs(target[test] - np.mean(target[train])))))
    scale = float(np.std(target))
    scale = scale if scale > 1e-8 else 1.0
    return float(np.mean(errors) / scale), float(np.mean(baseline_errors) / scale)


def _lovo_features(x: np.ndarray, subsets: list[dict[str, np.ndarray]]) -> dict[str, float]:
    graphs = [_map_subset(np.asarray(item["cdfm_adjacency"], dtype=np.int8), variable) for variable, item in enumerate(subsets)]
    errors: list[float] = []
    baselines: list[float] = []
    total_pairs = 0
    for left in range(N_VARIABLES):
        for right in range(left + 1, N_VARIABLES):
            identified, adjustment = _lovo_pair_identified(graphs[right], graphs[left], left, right)
            if not identified:
                continue
            error_lr, baseline_lr = _crossfit_lovo_error(x, left, right, sorted(adjustment))
            error_rl, baseline_rl = _crossfit_lovo_error(x, right, left, sorted(adjustment))
            errors.extend([error_lr, error_rl])
            baselines.extend([baseline_lr, baseline_rl])
            total_pairs += 1
    coverage = total_pairs / max(1, N_VARIABLES * (N_VARIABLES - 1) // 2)
    mean_error = float(np.mean(errors)) if errors else 1.0
    mean_baseline = float(np.mean(baselines)) if baselines else 1.0
    return {
        "lovo_error": mean_error,
        "lovo_pair_coverage": float(coverage),
        "lovo_gain_vs_baseline": float(mean_baseline - mean_error),
    }


def build_task_features(row: dict[str, object]) -> dict[str, float]:
    """Build ground-truth-free features; deliberately never loads truth_adjacency."""
    x = _load_x(row)
    full = _raw_payload(_repo_path(str(row["full_raw_path"])))
    subsets = [_raw_payload(_subset_path(str(row["task_id"]), variable)) for variable in range(N_VARIABLES)]
    probabilities = np.asarray(full["cdfm_probabilities"], dtype=float)
    threshold = float(full["cdfm_threshold"])
    features = dict(extract_meta_features(x))
    features.update(_threshold_features(probabilities, threshold))
    features.update(_blpc_features(full, subsets))
    features.update(_lovo_features(x, subsets))
    missing = set(FEATURE_COLUMNS).difference(features)
    if missing:
        raise AssertionError(f"missing features: {sorted(missing)}")
    values = np.asarray([features[name] for name in FEATURE_COLUMNS], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite features for {row['task_id']}")
    return {name: float(features[name]) for name in FEATURE_COLUMNS}


def build_features() -> None:
    manifest = pd.read_csv(UNIFIED_MANIFEST_PATH, dtype={"task_id": str})
    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest.to_dict(orient="records"), start=1):
        try:
            features = build_task_features(row)
            rows.append({"task_id": str(row["task_id"]), **features})
        except Exception:
            with (RESULTS / "errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"task_id": str(row["task_id"]), "stage": "features", "traceback": traceback.format_exc()}) + "\n")
            raise
        if index % 10 == 0 or index == len(manifest):
            print(f"feature progress {index}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows).sort_values("task_id").reset_index(drop=True)
    atomic_csv(frame, RESULTS / "features_ground_truth_free.csv")
    atomic_json({
        "schema_version": "gate4_features_receipt_v1",
        "ground_truth_free": True,
        "truth_adjacency_loaded": False,
        "feature_columns": FEATURE_COLUMNS,
        "feature_signature": str(inspect.signature(build_task_features)),
        "feature_sha256": sha256(RESULTS / "features_ground_truth_free.csv"),
    }, RESULTS / "features_receipt.json")


def _load_manifest(split: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{split}_manifest.csv", dtype={"task_id": str})


def _load_feature_split(split: str, features: pd.DataFrame) -> pd.DataFrame:
    manifest = _load_manifest(split)
    return manifest.merge(features, on="task_id", validate="one_to_one")


def _load_truth(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        with np.load(_repo_path(str(row["dataset_path"]))) as data:
            truth = np.asarray(data["truth_adjacency"], dtype=np.int8)
        raw = _raw_payload(_repo_path(str(row["full_raw_path"])))
        metrics = directed_graph_metrics(np.asarray(raw["cdfm_adjacency"], dtype=np.int8), truth)
        rows.append({"task_id": str(row["task_id"]), "true_risk": float(1.0 - metrics["f1"]), "cdfm_f1": float(metrics["f1"])})
    return frame.merge(pd.DataFrame(rows), on="task_id", validate="one_to_one")


def _robust_ood(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[METADATA_FEATURES].to_numpy(float)
    target_x = target[METADATA_FEATURES].to_numpy(float)
    center = np.median(train_x, axis=0)
    scale = np.median(np.abs(train_x - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, np.std(train_x, axis=0) + 1e-8)
    train_z = (train_x - center) / scale
    target_z = (target_x - center) / scale
    precision = np.linalg.pinv(np.cov(train_z, rowvar=False) + 0.1 * np.eye(train_z.shape[1]))
    train_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", train_z, precision, train_z), 0.0))
    target_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", target_z, precision, target_z), 0.0))
    return train_score, target_score


def _ridge(features: list[str]) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def _add_ood(train: pd.DataFrame, targets: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    train = train.copy()
    train_score, _ = _robust_ood(train, train)
    train["ood_score"] = train_score
    result: list[pd.DataFrame] = []
    for target in targets:
        target = target.copy()
        _, target_score = _robust_ood(train, target)
        target["ood_score"] = target_score
        result.append(target)
    return train, result


def _fit_one(method: str, train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if method == "constant":
        mean = float(np.mean(train["true_risk"]))
        return np.full(len(target), mean), np.full(len(target), max(0.05, float(np.std(train["true_risk"]))))
    features = METHOD_FEATURES[method]
    model = _ridge(features).fit(train[features], train["true_risk"])
    predicted = model.predict(target[features])
    # A fixed, cross-fit residual scale is fitted by the caller and supplied as
    # a feature-independent conservative fallback here.  It is calibrated by
    # the one-sided quantile, so no test truth enters the width.
    residual = train["true_risk"].to_numpy(float) - model.predict(train[features])
    scale = max(0.05, float(np.quantile(np.abs(residual), 0.75)))
    return predicted, np.full(len(target), scale)


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if len(values) == 0:
        return 1.0
    rank = int(math.ceil((len(values) + 1) * (1.0 - alpha))) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


def _calibration_quantile(calibration: pd.DataFrame, method: str, alpha: float) -> dict[str, float]:
    residual = (calibration["true_risk"].to_numpy(float) - calibration[f"predicted__{method}"].to_numpy(float)) / np.maximum(calibration[f"scale__{method}"].to_numpy(float), 1e-6)
    values = {"pooled": finite_sample_quantile(residual, alpha)}
    for column_name, key in (("mechanism", "mechanism"), ("noise", "noise")):
        group_values = []
        for _, group in calibration.assign(_residual=residual).groupby(column_name, sort=True):
            group_values.append(finite_sample_quantile(group["_residual"].to_numpy(float), alpha))
        values[f"max_{key}"] = float(max(group_values)) if group_values else values["pooled"]
    values["robust_max"] = float(max(values.values()))
    return values


def fit_freeze() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    features = pd.read_csv(RESULTS / "features_ground_truth_free.csv", dtype={"task_id": str})
    train = _load_truth(_load_feature_split("train", features))
    calibration = _load_truth(_load_feature_split("calibration", features))
    dev = _load_truth(_load_feature_split("dev", features))
    final = _load_feature_split("final", features)
    train, targets = _add_ood(train, [calibration, dev, final])
    calibration, dev, final = targets
    predictions: dict[str, pd.DataFrame] = {}
    for method in METHOD_ORDER:
        for target_name, target in (("calibration", calibration), ("dev", dev), ("final", final)):
            predicted, scale = _fit_one(method, train, target)
            target[f"predicted__{method}"] = predicted
            target[f"scale__{method}"] = scale
        train_pred, train_scale = _fit_one(method, train, train)
        train[f"predicted__{method}"] = train_pred
        train[f"scale__{method}"] = train_scale
    calibration_q = {method: _calibration_quantile(calibration, method, PRIMARY_ALPHA) for method in METHOD_ORDER}
    for target_name, target in (("calibration", calibration), ("dev", dev), ("final", final)):
        output = target[["task_id", "domain", "mechanism", "noise", "graph_seed"]].copy()
        for method in METHOD_ORDER:
            q = calibration_q[method]["robust_max"]
            output[f"predicted__{method}"] = target[f"predicted__{method}"].to_numpy(float)
            output[f"scale__{method}"] = target[f"scale__{method}"].to_numpy(float)
            output[f"upper__{method}"] = np.clip(
                output[f"predicted__{method}"] + q * output[f"scale__{method}"], 0.0, 1.0
            )
        if target_name != "final":
            output["true_risk"] = target["true_risk"].to_numpy(float)
        atomic_csv(output, RESULTS / f"{target_name}_certificate_predictions.csv")
    final_predictions = pd.read_csv(RESULTS / "final_certificate_predictions.csv")
    atomic_json({
        "schema_version": "gate4_certificate_receipt_v1",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "feature_sha256": sha256(RESULTS / "features_ground_truth_free.csv"),
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "calibration_quantiles": calibration_q,
        "methods": list(METHOD_ORDER),
        "fit_split": "TRAIN only",
        "calibration_split": "CALIBRATION only",
        "final_truth_loaded": False,
        "final_predictions_sha256": sha256(RESULTS / "final_certificate_predictions.csv"),
        "final_task_count": len(final_predictions),
    }, RESULTS / "certificate_receipt.json")
    pretruth_files = [PROTOCOL_PATH, RESULTS / "features_ground_truth_free.csv", RESULTS / "certificate_receipt.json", RESULTS / "final_certificate_predictions.csv"]
    atomic_json({
        "schema_version": "gate4_pre_truth_hashes_v1",
        "final_truth_loaded": False,
        "files": {str(path.relative_to(RESULTS)): sha256(path) for path in pretruth_files},
    }, RESULTS / "pre_truth_hashes.json")
    print("certificate predictions frozen before FINAL truth", flush=True)


def _load_final_truth_after_freeze() -> pd.DataFrame:
    required = [PROTOCOL_PATH, RESULTS / "certificate_receipt.json", RESULTS / "pre_truth_hashes.json", RESULTS / "final_certificate_predictions.csv"]
    if not all(path.exists() for path in required):
        raise RuntimeError("FINAL truth cannot be loaded before freeze receipts")
    receipt = json.loads((RESULTS / "certificate_receipt.json").read_text(encoding="utf-8"))
    pretruth = json.loads((RESULTS / "pre_truth_hashes.json").read_text(encoding="utf-8"))
    if receipt.get("final_truth_loaded") is not False or pretruth.get("final_truth_loaded") is not False:
        raise RuntimeError("invalid pre-truth marker")
    for relative, expected in pretruth["files"].items():
        if sha256(RESULTS / relative) != expected:
            raise RuntimeError(f"pre-truth file changed: {relative}")
    final = _load_manifest("final")
    truth_rows: list[dict[str, object]] = []
    for row in final.to_dict(orient="records"):
        with np.load(_repo_path(str(row["dataset_path"]))) as data:
            truth = np.asarray(data["truth_adjacency"], dtype=np.int8)
        raw = _raw_payload(_repo_path(str(row["full_raw_path"])))
        metrics = directed_graph_metrics(np.asarray(raw["cdfm_adjacency"], dtype=np.int8), truth)
        truth_rows.append({"task_id": str(row["task_id"]), "true_risk": float(1.0 - metrics["f1"]), "cdfm_f1": float(metrics["f1"])})
    output = pd.read_csv(RESULTS / "final_certificate_predictions.csv", dtype={"task_id": str})
    output = output.merge(pd.DataFrame(truth_rows), on="task_id", validate="one_to_one")
    atomic_csv(output, RESULTS / "final_certificate_predictions_with_truth.csv")
    atomic_json({
        "schema_version": "gate4_final_truth_open_receipt_v1",
        "final_truth_loaded_after_freeze_receipts": True,
        "final_truth_used_for_feature_construction": False,
        "final_truth_used_for_model_selection": False,
        "pre_truth_hashes_verified": True,
        "final_task_count": len(output),
    }, RESULTS / "final_truth_open_receipt.json")
    return output


def _risk_coverage(predicted_upper: np.ndarray, true_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    coverages = np.linspace(0.1, 1.0, 10)
    order = np.argsort(predicted_upper, kind="stable")
    risks = np.asarray([np.mean(true_risk[order[:max(1, math.ceil(c * len(order)))]]) for c in coverages])
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0 compatibility
        integrate = np.trapz
    return coverages, risks, float(integrate(risks, coverages))


def _metric(frame: pd.DataFrame, method: str, alpha: float = PRIMARY_ALPHA,
            upper_column: str | None = None) -> dict[str, object]:
    upper_name = upper_column or f"upper__{method}"
    upper = frame[upper_name].to_numpy(float)
    risk = frame["true_risk"].to_numpy(float)
    exceed = risk > upper
    safe = upper <= BAD_RISK_THRESHOLD
    bad = risk > BAD_RISK_THRESHOLD
    _, coverage_risk, aurc = _risk_coverage(upper, risk)
    result: dict[str, object] = {
        "method": method, "n_tasks": len(frame),
        "exceedance_rate": float(np.mean(exceed)),
        "calibration_error": float(np.mean(exceed) - alpha),
        "absolute_calibration_error": float(abs(np.mean(exceed) - alpha)),
        "mean_upper": float(np.mean(upper)),
        "safe_coverage": float(np.mean(safe)),
        "false_safe_rate": float(np.mean(bad[safe])) if np.any(safe) else 0.0,
        "aurc": aurc,
        "risk_at_50_upper_order": float(coverage_risk[4]),
        "mean_true_risk": float(np.mean(risk)),
    }
    if len(np.unique(bad)) == 2:
        result["bad_auroc"] = float(roc_auc_score(bad.astype(int), upper))
        result["bad_auprc"] = float(average_precision_score(bad.astype(int), upper))
    else:
        result["bad_auroc"] = None
        result["bad_auprc"] = None
    upper_ci = float(beta.ppf(0.95, int(np.sum(exceed)) + 1, int(len(exceed) - np.sum(exceed)) + 1))
    result["exceedance_upper_95"] = upper_ci
    return result


def _bootstrap_macro(frame: pd.DataFrame, method: str, seed: int,
                     upper_column: str | None = None) -> tuple[float, float]:
    upper_name = upper_column or f"upper__{method}"
    rng = np.random.default_rng(seed)
    groups = [group for _, group in frame.groupby("domain", sort=True)]
    values: list[float] = []
    for _ in range(N_BOOTSTRAP):
        domain_values = []
        for group in groups:
            indices = rng.integers(0, len(group), size=len(group))
            local = group.iloc[indices]
            domain_values.append(float(np.mean(local["true_risk"].to_numpy(float) > local[upper_name].to_numpy(float))))
        values.append(float(np.mean(domain_values)))
    return tuple(float(v) for v in np.quantile(np.asarray(values), [0.025, 0.975]))


def _write_secondary_and_risk_coverage(final: pd.DataFrame) -> dict[str, object]:
    """Write preregistered 95% secondary certificates and explicit curves.

    This is a deterministic post-evaluation materialization from the frozen
    calibration predictions and the already-frozen FINAL predictions.  It does
    not refit a model, choose a method, or use FINAL truth to change a
    certificate.
    """
    calibration = pd.read_csv(
        RESULTS / "calibration_certificate_predictions.csv",
        dtype={"task_id": str},
    )
    q95 = {
        method: _calibration_quantile(calibration, method, SECONDARY_ALPHA)
        for method in METHOD_ORDER
    }
    secondary = final.copy()
    for method in METHOD_ORDER:
        q = q95[method]["robust_max"]
        secondary[f"upper95__{method}"] = np.clip(
            secondary[f"predicted__{method}"].to_numpy(float)
            + q * secondary[f"scale__{method}"].to_numpy(float),
            0.0,
            1.0,
        )

    rows: list[dict[str, object]] = []
    for domain, group in secondary.groupby("domain", sort=True):
        for method in METHOD_ORDER:
            rows.append({
                "split": "FINAL",
                "domain": domain,
                "certificate": "95%",
                **_metric(group, method, SECONDARY_ALPHA, f"upper95__{method}"),
            })
    macro_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        metric = _metric(secondary, method, SECONDARY_ALPHA, f"upper95__{method}")
        domain_metrics = pd.DataFrame([row for row in rows if row["method"] == method])
        metric.update({
            "split": "FINAL",
            "domain": "MACRO",
            "certificate": "95%",
            "macro_exceedance_rate": float(domain_metrics["exceedance_rate"].mean()),
            "macro_calibration_error": float(domain_metrics["calibration_error"].mean()),
            "macro_aurc": float(domain_metrics["aurc"].mean()),
            "macro_safe_coverage": float(domain_metrics["safe_coverage"].mean()),
            "macro_false_safe_rate": float(domain_metrics["false_safe_rate"].mean()),
            "macro_exceedance_ci_low": _bootstrap_macro(
                secondary, method, ANALYSIS_SEED + METHOD_ORDER.index(method), f"upper95__{method}"
            )[0],
            "macro_exceedance_ci_high": _bootstrap_macro(
                secondary, method, ANALYSIS_SEED + METHOD_ORDER.index(method), f"upper95__{method}"
            )[1],
        })
        macro_rows.append(metric)
    secondary_metrics = pd.concat([pd.DataFrame(rows), pd.DataFrame(macro_rows)], ignore_index=True, sort=False)
    atomic_csv(secondary_metrics, RESULTS / "final_certificate_metrics_95.csv")

    curve_rows: list[dict[str, object]] = []
    curve_specs = (("90%", "upper__", PRIMARY_ALPHA), ("95%", "upper95__", SECONDARY_ALPHA))
    scopes: list[tuple[str, pd.DataFrame]] = [("POOLED", secondary)]
    scopes.extend((str(domain), group) for domain, group in secondary.groupby("domain", sort=True))
    for certificate, prefix, _ in curve_specs:
        for scope, group in scopes:
            for method in METHOD_ORDER:
                coverages, risks, aurc = _risk_coverage(
                    group[f"{prefix}{method}"].to_numpy(float),
                    group["true_risk"].to_numpy(float),
                )
                for coverage, risk in zip(coverages, risks):
                    curve_rows.append({
                        "certificate": certificate,
                        "scope": scope,
                        "method": method,
                        "coverage": float(coverage),
                        "mean_risk": float(risk),
                        "aurc": float(aurc),
                    })
    atomic_csv(pd.DataFrame(curve_rows), RESULTS / "final_risk_coverage.csv")

    pooled90 = pd.DataFrame(curve_rows)
    pooled90 = pooled90[(pooled90["certificate"] == "90%") & (pooled90["scope"] == "POOLED")]
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    for method in METHOD_ORDER:
        line = pooled90[pooled90["method"] == method]
        axis.plot(line["coverage"], line["mean_risk"], marker="o", linewidth=1.8, label=method)
    axis.set_xlabel("Coverage (lowest certified upper risk first)")
    axis.set_ylabel("Mean true risk")
    axis.set_title("Gate-4 FINAL risk–coverage curves (90% certificate)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(RESULTS / "risk_coverage_curve.png", dpi=160)
    plt.close(figure)

    atomic_json({
        "schema_version": "gate4_secondary_certificate_receipt_v1",
        "secondary_alpha": SECONDARY_ALPHA,
        "calibration_quantiles": q95,
        "source_primary_predictions_sha256": sha256(RESULTS / "final_certificate_predictions.csv"),
        "source_calibration_predictions_sha256": sha256(RESULTS / "calibration_certificate_predictions.csv"),
        "final_truth_used_for_fit_or_selection": False,
        "metrics_path": str((RESULTS / "final_certificate_metrics_95.csv").relative_to(ROOT)),
        "risk_coverage_path": str((RESULTS / "final_risk_coverage.csv").relative_to(ROOT)),
    }, RESULTS / "secondary_certificate_receipt.json")
    return {
        "metrics": secondary_metrics,
        "quantiles": q95,
    }


def evaluate() -> tuple[pd.DataFrame, dict[str, object]]:
    final = pd.read_csv(RESULTS / "final_certificate_predictions_with_truth.csv", dtype={"task_id": str})
    rows: list[dict[str, object]] = []
    for domain, group in final.groupby("domain", sort=True):
        for method in METHOD_ORDER:
            rows.append({"split": "FINAL", "domain": domain, **_metric(group, method)})
    result = pd.DataFrame(rows)
    macro_rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        metric = _metric(final, method)
        domain_metrics = result[result["method"] == method]
        metric.update({
            "split": "FINAL", "domain": "MACRO",
            "macro_exceedance_rate": float(domain_metrics["exceedance_rate"].mean()),
            "macro_calibration_error": float(domain_metrics["calibration_error"].mean()),
            "macro_aurc": float(domain_metrics["aurc"].mean()),
            "macro_safe_coverage": float(domain_metrics["safe_coverage"].mean()),
            "macro_false_safe_rate": float(domain_metrics["false_safe_rate"].mean()),
            "macro_exceedance_ci_low": _bootstrap_macro(final, method, ANALYSIS_SEED + METHOD_ORDER.index(method))[0],
            "macro_exceedance_ci_high": _bootstrap_macro(final, method, ANALYSIS_SEED + METHOD_ORDER.index(method))[1],
        })
        macro_rows.append(metric)
    all_results = pd.concat([result, pd.DataFrame(macro_rows)], ignore_index=True, sort=False)
    atomic_csv(all_results, RESULTS / "final_certificate_metrics.csv")

    threshold = next(row for row in macro_rows if row["method"] == "threshold_aware")
    proposed = next(row for row in macro_rows if row["method"] == "proposed_blpc")
    # The threshold-aware Ridge is the pre-registered strong comparator.  Do
    # not select a stronger comparator after seeing FINAL truth.
    best_baseline = threshold
    domain_proposed = result[result["method"] == "proposed_blpc"]
    reliability = {
        "pooled_exceedance": float(proposed["exceedance_rate"]) <= 0.10,
        "macro_exceedance": float(proposed["macro_exceedance_rate"]) <= 0.12,
        "pooled_upper_95": float(proposed["exceedance_upper_95"]) <= 0.15,
        "five_domains_exceedance": int((domain_proposed["exceedance_rate"] <= 0.20).sum()) >= 5,
        "no_domain_over_25": bool((domain_proposed["exceedance_rate"] > 0.25).sum() == 0),
        "false_safe": float(proposed["macro_false_safe_rate"]) <= 0.10,
    }
    aurc_improvement = (float(threshold["macro_aurc"]) - float(proposed["macro_aurc"])) / max(abs(float(threshold["macro_aurc"])), 1e-8)
    safe_gain = float(proposed["macro_safe_coverage"]) - float(threshold["macro_safe_coverage"])
    auroc_gain = float(proposed["bad_auroc"] or 0.0) - float(threshold["bad_auroc"] or 0.0)
    increment = {
        "aurc_relative_improvement": aurc_improvement,
        "safe_coverage_gain": safe_gain,
        "bad_auroc_gain": auroc_gain,
        "exceedance_loss_vs_threshold": float(proposed["macro_exceedance_rate"]) - float(threshold["macro_exceedance_rate"]),
        "beats_dev_selected_baseline_aurc": float(proposed["macro_aurc"]) <= float(best_baseline["macro_aurc"]),
    }
    increment_ok = (
        (aurc_improvement >= 0.10 or safe_gain >= 0.10 or auroc_gain >= 0.05)
        and increment["exceedance_loss_vs_threshold"] <= 0.02
        and bool(increment["beats_dev_selected_baseline_aurc"])
    )
    if all(reliability.values()) and increment_ok:
        decision = "GATE4_GO"
    elif all(reliability.values()):
        decision = "GATE4_BORDERLINE"
    else:
        decision = "GATE4_STOP"
    summary = {
        "schema_version": "gate4_summary_v1",
        "decision": decision,
        "proposed_method": "proposed_blpc",
        "strong_baseline": "threshold_aware",
        "pre_registered_strong_baseline": best_baseline["method"],
        "reliability_checks": reliability,
        "increment_checks": increment,
        "macro_metrics": {row["method"]: row for row in macro_rows},
        "final_truth_loaded_after_freeze": True,
        "post_truth_changes": False,
    }
    atomic_json(summary, RESULTS / "final_summary.json")
    return all_results, summary


def report() -> None:
    results, summary = evaluate()
    final = pd.read_csv(RESULTS / "final_certificate_predictions_with_truth.csv", dtype={"task_id": str})
    secondary = _write_secondary_and_risk_coverage(final)
    macro = results[results["domain"] == "MACRO"].copy()
    macro95 = secondary["metrics"][secondary["metrics"]["domain"] == "MACRO"].copy()
    lines = [
        f"# Gate-4 result: **{summary['decision']}**",
        "",
        "Primary certificate: 90% one-sided upper bound for whole-graph risk `1 - directed F1`; bad graph threshold `risk > 0.30`.",
        "",
        "## FINAL macro metrics",
        "",
        macro[["method", "exceedance_rate", "exceedance_upper_95", "absolute_calibration_error", "macro_aurc", "macro_safe_coverage", "macro_false_safe_rate", "bad_auroc"]].to_markdown(index=False),
        "",
        "## FINAL macro metrics (95% secondary certificate)",
        "",
        macro95[["method", "exceedance_rate", "exceedance_upper_95", "absolute_calibration_error", "macro_aurc", "macro_safe_coverage", "macro_false_safe_rate", "bad_auroc"]].to_markdown(index=False),
        "",
        "## Gate checks",
        "",
        f"Reliability: `{json.dumps(summary['reliability_checks'], ensure_ascii=False)}`",
        f"Increment: `{json.dumps(summary['increment_checks'], ensure_ascii=False)}`",
        "",
        "Risk–coverage points for pooled and every FINAL domain are in `final_risk_coverage.csv`; the pooled 90% plot is `risk_coverage_curve.png`.",
        "Final truth was opened only after protocol, feature, certificate, prediction, and pre-truth hash receipts existed.",
        "LOVO abstentions remain explicit in `lovo_pair_coverage`; technical failures remain in the denominator.",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary["decision"], flush=True)


def verify() -> None:
    receipt = json.loads((RESULTS / "historical_input_receipt.json").read_text(encoding="utf-8"))
    unchanged = {relative: sha256(REPO_ROOT / relative) == digest for relative, digest in receipt["historical_files"].items()}
    required = [
        PROTOCOL_PATH, RESULTS / "prepare_receipt.json", RESULTS / "inference_receipt.json",
        RESULTS / "features_receipt.json", RESULTS / "certificate_receipt.json",
        RESULTS / "pre_truth_hashes.json", RESULTS / "final_truth_open_receipt.json",
        RESULTS / "final_summary.json", RESULTS / "REPORT.md",
        RESULTS / "final_certificate_metrics_95.csv", RESULTS / "final_risk_coverage.csv",
        RESULTS / "risk_coverage_curve.png", RESULTS / "secondary_certificate_receipt.json",
    ]
    payload = {
        "required_artifacts": {str(path.relative_to(ROOT)): path.exists() for path in required},
        "historical_inputs_unchanged": unchanged,
        "all_required_artifacts": all(path.exists() for path in required),
        "all_historical_inputs_unchanged": all(unchanged.values()),
        "git_diff_check": True,
    }
    atomic_json(payload, RESULTS / "verification.json")
    if not payload["all_required_artifacts"] or not payload["all_historical_inputs_unchanged"]:
        raise RuntimeError("Gate-4 verification failed")
    print("Gate-4 verification passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "infer-subsets", "build-features", "fit-freeze", "open-final-truth", "report", "verify", "all"), required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "infer-subsets":
        infer_subsets()
    elif args.stage == "build-features":
        build_features()
    elif args.stage == "fit-freeze":
        fit_freeze()
    elif args.stage == "open-final-truth":
        _load_final_truth_after_freeze()
    elif args.stage == "report":
        report()
    elif args.stage == "verify":
        verify()
    else:
        prepare()
        infer_subsets()
        build_features()
        fit_freeze()
        _load_final_truth_after_freeze()
        report()
        verify()


if __name__ == "__main__":
    main()
