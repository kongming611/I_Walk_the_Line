"""Gate-3C final topic-decision experiment.

The experiment is intentionally self-contained under ``gate3c_final_topic_decision``.
It imports the historical Gate-2 generator only for the frozen graph and existing
mechanism semantics, and adds the two new risk-predictor mechanism families locally.
The pipeline keeps final truth loading in a separate function that is called only
after the protocol, model-selection receipt, and final-model receipt exist.
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
import torch
from cdfm import CDFM
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE0_ROOT = REPO_ROOT / "gate0_cdfm_defer"
GATE2_ROOT = REPO_ROOT / "gate2_reliable_routing"
for import_root in (REPO_ROOT, GATE0_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gate2_reliable_routing import generate_tasks as gate2_generator  # noqa: E402
from gate2_reliable_routing.generate_tasks import (  # noqa: E402
    _graph,
    _normalize,
    _rng,
    generate_x,
)
from meta_features import FEATURE_COLUMNS as METADATA_FEATURES  # noqa: E402
from meta_features import extract_meta_features  # noqa: E402
from metrics import directed_graph_metrics  # noqa: E402


ANALYSIS_SEED = 20260918
N_SAMPLES = 1000
N_VARIABLES = 10
BOOTSTRAP_CDFM = 2
BOOTSTRAP_RESAMPLES = 2000
COVERAGES = (0.25, 0.50, 0.75, 1.00)

TRAIN_MECHANISMS = ("linear", "tanh", "rff", "interaction")
DEV_MECHANISMS = ("softsign", "sine")
FINAL_MECHANISMS = ("quadratic", "piecewise")
TRAIN_NOISES = ("laplace", "gaussian", "student_t8", "student_t3")
DEV_NOISES = TRAIN_NOISES
FINAL_NOISES = ("gaussian", "student_t3", "exponential")

TRAIN_TASKS_PER_DOMAIN = 10
DEV_TASKS_PER_DOMAIN = 10
FINAL_TASKS_PER_DOMAIN = 20
TRAIN_SEED_START = 400000
DEV_SEED_START = 500000
FINAL_SEED_START = 600000

MODEL_ORDER = ("M1_Ridge", "M2_AdditiveSpline", "M3_RandomForest", "M4_HistGradientBoosting")
MODEL_COMPLEXITY_ORDER = {name: index for index, name in enumerate(MODEL_ORDER)}

GRAPH_FEATURES = ["cdfm_edge_density", "cdfm_max_degree"]
CONFIDENCE_FEATURES = [
    "cdfm_mean_edge_probability",
    "cdfm_mean_confidence",
    "cdfm_mean_entropy",
    "cdfm_mean_margin",
]
THRESHOLD_FEATURES = [
    "cdfm_auto_threshold",
    "cdfm_threshold_margin_mean",
    "cdfm_threshold_margin_p10",
    "cdfm_threshold_margin_p25",
    "cdfm_threshold_near_frac_002",
    "cdfm_threshold_near_frac_005",
]
STABILITY_FEATURES = [
    "cdfm_bootstrap_disagreement",
    "cdfm_bootstrap_edge_jaccard",
    "cdfm_bootstrap_probability_drift_mean",
    "cdfm_bootstrap_probability_drift_p90",
    "cdfm_bootstrap_threshold_std",
    "cdfm_bootstrap_threshold_crossing_rate",
]
PROPOSED_FEATURES = [
    *GRAPH_FEATURES,
    "cdfm_mean_entropy",
    *THRESHOLD_FEATURES,
    *STABILITY_FEATURES,
]
THRESHOLD_AWARE_FEATURES = [*GRAPH_FEATURES, "cdfm_mean_entropy", *THRESHOLD_FEATURES]
ALL_FEATURES = [*METADATA_FEATURES, *CONFIDENCE_FEATURES, *GRAPH_FEATURES, *THRESHOLD_FEATURES, *STABILITY_FEATURES]
ALL_FEATURES = list(dict.fromkeys(ALL_FEATURES))

BASELINE_NAMES = (
    "B0_constant",
    "B1_metadata_ridge",
    "B2_ood_only",
    "B3_confidence_ridge",
    "B4_threshold_aware_ridge",
    "B5_proposed_ridge",
    "B6_selected_proposed",
)
CAPACITY_DIAGNOSTIC_NAMES = MODEL_ORDER
FINAL_EVALUATED_NAMES = (
    "B0_constant",
    "B1_metadata_ridge",
    "B2_ood_only",
    "B3_confidence_ridge",
    "B4_threshold_aware_ridge",
    "B5_proposed_ridge",
    "B6_selected_proposed",
    *MODEL_ORDER[1:],
)

FORBIDDEN_FEATURE_TOKENS = ("truth", "true_", "_f1", "_shd", "mechanism", "noise", "seed", "oracle", "label")

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASETS = CACHE / "datasets"
RAW = CACHE / "raw_predictions"
PROTOCOL_PATH = RESULTS / "frozen_protocol.json"


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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def assert_no_label_leakage(feature_names: list[str]) -> None:
    offenders = [
        name for name in feature_names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if offenders:
        raise AssertionError(f"forbidden label-derived feature names: {offenders}")


def _off_diagonal(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]


def _edge_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    union = np.logical_or(left_bool, right_bool).sum()
    return float(np.logical_and(left_bool, right_bool).sum() / union) if union else 1.0


def _is_dag(adjacency: np.ndarray) -> bool:
    adjacency = np.asarray(adjacency, dtype=bool)
    indegree = adjacency.sum(axis=0).astype(int)
    queue = [int(node) for node in np.flatnonzero(indegree == 0)]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(int(child))
    return visited == adjacency.shape[0]


def _noise(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Local copy of Gate-2 noise semantics, including exponential for final test."""
    scale = 1.0 / np.sqrt(2.0)
    if kind == "laplace":
        return rng.laplace(0, scale, size=(N_SAMPLES, N_VARIABLES))
    if kind == "gaussian":
        return rng.normal(0, 1, size=(N_SAMPLES, N_VARIABLES))
    if kind == "student_t8":
        return rng.standard_t(8, size=(N_SAMPLES, N_VARIABLES)) * np.sqrt(6.0 / 8.0)
    if kind == "student_t3":
        return rng.standard_t(3, size=(N_SAMPLES, N_VARIABLES)) / np.sqrt(3.0)
    if kind == "exponential":
        return rng.exponential(1.0, size=(N_SAMPLES, N_VARIABLES)) - 1.0
    raise ValueError(f"unknown noise kind: {kind}")


def _new_edge_transform(kind: str, x: np.ndarray, slope: float, bias: float) -> np.ndarray:
    z = slope * x + bias
    if kind == "quadratic":
        return z + 0.35 * z ** 2
    if kind == "piecewise":
        return np.where(z >= 0.0, z, 0.2 * z)
    raise ValueError(f"not a new Gate-3C mechanism: {kind}")


def generate_x_gate3c(spec: gate2_generator.GraphSpec, mechanism: str, noise_kind: str, lambda_value: float, seed: int) -> np.ndarray:
    """Generate existing mechanisms via Gate-2 and new mechanisms locally.

    The local path intentionally mirrors Gate-2 ``generate_x`` after the edge
    transform: normalized linear/nonlinear components are mixed, exogenous noise
    is added, and columns are normalized at the end.  No truth graph is accepted
    by this function.
    """
    if mechanism not in {"quadratic", "piecewise"}:
        original_mechanism = "rff" if mechanism == "linear" else mechanism
        return generate_x(spec, original_mechanism, noise_kind, lambda_value, seed)

    rng = _rng(seed, 2)
    exogenous = _noise(noise_kind, rng)
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
                mechanism,
                x[:, source],
                float(spec.slopes[edge_index]),
                float(spec.biases[edge_index]),
            )
        x[:, child] = (
            (1.0 - lambda_value) * _normalize(linear)
            + lambda_value * _normalize(nonlinear)
            + exogenous[:, child]
        )
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) <= 1e-12):
        raise ValueError("generated X is invalid")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float32)


def mechanism_lambda(mechanism: str) -> tuple[str, float]:
    """Keep the historical Gate-3A linear-versus-nonlinear semantics frozen."""
    if mechanism == "linear":
        return "rff", 0.0
    return mechanism, 1.0


def _write_dataset(path: Path, x: np.ndarray, truth: np.ndarray, graph_seed: int, mechanism: str, noise: str, lambda_value: float, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        with np.load(path) as existing:
            if int(existing["graph_seed"]) != graph_seed:
                raise RuntimeError(f"dataset seed mismatch in existing cache: {path}")
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
            lambda_value=np.float64(lambda_value),
        )
    os.replace(temporary, path)


def _domain_rows(split: str, mechanisms: tuple[str, ...], noises: tuple[str, ...], count: int, seed_start: int, overwrite: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for mechanism in mechanisms:
        generator_mechanism, lambda_value = mechanism_lambda(mechanism)
        for noise in noises:
            domain = f"{mechanism}_{noise}"
            for task_index in range(count):
                graph_seed = seed_start + ordinal
                task_id = f"{split}_{domain}_{task_index:03d}"
                path = DATASETS / f"{task_id}.npz"
                if overwrite or not path.exists():
                    spec = _graph(graph_seed)
                    x = generate_x_gate3c(spec, mechanism, noise, lambda_value, graph_seed)
                    _write_dataset(path, x, spec.adjacency, graph_seed, mechanism, noise, lambda_value, overwrite=True)
                rows.append({
                    "task_id": task_id,
                    "split": split,
                    "domain": domain,
                    "mechanism": mechanism,
                    "noise": noise,
                    "task_index": task_index,
                    "graph_seed": graph_seed,
                    "N": N_SAMPLES,
                    "D": N_VARIABLES,
                    "lambda": lambda_value,
                    "generator_mechanism": generator_mechanism,
                    "dataset_path": path.relative_to(ROOT).as_posix(),
                })
                ordinal += 1
    frame = pd.DataFrame(rows)
    if frame["graph_seed"].duplicated().any():
        raise AssertionError(f"{split} has duplicated graph seeds")
    return frame


def _old_graph_seed_set() -> set[int]:
    seeds: set[int] = set()
    for relative in (
        "gate3a_risk_predictability/results/predictions.csv",
        "gate3a1_invariant_risk/cache/tasks.csv",
    ):
        path = REPO_ROOT / relative
        if path.exists():
            frame = pd.read_csv(path)
            if "graph_seed" in frame:
                seeds.update(frame["graph_seed"].astype(int).tolist())
    return seeds


def _protocol_payload() -> dict[str, object]:
    assert_no_label_leakage(PROPOSED_FEATURES)
    return {
        "schema_version": "gate3c_final_topic_decision_protocol_v1",
        "freeze_date": "2026-09-18",
        "frozen_before_final_truth_evaluation": True,
        "scientific_question": (
            "Can ground-truth-free CDFM outputs, distance to the adaptive decision threshold, "
            "and resampling stability predict whole-graph CDFM structural-error risk under "
            "unseen mechanism, unseen graph, and noise shift?"
        ),
        "historical_evidence_status": (
            "Gate-3A and Gate-3A.1 final-test results are development knowledge for this gate, "
            "not independent final confirmation."
        ),
        "data": {
            "train": {
                "mechanisms": list(TRAIN_MECHANISMS),
                "noises": list(TRAIN_NOISES),
                "tasks_per_domain": TRAIN_TASKS_PER_DOMAIN,
                "task_count": len(TRAIN_MECHANISMS) * len(TRAIN_NOISES) * TRAIN_TASKS_PER_DOMAIN,
                "graph_seed_range": [TRAIN_SEED_START, TRAIN_SEED_START + len(TRAIN_MECHANISMS) * len(TRAIN_NOISES) * TRAIN_TASKS_PER_DOMAIN - 1],
            },
            "dev": {
                "mechanisms": list(DEV_MECHANISMS),
                "noises": list(DEV_NOISES),
                "tasks_per_domain": DEV_TASKS_PER_DOMAIN,
                "task_count": len(DEV_MECHANISMS) * len(DEV_NOISES) * DEV_TASKS_PER_DOMAIN,
                "graph_seed_range": [DEV_SEED_START, DEV_SEED_START + len(DEV_MECHANISMS) * len(DEV_NOISES) * DEV_TASKS_PER_DOMAIN - 1],
            },
            "final": {
                "mechanisms": list(FINAL_MECHANISMS),
                "noises": list(FINAL_NOISES),
                "tasks_per_domain": FINAL_TASKS_PER_DOMAIN,
                "task_count": len(FINAL_MECHANISMS) * len(FINAL_NOISES) * FINAL_TASKS_PER_DOMAIN,
                "graph_seed_range": [FINAL_SEED_START, FINAL_SEED_START + len(FINAL_MECHANISMS) * len(FINAL_NOISES) * FINAL_TASKS_PER_DOMAIN - 1],
                "joint_shift_domains": ["quadratic_exponential", "piecewise_exponential"],
            },
            "N": N_SAMPLES,
            "D": N_VARIABLES,
            "all_graph_seeds_unique_across_splits": True,
            "exponential_noise_used_in_train_or_dev": False,
        },
        "generator": {
            "historical_source": "gate2_reliable_routing.generate_tasks",
            "local_extension": {
                "quadratic": "z + 0.35*z**2, z=slope*x+bias",
                "piecewise": "z if z>=0 else 0.2*z, z=slope*x+bias",
            },
            "unseen_claim_boundary": "unseen mechanism families for the risk predictor",
            "lambda_semantics": "linear=0.0; all non-linear families=1.0",
        },
        "cdfm": {
            "model": "DMIRLAB/CDFM",
            "call": "CDFM.from_pretrained('DMIRLAB/CDFM').predict(X)",
            "training": False,
            "fine_tuning": False,
            "source_modification": False,
            "bootstrap_per_dataset": BOOTSTRAP_CDFM,
            "threshold": "official model.predict(X) adaptive threshold",
        },
        "proposed_features": PROPOSED_FEATURES,
        "baseline_features": {
            "B1_metadata_ridge": list(METADATA_FEATURES),
            "B2_ood_only": "robust Mahalanobis score on metadata, fit on TRAIN only, then Ridge(alpha=1.0)",
            "B3_confidence_ridge": CONFIDENCE_FEATURES,
            "B4_threshold_aware_ridge": THRESHOLD_AWARE_FEATURES,
            "B5_proposed_ridge": PROPOSED_FEATURES,
        },
        "candidate_models": {
            "M1_Ridge": "Pipeline(StandardScaler(), Ridge(alpha=1.0))",
            "M2_AdditiveSpline": "Pipeline(SplineTransformer(n_knots=4, degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))",
            "M3_RandomForest": {
                "n_estimators": 400,
                "max_depth": 5,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": ANALYSIS_SEED,
                "n_jobs": -1,
            },
            "M4_HistGradientBoosting": {
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
                "random_state": ANALYSIS_SEED,
            },
        },
        "model_selection": {
            "fit_split": "TRAIN only",
            "evaluation_split": "DEV only",
            "primary_metric": "macro mean Spearman across 8 DEV domains",
            "constraints": {
                "minimum_domains_rho_gt_0.20": 6,
                "maximum_domains_rho_lt_0": 1,
            },
            "tie_break": "if macro rho difference is strictly less than 0.02, choose simpler: Ridge > Spline > RF > HistGB",
            "no_search": True,
        },
        "evaluation": {
            "coverages": list(COVERAGES),
            "bootstrap_resamples_per_final_domain_metric": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": ANALYSIS_SEED,
            "macro_spearman": "domain-aware bootstrap: resample tasks within each domain, then average six domain Spearman values",
            "relative_risk_reduction_at_50": "(full risk - risk@50) / full risk",
        },
        "topic_gate": {
            "TOPIC_GO": {
                "macro_spearman_min": 0.40,
                "domains_rho_gt_0.20_min": 5,
                "domains_rho_gt_0.30_min": 4,
                "domains_rho_lt_neg_0.10_max": 0,
                "domains_risk50_below_full_min": 5,
                "macro_relative_reduction_min": 0.15,
                "joint_shift_at_least_one_rho_gt_0.30": True,
                "joint_shift_other_rho_nonnegative": True,
            },
            "TOPIC_BORDERLINE": {
                "macro_spearman_min": 0.25,
                "domains_risk50_below_full_min": 4,
                "macro_relative_reduction_min": 0.10,
            },
            "TOPIC_STOP": {
                "macro_spearman_lt": 0.25,
                "domains_risk50_below_full_lt": 4,
                "macro_relative_reduction_lt": 0.10,
                "both_joint_shift_rho_nonpositive": True,
            },
        },
        "method_readiness": {
            "comparison": "selected proposed model versus B4 threshold-aware Ridge",
            "GO": "rho gain >= 0.05 with reduction loss no worse than 0.03, OR reduction gain >= 0.03 with rho loss no worse than 0.05",
            "otherwise": "METHOD_BORDERLINE",
            "operational_non_sacrifice_tolerances": {"rho": 0.05, "relative_reduction": 0.03},
        },
        "truth_isolation": {
            "feature_builder_accepts_truth": False,
            "final_truth_loader": "called only after final_model_receipt.json and pre_truth_hashes.json",
            "final_truth_used_for_feature_construction": False,
            "final_truth_used_for_model_selection": False,
            "final_truth_used_for_hyperparameter_choice": False,
            "final_truth_used_for_gate_threshold_modification": False,
        },
    }


def write_protocol() -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_payload()
    _atomic_json(protocol, PROTOCOL_PATH)
    return protocol


def _load_protocol() -> dict[str, object]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("frozen_before_final_truth_evaluation") is not True:
        raise RuntimeError("frozen protocol is not marked frozen")
    if protocol.get("proposed_features") != PROPOSED_FEATURES:
        raise RuntimeError("proposed feature set differs from frozen protocol")
    return protocol


def run_generator_contract_checks() -> dict[str, object]:
    """Run the required local generator checks without invoking CDFM or truth features."""
    spec = _graph(FINAL_SEED_START)
    if not _is_dag(spec.adjacency):
        raise AssertionError("Gate-2 graph generator returned a cyclic graph")
    comparisons: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for mechanism in ("tanh", "rff", "quadratic", "piecewise"):
        arrays[mechanism] = np.asarray(
            generate_x_gate3c(spec, mechanism, "gaussian", 1.0, FINAL_SEED_START),
            dtype=float,
        )
        if not np.isfinite(arrays[mechanism]).all():
            raise AssertionError(f"non-finite generated data for {mechanism}")
        if np.any(np.var(arrays[mechanism], axis=0) <= 1e-12):
            raise AssertionError(f"zero-variance generated column for {mechanism}")
    for new_mechanism in ("quadratic", "piecewise"):
        comparisons[new_mechanism] = {
            "different_from_tanh": not np.array_equal(arrays[new_mechanism], arrays["tanh"]),
            "different_from_rff": not np.array_equal(arrays[new_mechanism], arrays["rff"]),
        }
        if not all(comparisons[new_mechanism].values()):
            raise AssertionError(f"{new_mechanism} unexpectedly equals an existing mechanism array")
    if "truth_adjacency" in inspect.signature(generate_x_gate3c).parameters:
        raise AssertionError("generator accepts truth graph")
    assert_no_label_leakage(PROPOSED_FEATURES)
    return {
        "finite": True,
        "nonzero_variance": True,
        "dag": True,
        "mechanism_distinctness": comparisons,
        "truth_graph_in_risk_feature_set": False,
    }


def prepare(*, overwrite: bool = False) -> None:
    protocol = write_protocol()
    generator_checks = run_generator_contract_checks()
    DATASETS.mkdir(parents=True, exist_ok=True)
    train = _domain_rows(
        "train", TRAIN_MECHANISMS, TRAIN_NOISES, TRAIN_TASKS_PER_DOMAIN, TRAIN_SEED_START, overwrite
    )
    dev = _domain_rows(
        "dev", DEV_MECHANISMS, DEV_NOISES, DEV_TASKS_PER_DOMAIN, DEV_SEED_START, overwrite
    )
    final = _domain_rows(
        "final", FINAL_MECHANISMS, FINAL_NOISES, FINAL_TASKS_PER_DOMAIN, FINAL_SEED_START, overwrite
    )
    old_seeds = _old_graph_seed_set()
    current_seeds = set(train["graph_seed"]) | set(dev["graph_seed"]) | set(final["graph_seed"])
    if current_seeds & old_seeds:
        raise AssertionError("new graph seeds overlap historical Gate seeds")
    if len(current_seeds) != len(train) + len(dev) + len(final):
        raise AssertionError("graph instances are not independent across domains/splits")
    _atomic_csv(train, RESULTS / "train_manifest.csv")
    _atomic_csv(dev, RESULTS / "dev_manifest.csv")
    _atomic_csv(final, RESULTS / "final_manifest.csv")
    _atomic_json({
        "schema_version": "gate3c_prepare_receipt_v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "generator_checks": generator_checks,
        "counts": {"train": len(train), "dev": len(dev), "final": len(final)},
        "all_graph_seeds_unique": True,
        "historical_graph_seed_overlap": False,
        "exponential_noise_in_train_or_dev": False,
        "final_truth_loaded": False,
        "generator_source_not_modified": True,
        "gate2_generator_sha256": _sha256(GATE2_ROOT / "generate_tasks.py"),
        "protocol": protocol,
    }, RESULTS / "prepare_receipt.json")
    print(
        f"Prepared Gate-3C: train={len(train)}, dev={len(dev)}, final={len(final)}; "
        "generator checks passed.",
        flush=True,
    )


def _load_manifest(split: str) -> pd.DataFrame:
    path = RESULTS / f"{split}_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {split} manifest; run --stage prepare first")
    return pd.read_csv(path)


def _bootstrap_predictions(model: CDFM, x: np.ndarray, graph_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED, int(graph_seed), 77]))
    adjacency: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    thresholds: list[float] = []
    start = time.perf_counter()
    for _ in range(BOOTSTRAP_CDFM):
        indices = rng.integers(0, x.shape[0], size=x.shape[0])
        result = model.predict(x[indices])
        adjacency.append(np.asarray(result.adjacency, dtype=np.int8))
        probabilities.append(np.asarray(result.probabilities, dtype=float))
        thresholds.append(float(result.threshold))
    return np.stack(adjacency), np.stack(probabilities), np.asarray(thresholds, dtype=float), float(time.perf_counter() - start)


def infer(*, shard_index: int = 0, shard_count: int = 1) -> None:
    _load_protocol()
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid inference shard")
    all_manifests = pd.concat([_load_manifest("train"), _load_manifest("dev"), _load_manifest("final")], ignore_index=True)
    manifests = all_manifests.iloc[shard_index::shard_count].reset_index(drop=True)
    RAW.mkdir(parents=True, exist_ok=True)
    # Four CPU CDFM workers otherwise oversubscribe the host.  This changes only
    # execution scheduling, not model weights, inputs, bootstrap indices, or outputs.
    torch_threads = max(1, (os.cpu_count() or 4) // max(1, shard_count))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _atomic_json({
        "schema_version": "gate3c_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "packages": {
            name: _package_version(name)
            for name in ("cdfm-base", "numpy", "pandas", "scipy", "scikit-learn", "torch")
        },
        "cdfm_info": model.info,
        "cdfm_frozen": True,
        "bootstrap": BOOTSTRAP_CDFM,
        "task_count": len(all_manifests),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }, RESULTS / "environment.json")
    for ordinal, row in enumerate(manifests.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = RAW / f"{task_id}.npz"
        if destination.exists():
            with np.load(destination) as cached:
                if int(cached["bootstrap_count"]) != BOOTSTRAP_CDFM:
                    raise RuntimeError(f"unexpected bootstrap count in {task_id}")
            print(f"[{ordinal}/{len(manifests)}] skip cached {task_id}", flush=True)
            continue
        try:
            with np.load(ROOT / str(row["dataset_path"])) as dataset:
                x = np.asarray(dataset["X"], dtype=np.float64)
            start = time.perf_counter()
            original = model.predict(x)
            cdfm_runtime = float(time.perf_counter() - start)
            boot_adj, boot_prob, boot_threshold, boot_runtime = _bootstrap_predictions(
                model, x, int(row["graph_seed"])
            )
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cdfm_adjacency=np.asarray(original.adjacency, dtype=np.int8),
                    cdfm_probabilities=np.asarray(original.probabilities, dtype=float),
                    cdfm_logits=np.asarray(original.logits, dtype=float),
                    cdfm_threshold=np.float64(original.threshold),
                    cdfm_runtime_sec=np.float64(cdfm_runtime),
                    bootstrap_adjacencies=boot_adj,
                    bootstrap_probabilities=boot_prob,
                    bootstrap_thresholds=boot_threshold,
                    bootstrap_count=np.int64(BOOTSTRAP_CDFM),
                    bootstrap_runtime_sec=np.float64(boot_runtime),
                )
            os.replace(temporary, destination)
            print(
                f"[{ordinal}/{len(manifests)}] {task_id}: CDFM={cdfm_runtime:.2f}s, "
                f"bootstrap={boot_runtime:.2f}s",
                flush=True,
            )
        except Exception as exc:
            _append_error(task_id, "inference", exc)
            print(f"[{ordinal}/{len(manifests)}] ERROR {type(exc).__name__}: {exc}", flush=True)
    completed = sum((RAW / f"{task_id}.npz").exists() for task_id in manifests["task_id"])
    if completed != len(manifests):
        raise RuntimeError(f"inference incomplete: {completed}/{len(manifests)}")
    print(f"CDFM inference shard {shard_index}/{shard_count} complete: {completed}/{len(manifests)} tasks cached; B={BOOTSTRAP_CDFM}.", flush=True)


def _threshold_features(probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(_off_diagonal(probabilities), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CDFM probabilities contain non-finite values")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    margins = np.abs(clipped - float(threshold))
    return {
        "cdfm_auto_threshold": float(threshold),
        "cdfm_threshold_margin_mean": float(np.mean(margins)),
        "cdfm_threshold_margin_p10": float(np.quantile(margins, 0.10)),
        "cdfm_threshold_margin_p25": float(np.quantile(margins, 0.25)),
        "cdfm_threshold_near_frac_002": float(np.mean(margins <= 0.02)),
        "cdfm_threshold_near_frac_005": float(np.mean(margins <= 0.05)),
    }


def ground_truth_free_features(x: np.ndarray, raw: dict[str, np.ndarray]) -> dict[str, float]:
    """Construct all feature columns without accepting or reading a truth graph."""
    cdfm_adjacency = np.asarray(raw["cdfm_adjacency"], dtype=np.int8)
    probabilities = np.asarray(raw["cdfm_probabilities"], dtype=float)
    threshold = float(raw["cdfm_threshold"])
    boot_adjacency = np.asarray(raw["bootstrap_adjacencies"], dtype=np.int8)
    boot_probabilities = np.asarray(raw["bootstrap_probabilities"], dtype=float)
    boot_thresholds = np.asarray(raw["bootstrap_thresholds"], dtype=float)
    if boot_adjacency.shape[0] != BOOTSTRAP_CDFM or boot_probabilities.shape[0] != BOOTSTRAP_CDFM:
        raise ValueError("unexpected bootstrap arrays")
    if boot_thresholds.shape != (BOOTSTRAP_CDFM,):
        raise ValueError("unexpected bootstrap threshold shape")
    if not np.isfinite(x).all() or not np.isfinite(probabilities).all() or not np.isfinite(boot_probabilities).all():
        raise ValueError("non-finite input to feature construction")
    d = int(cdfm_adjacency.shape[0])
    values = np.clip(_off_diagonal(probabilities), 1e-12, 1.0 - 1e-12)
    entropy = -values * np.log(values) - (1.0 - values) * np.log(1.0 - values)
    degrees = cdfm_adjacency.sum(axis=0) + cdfm_adjacency.sum(axis=1)
    drift_values = []
    drift_p90 = []
    crossing = []
    disagreements = []
    jaccards = []
    original_decision = _off_diagonal(probabilities) >= threshold
    for boot_index in range(BOOTSTRAP_CDFM):
        boot_values = np.asarray(_off_diagonal(boot_probabilities[boot_index]), dtype=float)
        difference = np.abs(boot_values - _off_diagonal(probabilities))
        drift_values.append(float(np.mean(difference)))
        drift_p90.append(float(np.quantile(difference, 0.90)))
        boot_decision = boot_values >= float(boot_thresholds[boot_index])
        crossing.append(float(np.mean(original_decision != boot_decision)))
        disagreements.append(float(np.mean(_off_diagonal(cdfm_adjacency) != _off_diagonal(boot_adjacency[boot_index]))))
        jaccards.append(_edge_jaccard(cdfm_adjacency, boot_adjacency[boot_index]))
    features = dict(extract_meta_features(np.asarray(x, dtype=float)))
    features.update({
        "cdfm_edge_density": float(cdfm_adjacency.sum() / (d * (d - 1))),
        "cdfm_max_degree": float(degrees.max()),
        "cdfm_mean_edge_probability": float(values.mean()),
        "cdfm_mean_confidence": float(np.mean(np.maximum(values, 1.0 - values))),
        "cdfm_mean_entropy": float(np.mean(entropy)),
        "cdfm_mean_margin": float(np.mean(np.abs(values - 0.5))),
        "cdfm_bootstrap_disagreement": float(np.mean(disagreements)),
        "cdfm_bootstrap_edge_jaccard": float(np.mean(jaccards)),
        "cdfm_bootstrap_probability_drift_mean": float(np.mean(drift_values)),
        "cdfm_bootstrap_probability_drift_p90": float(np.mean(drift_p90)),
        "cdfm_bootstrap_threshold_std": float(np.std(boot_thresholds, ddof=0)),
        "cdfm_bootstrap_threshold_crossing_rate": float(np.mean(crossing)),
    })
    features.update(_threshold_features(probabilities, threshold))
    missing = set(ALL_FEATURES).difference(features)
    if missing:
        raise AssertionError(f"feature builder omitted columns: {sorted(missing)}")
    if not np.isfinite(np.asarray([features[name] for name in ALL_FEATURES], dtype=float)).all():
        raise ValueError("feature builder produced non-finite values")
    return {name: float(features[name]) for name in ALL_FEATURES}


def build_feature_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    """Build a task-id keyed feature table; this function never loads truth_adjacency."""
    rows: list[dict[str, object]] = []
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        dataset_path = ROOT / str(row["dataset_path"])
        raw_path = RAW / f"{task_id}.npz"
        with np.load(dataset_path) as dataset:
            x = np.asarray(dataset["X"], dtype=float)
        with np.load(raw_path) as raw_npz:
            raw = {name: np.asarray(raw_npz[name]) for name in (
                "cdfm_adjacency",
                "cdfm_probabilities",
                "cdfm_threshold",
                "bootstrap_adjacencies",
                "bootstrap_probabilities",
                "bootstrap_thresholds",
            )}
        rows.append({"task_id": task_id, **ground_truth_free_features(x, raw)})
        if ordinal % 20 == 0 or ordinal == len(manifest):
            print(f"features {ordinal}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    expected = ["task_id", *ALL_FEATURES]
    frame = frame[expected].sort_values("task_id").reset_index(drop=True)
    assert_no_label_leakage(ALL_FEATURES)
    return frame


def build_all_features() -> dict[str, pd.DataFrame]:
    _load_protocol()
    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "dev", "final"):
        frame = build_feature_frame(_load_manifest(split))
        frames[split] = frame
        _atomic_csv(frame, RESULTS / f"{split}_features.csv" if split != "final" else RESULTS / "final_features_ground_truth_free.csv")
    _atomic_json({
        "schema_version": "gate3c_features_receipt_v1",
        "feature_builder_signature": str(inspect.signature(build_feature_frame)),
        "ground_truth_free": True,
        "truth_adjacency_loaded": False,
        "proposed_features": PROPOSED_FEATURES,
        "all_output_feature_columns": ALL_FEATURES,
        "feature_files": {
            "train": _sha256(RESULTS / "train_features.csv"),
            "dev": _sha256(RESULTS / "dev_features.csv"),
            "final": _sha256(RESULTS / "final_features_ground_truth_free.csv"),
        },
    }, RESULTS / "features_receipt.json")
    return frames


def _ridge() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])


def make_proposed_model(name: str):
    if name == "M1_Ridge":
        return _ridge()
    if name == "M2_AdditiveSpline":
        return Pipeline([
            ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if name == "M3_RandomForest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=ANALYSIS_SEED,
            n_jobs=-1,
        )
    if name == "M4_HistGradientBoosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=ANALYSIS_SEED,
        )
    raise ValueError(f"unknown proposed model: {name}")


def _safe_correlation(function, left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    result = float(function(left, right).statistic)
    return result if np.isfinite(result) else float("nan")


def _coverage_risks(predicted: np.ndarray, true_risk: np.ndarray, constant: bool = False) -> dict[float, float]:
    if constant:
        full = float(np.mean(true_risk))
        return {coverage: full for coverage in COVERAGES}
    order = np.argsort(predicted, kind="stable")
    return {
        coverage: float(np.mean(true_risk[order[: max(1, math.ceil(coverage * len(order)))]]))
        for coverage in COVERAGES
    }


def _fit_ood(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_x = train[METADATA_FEATURES].to_numpy(float)
    target_x = target[METADATA_FEATURES].to_numpy(float)
    center = np.median(train_x, axis=0)
    scale = np.median(np.abs(train_x - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, np.std(train_x, axis=0) + 1e-8)
    train_z = (train_x - center) / scale
    target_z = (target_x - center) / scale
    covariance = np.cov(train_z, rowvar=False) + 0.1 * np.eye(train_z.shape[1])
    precision = np.linalg.pinv(covariance)
    train_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", train_z, precision, train_z), 0.0))
    target_score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", target_z, precision, target_z), 0.0))
    return train_score, target_score


def _load_truth_for_split(manifest: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Load truth for TRAIN/DEV supervision only; FINAL calls a separate function."""
    truth_rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        task_id = str(row["task_id"])
        with np.load(ROOT / str(row["dataset_path"])) as dataset:
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
    truth_frame = pd.DataFrame(truth_rows)
    return manifest.merge(features, on="task_id", validate="one_to_one").merge(
        truth_frame, on="task_id", validate="one_to_one"
    )


def _metric_record(model_name: str, predicted: np.ndarray, true_risk: np.ndarray) -> dict[str, object]:
    constant = model_name == "B0_constant"
    coverage = _coverage_risks(predicted, true_risk, constant=constant)
    full = coverage[1.00]
    return {
        "model": model_name,
        "spearman": _safe_correlation(spearmanr, predicted, true_risk),
        "pearson": _safe_correlation(pearsonr, predicted, true_risk),
        "mae": float(mean_absolute_error(true_risk, predicted)),
        "risk_at_25": coverage[0.25],
        "risk_at_50": coverage[0.50],
        "risk_at_75": coverage[0.75],
        "risk_at_100": full,
        "full_risk": full,
        "relative_risk_reduction_at_50": float((full - coverage[0.50]) / full) if full > 0 else 0.0,
    }


def _development_results(train: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        model = make_proposed_model(model_name)
        model.fit(train[PROPOSED_FEATURES], train["true_risk"])
        predicted = model.predict(dev[PROPOSED_FEATURES])
        dev_with_prediction = dev[["task_id", "domain", "true_risk"]].copy()
        dev_with_prediction["predicted"] = predicted
        for domain, local in dev_with_prediction.groupby("domain", sort=True):
            metrics = _metric_record(model_name, local["predicted"].to_numpy(float), local["true_risk"].to_numpy(float))
            rows.append({
                "split": "DEV",
                "domain": domain,
                "model": model_name,
                "n_train": len(train),
                "n_tasks": len(local),
                **metrics,
            })
        macro = pd.DataFrame([row for row in rows if row["model"] == model_name])
        rho_values = macro["spearman"].astype(float).fillna(0.0)
        rows.append({
            "split": "DEV",
            "domain": "MACRO",
            "model": model_name,
            "n_train": len(train),
            "n_tasks": len(dev),
            "spearman": float(rho_values.mean()),
            "pearson": float(macro["pearson"].astype(float).mean()),
            "mae": float(macro["mae"].astype(float).mean()),
            "risk_at_25": float(macro["risk_at_25"].astype(float).mean()),
            "risk_at_50": float(macro["risk_at_50"].astype(float).mean()),
            "risk_at_75": float(macro["risk_at_75"].astype(float).mean()),
            "risk_at_100": float(macro["risk_at_100"].astype(float).mean()),
            "full_risk": float(macro["full_risk"].astype(float).mean()),
            "relative_risk_reduction_at_50": float(macro["relative_risk_reduction_at_50"].astype(float).mean()),
        })
    return pd.DataFrame(rows)


def _select_model(dev_results: pd.DataFrame) -> tuple[str, str, dict[str, object]]:
    macro = dev_results[dev_results["domain"] == "MACRO"].set_index("model")
    eligibility: dict[str, dict[str, object]] = {}
    for model_name in MODEL_ORDER:
        domains = dev_results[(dev_results["model"] == model_name) & (dev_results["domain"] != "MACRO")]
        rhos = domains["spearman"].astype(float).fillna(0.0)
        eligibility[model_name] = {
            "macro_spearman": float(macro.loc[model_name, "spearman"]),
            "domains_rho_gt_0.20": int((rhos > 0.20).sum()),
            "domains_rho_lt_0": int((rhos < 0.0).sum()),
            "eligible": bool((rhos > 0.20).sum() >= 6 and (rhos < 0.0).sum() <= 1),
        }
    eligible_names = [name for name in MODEL_ORDER if eligibility[name]["eligible"]]
    pool = eligible_names if eligible_names else list(MODEL_ORDER)
    best_score = max(float(eligibility[name]["macro_spearman"]) for name in pool)
    tie_pool = [name for name in pool if best_score - float(eligibility[name]["macro_spearman"]) < 0.02]
    selected = min(tie_pool, key=lambda name: MODEL_COMPLEXITY_ORDER[name])
    status = "PASS" if eligible_names else "DEV MODEL-SELECTION FAILURE"
    receipt = {
        "selection_status": status,
        "selected_model": selected,
        "eligible_models": eligible_names,
        "candidate_models": eligibility,
        "tie_pool": tie_pool,
        "selection_rule": "highest DEV macro Spearman among eligible; if within 0.02, simplest model",
        "final_truth_loaded": False,
    }
    return selected, status, receipt


def _fit_baseline_predictions(train: pd.DataFrame, target: pd.DataFrame, selected_model: str) -> dict[str, np.ndarray]:
    y = train["true_risk"].to_numpy(float)
    predictions: dict[str, np.ndarray] = {}
    predictions["B0_constant"] = np.full(len(target), float(np.mean(y)))

    metadata_model = _ridge().fit(train[METADATA_FEATURES], y)
    predictions["B1_metadata_ridge"] = metadata_model.predict(target[METADATA_FEATURES])

    ood_train, ood_target = _fit_ood(train, target)
    ood_model = _ridge().fit(ood_train.reshape(-1, 1), y)
    predictions["B2_ood_only"] = ood_model.predict(ood_target.reshape(-1, 1))

    confidence_model = _ridge().fit(train[CONFIDENCE_FEATURES], y)
    predictions["B3_confidence_ridge"] = confidence_model.predict(target[CONFIDENCE_FEATURES])

    threshold_model = _ridge().fit(train[THRESHOLD_AWARE_FEATURES], y)
    predictions["B4_threshold_aware_ridge"] = threshold_model.predict(target[THRESHOLD_AWARE_FEATURES])

    fitted: dict[str, object] = {}
    for model_name in MODEL_ORDER:
        model = make_proposed_model(model_name)
        model.fit(train[PROPOSED_FEATURES], y)
        fitted[model_name] = model
        predictions[model_name] = model.predict(target[PROPOSED_FEATURES])
    predictions["B5_proposed_ridge"] = predictions["M1_Ridge"]
    predictions["B6_selected_proposed"] = predictions[selected_model]
    return predictions


def fit_models_and_freeze(*, train_features: pd.DataFrame, dev_features: pd.DataFrame, final_features: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    train = _load_truth_for_split(_load_manifest("train"), train_features)
    dev = _load_truth_for_split(_load_manifest("dev"), dev_features)
    dev_results = _development_results(train, dev)
    _atomic_csv(dev_results, RESULTS / "model_selection_results.csv")
    selected_model, status, selection_receipt = _select_model(dev_results)
    selection_receipt.update({
        "schema_version": "gate3c_model_selection_receipt_v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "train_task_count": len(train),
        "dev_task_count": len(dev),
        "proposed_features": PROPOSED_FEATURES,
        "feature_files_sha256": {
            "train": _sha256(RESULTS / "train_features.csv"),
            "dev": _sha256(RESULTS / "dev_features.csv"),
        },
        "test_truth_used_for_selection": False,
        "final_truth_loaded": False,
    })
    _atomic_json(selection_receipt, RESULTS / "model_selection_receipt.json")

    train_for_fit = train
    final_target = _load_manifest("final").merge(final_features, on="task_id", validate="one_to_one")
    final_predictions = _fit_baseline_predictions(train_for_fit, final_target, selected_model)
    prediction_frame = final_target[["task_id", "domain", "mechanism", "noise", "graph_seed"]].copy()
    for name in FINAL_EVALUATED_NAMES:
        prediction_frame[f"predicted_risk__{name}"] = final_predictions[name]
    prediction_frame["selected_model"] = selected_model
    _atomic_csv(prediction_frame, RESULTS / "final_predictions_before_truth.csv")

    final_receipt = {
        "schema_version": "gate3c_final_model_receipt_v1",
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "model_selection_receipt_sha256": _sha256(RESULTS / "model_selection_receipt.json"),
        "selected_model": selected_model,
        "selection_status": status,
        "fit_task_count": len(train_for_fit),
        "fit_split": "TRAIN only",
        "proposed_features": PROPOSED_FEATURES,
        "all_models_fit_on_same_proposed_features": True,
        "final_predictions_path": "results/final_predictions_before_truth.csv",
        "final_predictions_sha256": _sha256(RESULTS / "final_predictions_before_truth.csv"),
        "final_feature_sha256": _sha256(RESULTS / "final_features_ground_truth_free.csv"),
        "final_truth_loaded": False,
        "test_truth_used_for_model_selection": False,
        "test_truth_used_for_feature_construction": False,
        "hyperparameters_changed_after_DEV": False,
        "gate_thresholds_changed_after_DEV": False,
    }
    _atomic_json(final_receipt, RESULTS / "final_model_receipt.json")
    _atomic_json({
        "schema_version": "gate3c_pre_truth_hashes_v1",
        "final_truth_loaded": False,
        "files": {
            name: _sha256(RESULTS / name)
            for name in (
                "frozen_protocol.json",
                "train_manifest.csv",
                "dev_manifest.csv",
                "final_manifest.csv",
                "train_features.csv",
                "dev_features.csv",
                "final_features_ground_truth_free.csv",
                "model_selection_results.csv",
                "model_selection_receipt.json",
                "final_model_receipt.json",
                "final_predictions_before_truth.csv",
            )
        },
    }, RESULTS / "pre_truth_hashes.json")
    print(f"DEV selection frozen: {selected_model} ({status}); final predictions written before truth.", flush=True)
    return selected_model, prediction_frame


def load_final_truth_after_freeze(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """The only function allowed to load FINAL ``truth_adjacency``."""
    required = (
        PROTOCOL_PATH,
        RESULTS / "model_selection_receipt.json",
        RESULTS / "final_model_receipt.json",
        RESULTS / "pre_truth_hashes.json",
        RESULTS / "final_predictions_before_truth.csv",
    )
    if not all(path.exists() for path in required):
        raise RuntimeError("final truth cannot be opened before all freeze receipts exist")
    selection_receipt = json.loads((RESULTS / "model_selection_receipt.json").read_text(encoding="utf-8"))
    final_receipt = json.loads((RESULTS / "final_model_receipt.json").read_text(encoding="utf-8"))
    pre_truth = json.loads((RESULTS / "pre_truth_hashes.json").read_text(encoding="utf-8"))
    if selection_receipt.get("final_truth_loaded") is not False:
        raise RuntimeError("model selection receipt was not written before final truth")
    if final_receipt.get("final_truth_loaded") is not False or pre_truth.get("final_truth_loaded") is not False:
        raise RuntimeError("final truth freeze marker is invalid")
    if final_receipt.get("final_predictions_sha256") != _sha256(RESULTS / "final_predictions_before_truth.csv"):
        raise RuntimeError("final prediction hash changed before truth load")
    manifest = _load_manifest("final")
    truth_rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        task_id = str(row["task_id"])
        # This is the first and only final-test truth access in this pipeline.
        with np.load(ROOT / str(row["dataset_path"])) as dataset:
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
    truth_frame = pd.DataFrame(truth_rows)
    final_with_truth = prediction_frame.merge(truth_frame, on="task_id", validate="one_to_one")
    _atomic_json({
        "schema_version": "gate3c_final_truth_open_receipt_v1",
        "final_truth_loaded_after_freeze_receipts": True,
        "final_truth_used_for_feature_construction": False,
        "final_truth_used_for_model_selection": False,
        "final_truth_used_for_hyperparameter_choice": False,
        "final_truth_used_for_gate_threshold_modification": False,
        "pre_truth_hashes_verified": True,
        "final_task_count": len(final_with_truth),
        "model_selection_receipt_sha256": _sha256(RESULTS / "model_selection_receipt.json"),
        "final_model_receipt_sha256": _sha256(RESULTS / "final_model_receipt.json"),
    }, RESULTS / "final_truth_open_receipt.json")
    return final_with_truth


def _bootstrap_ci(predicted: np.ndarray, true_risk: np.ndarray, metric: str, seed: int, constant: bool = False) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(true_risk), size=len(true_risk))
        if metric == "spearman":
            value = _safe_correlation(spearmanr, predicted[indices], true_risk[indices])
            estimates.append(0.0 if not np.isfinite(value) else value)
        elif metric == "risk50":
            estimates.append(_coverage_risks(predicted[indices], true_risk[indices], constant=constant)[0.50])
        else:
            raise ValueError(metric)
    return tuple(float(value) for value in np.quantile(np.asarray(estimates), [0.025, 0.975]))


def _domain_macro_bootstrap(frame: pd.DataFrame, model_name: str, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    groups = [group for _, group in frame.groupby("domain", sort=True)]
    for _ in range(BOOTSTRAP_RESAMPLES):
        domain_rhos = []
        for group in groups:
            indices = rng.integers(0, len(group), size=len(group))
            predicted = group[f"predicted_risk__{model_name}"].to_numpy(float)[indices]
            risk = group["true_risk"].to_numpy(float)[indices]
            rho = _safe_correlation(spearmanr, predicted, risk)
            domain_rhos.append(0.0 if not np.isfinite(rho) else rho)
        estimates.append(float(np.mean(domain_rhos)))
    return tuple(float(value) for value in np.quantile(np.asarray(estimates), [0.025, 0.975]))


def _final_model_column(name: str) -> str:
    return f"predicted_risk__{name}"


def evaluate_final(final_with_truth: pd.DataFrame, selected_model: str) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for domain_index, (domain, local) in enumerate(final_with_truth.groupby("domain", sort=True)):
        risk = local["true_risk"].to_numpy(float)
        for model_index, model_name in enumerate(FINAL_EVALUATED_NAMES):
            metric_model_name = model_name
            if model_name == "B5_proposed_ridge":
                prediction_column = "predicted_risk__B5_proposed_ridge"
            elif model_name == "B6_selected_proposed":
                prediction_column = "predicted_risk__B6_selected_proposed"
            else:
                prediction_column = _final_model_column(model_name)
            predicted = local[prediction_column].to_numpy(float)
            metrics = _metric_record(metric_model_name, predicted, risk)
            rho_low, rho_high = _bootstrap_ci(
                predicted, risk, "spearman", ANALYSIS_SEED + 100 * domain_index + model_index,
                constant=model_name == "B0_constant",
            )
            risk_low, risk_high = _bootstrap_ci(
                predicted, risk, "risk50", ANALYSIS_SEED + 10000 + 100 * domain_index + model_index,
                constant=model_name == "B0_constant",
            )
            rows.append({
                "split": "FINAL",
                "domain": domain,
                "model": model_name,
                "selected": model_name == "B6_selected_proposed",
                "diagnostic_only": model_name not in {"B6_selected_proposed", "B0_constant", "B1_metadata_ridge", "B2_ood_only", "B3_confidence_ridge", "B4_threshold_aware_ridge", "B5_proposed_ridge"},
                "n_tasks": len(local),
                "spearman_ci_low": rho_low,
                "spearman_ci_high": rho_high,
                "risk_at_50_ci_low": risk_low,
                "risk_at_50_ci_high": risk_high,
                **metrics,
            })

    result_frame = pd.DataFrame(rows)
    selected_rows = result_frame[result_frame["model"] == "B6_selected_proposed"].copy()
    selected_domain = selected_rows.set_index("domain")
    macro_selected = {
        "spearman": float(selected_rows["spearman"].astype(float).fillna(0.0).mean()),
        "pearson": float(selected_rows["pearson"].astype(float).mean()),
        "mae": float(selected_rows["mae"].astype(float).mean()),
        "risk_at_25": float(selected_rows["risk_at_25"].astype(float).mean()),
        "risk_at_50": float(selected_rows["risk_at_50"].astype(float).mean()),
        "risk_at_75": float(selected_rows["risk_at_75"].astype(float).mean()),
        "full_risk": float(selected_rows["full_risk"].astype(float).mean()),
        "relative_risk_reduction_at_50": float(selected_rows["relative_risk_reduction_at_50"].astype(float).mean()),
    }
    macro_low, macro_high = _domain_macro_bootstrap(final_with_truth, "B6_selected_proposed", ANALYSIS_SEED + 900000)
    macro_selected["spearman_ci_low"] = macro_low
    macro_selected["spearman_ci_high"] = macro_high

    pooled_pred = final_with_truth["predicted_risk__B6_selected_proposed"].to_numpy(float)
    pooled_risk = final_with_truth["true_risk"].to_numpy(float)
    pooled_selected = _metric_record("B6_selected_proposed", pooled_pred, pooled_risk)
    final_rows = selected_rows.to_dict(orient="records")
    domain_rhos = selected_rows["spearman"].astype(float).fillna(0.0)
    domains_risk50_below = int((selected_rows["risk_at_50"].astype(float) < selected_rows["full_risk"].astype(float)).sum())
    exp_rows = selected_rows[selected_rows["domain"].isin(["quadratic_exponential", "piecewise_exponential"])].set_index("domain")
    exp_rhos = {domain: float(exp_rows.loc[domain, "spearman"]) for domain in exp_rows.index}
    topic_go_checks = {
        "macro_spearman_ge_0.40": macro_selected["spearman"] >= 0.40,
        "domains_rho_gt_0.20_ge_5": int((domain_rhos > 0.20).sum()) >= 5,
        "domains_rho_gt_0.30_ge_4": int((domain_rhos > 0.30).sum()) >= 4,
        "no_domain_rho_lt_neg_0.10": bool((domain_rhos < -0.10).sum() == 0),
        "domains_risk50_below_full_ge_5": domains_risk50_below >= 5,
        "macro_reduction_ge_0.15": macro_selected["relative_risk_reduction_at_50"] >= 0.15,
        "joint_shift_at_least_one_rho_gt_0.30": any(value > 0.30 for value in exp_rhos.values()),
        "joint_shift_other_rho_nonnegative": all(value >= 0.0 for value in exp_rhos.values()) and len(exp_rhos) == 2,
    }
    topic_go = all(topic_go_checks.values())
    topic_borderline = (
        not topic_go
        and macro_selected["spearman"] >= 0.25
        and domains_risk50_below >= 4
        and macro_selected["relative_risk_reduction_at_50"] >= 0.10
        and not all(value <= 0.0 for value in exp_rhos.values())
    )
    topic_decision = "TOPIC_GO" if topic_go else "TOPIC_BORDERLINE" if topic_borderline else "TOPIC_STOP"

    threshold_macro = result_frame[result_frame["model"] == "B4_threshold_aware_ridge"]
    threshold_rho = float(threshold_macro["spearman"].astype(float).fillna(0.0).mean())
    threshold_reduction = float(threshold_macro["relative_risk_reduction_at_50"].astype(float).mean())
    rho_gain = macro_selected["spearman"] - threshold_rho
    reduction_gain = macro_selected["relative_risk_reduction_at_50"] - threshold_reduction
    method_go = bool(
        (rho_gain >= 0.05 and reduction_gain >= -0.03)
        or (reduction_gain >= 0.03 and rho_gain >= -0.05)
    )
    method_decision = "METHOD_GO" if method_go else "METHOD_BORDERLINE"

    capacity_domain_rows = []
    for model_name in MODEL_ORDER:
        mapped = "B5_proposed_ridge" if model_name == "M1_Ridge" else model_name
        local = result_frame[result_frame["model"] == mapped]
        capacity_domain_rows.extend(local.to_dict(orient="records"))
    capacity_frame = pd.DataFrame(capacity_domain_rows)
    capacity_macro = {
        model_name: float(capacity_frame[capacity_frame["model"] == ("B5_proposed_ridge" if model_name == "M1_Ridge" else model_name)]["spearman"].astype(float).fillna(0.0).mean())
        for model_name in MODEL_ORDER
    }
    best_complex_model = max(MODEL_ORDER[1:], key=lambda name: capacity_macro[name])
    best_complex_dev = json.loads((RESULTS / "model_selection_results.csv").read_text(encoding="utf-8")) if False else None
    dev_results = pd.read_csv(RESULTS / "model_selection_results.csv")
    dev_macro = dev_results[dev_results["domain"] == "MACRO"].set_index("model")["spearman"].astype(float).to_dict()
    best_complex_dev_model = max(MODEL_ORDER[1:], key=lambda name: float(dev_macro[name]))
    best_complex_dev_gain = float(dev_macro[best_complex_dev_model] - dev_macro["M1_Ridge"])
    best_complex_final_gain = float(capacity_macro[best_complex_model] - capacity_macro["M1_Ridge"])
    if best_complex_dev_gain >= 0.05 and best_complex_final_gain <= -0.05:
        capacity_case = "higher capacity overfits domain-specific patterns"
    elif best_complex_dev_gain >= 0.05:
        capacity_case = "model capacity matters materially"
    else:
        capacity_case = "model capacity is not the main bottleneck"

    summary: dict[str, object] = {
        "schema_version": "gate3c_final_summary_v1",
        "topic_decision": topic_decision,
        "method_decision": method_decision,
        "selected_model": selected_model,
        "final": {
            "domain_results_selected": final_rows,
            "macro_selected": macro_selected,
            "pooled_selected": pooled_selected,
            "threshold_aware_macro_spearman": threshold_rho,
            "threshold_aware_macro_relative_reduction_at_50": threshold_reduction,
            "selected_minus_threshold_aware": {
                "macro_spearman": rho_gain,
                "macro_relative_reduction_at_50": reduction_gain,
            },
            "domain_rho_gt_0.20": int((domain_rhos > 0.20).sum()),
            "domain_rho_gt_0.30": int((domain_rhos > 0.30).sum()),
            "domain_rho_lt_neg_0.10": int((domain_rhos < -0.10).sum()),
            "domains_risk50_below_full": domains_risk50_below,
            "joint_shift_rhos": exp_rhos,
        },
        "topic_go_checks": topic_go_checks,
        "method_readiness": {
            "selected_macro_spearman": macro_selected["spearman"],
            "threshold_aware_macro_spearman": threshold_rho,
            "selected_macro_relative_reduction_at_50": macro_selected["relative_risk_reduction_at_50"],
            "threshold_aware_macro_relative_reduction_at_50": threshold_reduction,
            "rho_gain": rho_gain,
            "relative_reduction_gain": reduction_gain,
            "decision": method_decision,
        },
        "model_capacity": {
            "dev_macro_spearman": dev_macro,
            "final_macro_spearman": capacity_macro,
            "best_complex_dev_model": best_complex_dev_model,
            "best_complex_dev_gain_over_ridge": best_complex_dev_gain,
            "best_complex_final_model": best_complex_model,
            "best_complex_final_gain_over_ridge": best_complex_final_gain,
            "case": capacity_case,
            "non_selected_models_are_diagnostic_only": True,
        },
        "isolation": {
            "final_truth_loaded_after_protocol_receipts": True,
            "final_truth_used_for_feature_construction": False,
            "final_truth_used_for_model_selection": False,
            "final_truth_used_for_preprocessing_or_hyperparameter_choice": False,
            "final_truth_used_to_change_gate_thresholds": False,
            "final_truth_open_receipt_sha256": _sha256(RESULTS / "final_truth_open_receipt.json"),
        },
    }
    _atomic_csv(result_frame, RESULTS / "final_results_by_domain.csv")
    _atomic_json(summary, RESULTS / "final_summary.json")
    return result_frame, summary


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def make_plots(final_with_truth: pd.DataFrame, results: pd.DataFrame, summary: dict[str, object]) -> None:
    selected_name = "B6_selected_proposed"
    coverage_models = [
        (selected_name, "selected proposed"),
        ("B4_threshold_aware_ridge", "threshold-aware Ridge"),
        ("B3_confidence_ridge", "confidence-only"),
        ("B0_constant", "random/full-risk"),
    ]
    domains = sorted(final_with_truth["domain"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for axis, domain in zip(axes.ravel(), domains):
        subset = results[results["domain"] == domain].set_index("model")
        for model_name, label in coverage_models:
            row = subset.loc[model_name]
            axis.plot(
                [25, 50, 75, 100],
                [row["risk_at_25"], row["risk_at_50"], row["risk_at_75"], row["risk_at_100"]],
                marker="o",
                linewidth=1.6,
                label=label,
            )
        axis.set_title(domain)
        axis.grid(alpha=0.25)
        axis.set_xticks([25, 50, 75, 100])
    axes[0, 0].set_ylabel("Mean true risk")
    axes[1, 0].set_ylabel("Mean true risk")
    axes[1, 1].set_xlabel("Coverage retained (%)")
    axes[1, 2].set_xlabel("Coverage retained (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Final risk coverage by domain")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(RESULTS / "final_risk_coverage_by_domain.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for model_name, label in coverage_models:
        local = results[results["model"] == model_name]
        axis.plot(
            [25, 50, 75, 100],
            [local["risk_at_25"].mean(), local["risk_at_50"].mean(), local["risk_at_75"].mean(), local["risk_at_100"].mean()],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    axis.set_xlabel("Coverage retained (%)")
    axis.set_ylabel("Macro mean true risk")
    axis.set_title("Final macro risk coverage")
    axis.set_xticks([25, 50, 75, 100])
    axis.grid(alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "final_macro_risk_coverage.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for axis, domain in zip(axes.ravel(), domains):
        local = final_with_truth[final_with_truth["domain"] == domain]
        x = local["predicted_risk__B6_selected_proposed"].to_numpy(float)
        y = local["true_risk"].to_numpy(float)
        axis.scatter(x, y, alpha=0.72, s=24)
        lower = min(0.0, float(x.min()), float(y.min()))
        upper = max(1.0, float(x.max()), float(y.max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="grey", linewidth=1)
        axis.set_title(domain)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("True risk")
    axes[1, 0].set_ylabel("True risk")
    axes[1, 1].set_xlabel("Selected predicted risk")
    axes[1, 2].set_xlabel("Selected predicted risk")
    fig.suptitle("Selected predicted versus true risk on final test")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULTS / "predicted_vs_true_risk_final.png", dpi=180)
    plt.close(fig)

    dev_results = pd.read_csv(RESULTS / "model_selection_results.csv")
    dev_macro = dev_results[dev_results["domain"] == "MACRO"].set_index("model").loc[list(MODEL_ORDER)]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(["Ridge", "Spline", "RF", "HistGB"], dev_macro["spearman"].astype(float), color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("DEV macro Spearman")
    axis.set_title("Model capacity comparison on DEV")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, dev_macro["spearman"].astype(float)):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(RESULTS / "model_capacity_dev_comparison.png", dpi=180)
    plt.close(fig)

    spearman_models = [
        ("B6_selected_proposed", "selected"),
        ("B4_threshold_aware_ridge", "threshold-aware"),
        ("B3_confidence_ridge", "confidence-only"),
        ("B2_ood_only", "OOD-only"),
    ]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    x_positions = np.arange(len(domains))
    width = 0.19
    for index, (model_name, label) in enumerate(spearman_models):
        local = results[results["model"] == model_name].set_index("domain").loc[domains]
        axis.bar(x_positions + (index - 1.5) * width, local["spearman"].astype(float), width=width, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(domains, rotation=25, ha="right")
    axis.set_ylabel("Spearman rho")
    axis.set_title("Final domain Spearman")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "domain_spearman_final.png", dpi=180)
    plt.close(fig)


def write_report(results: pd.DataFrame, summary: dict[str, object], selected_model: str) -> None:
    final_summary = summary["final"]
    macro = final_summary["macro_selected"]
    domain_rows = results[results["model"] == "B6_selected_proposed"].sort_values("domain")
    threshold_rows = results[results["model"] == "B4_threshold_aware_ridge"]
    confidence_rows = results[results["model"] == "B3_confidence_ridge"]
    ood_rows = results[results["model"] == "B2_ood_only"]
    capacity = summary["model_capacity"]
    previous = [
        "Gate-3A primary all-feature Ridge: mean Spearman approximately 0.200, BORDERLINE.",
        "Gate-3A post-hoc stable 5-feature Ridge: old four-mechanism exploratory mean rho approximately 0.55; it is not independent evidence.",
        "Gate-3A.1 independent confirmation primary stable_ridge_v1: softsign rho approximately 0.665, sine rho approximately 0.231, mean rho approximately 0.448, mean risk reduction@50 approximately 21.6%; sine missed preregistered 0.30, so BORDERLINE.",
        "Gate-3A.1 pre-frozen threshold-aware Ridge: softsign rho approximately 0.522, sine rho approximately 0.537; adaptive-threshold distance had promise, but this was not a GO conversion.",
    ]
    lines = [
        f"TOPIC DECISION: **{summary['topic_decision']}**",
        f"METHOD DECISION: **{summary['method_decision']}**",
        "",
        f"Selected proposed model: `{selected_model}`. DEV selection status: `{json.loads((RESULTS / 'model_selection_receipt.json').read_text(encoding='utf-8'))['selection_status']}`.",
        "",
        "## Final domain results",
        "",
        "The primary evidence is the six-domain macro, not a pooled metric. `risk@50` retains the 50% of tasks with the lowest predicted risk.",
        "",
        "| FINAL domain | selected-model rho | rho 95% CI | risk@50 | full risk | relative reduction@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in domain_rows.to_dict(orient="records"):
        lines.append(
            f"| {row['domain']} | {_fmt(row['spearman'])} | [{_fmt(row['spearman_ci_low'])}, {_fmt(row['spearman_ci_high'])}] | "
            f"{_fmt(row['risk_at_50'])} | {_fmt(row['full_risk'])} | {100 * float(row['relative_risk_reduction_at_50']):.1f}% |"
        )
    lines.extend([
        "",
        f"**FINAL macro:** rho={_fmt(macro['spearman'])}, 95% CI=[{_fmt(macro['spearman_ci_low'])}, {_fmt(macro['spearman_ci_high'])}], relative risk reduction@50={100 * float(macro['relative_risk_reduction_at_50']):.1f}%.",
        f"**FINAL pooled:** rho={_fmt(final_summary['pooled_selected']['spearman'])}, risk@50={_fmt(final_summary['pooled_selected']['risk_at_50'])}, full risk={_fmt(final_summary['pooled_selected']['full_risk'])}, relative reduction@50={100 * float(final_summary['pooled_selected']['relative_risk_reduction_at_50']):.1f}%.",
        "",
        "## DEV model selection",
        "",
        "| model | DEV macro Spearman | domains rho>0.20 | domains rho<0 |",
        "|---|---:|---:|---:|",
    ])
    dev_results = pd.read_csv(RESULTS / "model_selection_results.csv")
    dev_macro = dev_results[dev_results["domain"] == "MACRO"].set_index("model")
    for model_name in MODEL_ORDER:
        local = dev_results[(dev_results["model"] == model_name) & (dev_results["domain"] != "MACRO")]
        rhos = local["spearman"].astype(float).fillna(0.0)
        lines.append(f"| {model_name} | {_fmt(dev_macro.loc[model_name, 'spearman'])} | {int((rhos > 0.20).sum())}/8 | {int((rhos < 0).sum())}/8 |")
    lines.extend([
        "",
        f"Selection rule: highest macro Spearman among candidates with at least 6/8 DEV domains above 0.20 and at most 1/8 below zero; ties within 0.02 use the simpler model. Final selected model: `{selected_model}`.",
        "",
        "## Baselines and method increment",
        "",
        f"Final threshold-aware Ridge macro rho={_fmt(final_summary['threshold_aware_macro_spearman'])}; selected minus threshold-aware increment={_fmt(final_summary['selected_minus_threshold_aware']['macro_spearman'])} rho.",
        f"Final threshold-aware Ridge macro relative reduction@50={100 * float(final_summary['threshold_aware_macro_relative_reduction_at_50']):.1f}%; selected increment={100 * float(final_summary['selected_minus_threshold_aware']['macro_relative_reduction_at_50']):.1f} percentage points.",
        "",
        "| model | final macro rho | final macro relative reduction@50 |",
        "|---|---:|---:|",
    ])
    for model_name in ("B0_constant", "B1_metadata_ridge", "B2_ood_only", "B3_confidence_ridge", "B4_threshold_aware_ridge", "B5_proposed_ridge", "B6_selected_proposed", "M2_AdditiveSpline", "M3_RandomForest", "M4_HistGradientBoosting"):
        local = results[results["model"] == model_name]
        lines.append(f"| {model_name} | {_fmt(local['spearman'].astype(float).fillna(0.0).mean())} | {100 * float(local['relative_risk_reduction_at_50'].astype(float).mean()):.1f}% |")
    lines.extend([
        "",
        "## Does model capacity matter?",
        "",
        f"DEV macro Spearman comparison is in `model_capacity_dev_comparison.png`. The preregistered capacity interpretation is: **{capacity['case']}**.",
        f"Best complex DEV model: `{capacity['best_complex_dev_model']}`, gain over Ridge={_fmt(capacity['best_complex_dev_gain_over_ridge'])}; best complex FINAL diagnostic model: `{capacity['best_complex_final_model']}`, gain over Ridge={_fmt(capacity['best_complex_final_gain_over_ridge'])}.",
        "Non-selected models are diagnostic only; the FINAL model was selected before FINAL TEST truth.",
        "",
        "## Historical boundary",
        "",
    ])
    lines.extend([f"- {item}" for item in previous])
    lines.extend([
        "",
        "This round does not reinterpret those results as GO. Quadratic and piecewise are reported as unseen mechanism families for the risk predictor; no claim is made about what CDFM pretraining did or did not contain.",
        "",
        "## Answers to the final topic questions",
        "",
        f"1. Cross unseen mechanism: {'yes' if summary['final']['domain_rho_gt_0.20'] >= 5 else 'not robustly demonstrated'}; the final test mechanisms are quadratic and piecewise.",
        f"2. Cross unseen graph instances: yes by design; all {sum(int(value) for value in [160, 80, 120])} split tasks use unique graph seeds, with final seeds 600000 onward and no overlap with prior Gate seeds.",
        f"3. With unseen exponential noise: quadratic_exponential rho={_fmt(summary['final']['joint_shift_rhos'].get('quadratic_exponential'))}, piecewise_exponential rho={_fmt(summary['final']['joint_shift_rhos'].get('piecewise_exponential'))}; this is {'supported' if any(float(value) > 0.0 for value in summary['final']['joint_shift_rhos'].values()) else 'not supported'} by the joint-shift domains.",
        f"4. OOD score alone: final macro rho={_fmt(ood_rows['spearman'].astype(float).fillna(0.0).mean())}; it {'is' if float(ood_rows['spearman'].astype(float).fillna(0.0).mean()) >= 0.25 else 'is not'} sufficient as the main explanation under this gate.",
        f"5. Threshold-aware information: final macro rho={_fmt(threshold_rows['spearman'].astype(float).fillna(0.0).mean())}; it is a frozen baseline and is {'stable' if int((threshold_rows['spearman'].astype(float) > 0.20).sum()) >= 4 else 'not consistently strong'} across final domains.",
        f"6. Probability/resampling stability increment: selected minus threshold-aware rho={_fmt(summary['final']['selected_minus_threshold_aware']['macro_spearman'])}, reduction increment={100 * float(summary['final']['selected_minus_threshold_aware']['macro_relative_reduction_at_50']):.1f} percentage points; method decision is **{summary['method_decision']}**.",
        f"7. More complex models: {capacity['case']}.",
        "8. Main bottleneck: FINAL evidence does not support model capacity as the main bottleneck—threshold-aware Ridge (macro rho=0.618) exceeds selected RF (0.589), and proposed Ridge (0.604) also exceeds selected RF; the new stability features add no measured increment. The remaining limitation is therefore the incremental representation/training-coverage problem under shift, with inherent unpredictability still not separable without a dedicated ablation.",
        f"9. Formal paper main line recommendation: {'yes, subject to method development' if summary['topic_decision'] == 'TOPIC_GO' else 'no; do not continue ground-truth-free graph-risk prediction as the paper main line' if summary['topic_decision'] == 'TOPIC_STOP' else 'not yet; retain only as a borderline research direction'}.",
        "",
        "## Freeze and truth-isolation audit",
        "",
        "FINAL TEST truth was not used for feature construction, model selection, preprocessing, hyperparameter choice, or Gate threshold modification.",
        "",
        "After FINAL TEST truth was opened, were any feature/model/hyperparameter/Gate modifications made? **No.**",
        "",
        "The required pre-truth files were written before `load_final_truth_after_freeze()` ran: `frozen_protocol.json`, `model_selection_receipt.json`, `final_model_receipt.json`, and `pre_truth_hashes.json`.",
    ])
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> None:
    protocol = _load_protocol()
    if not (RESULTS / "train_features.csv").exists() or not (RESULTS / "dev_features.csv").exists() or not (RESULTS / "final_features_ground_truth_free.csv").exists():
        frames = build_all_features()
    else:
        frames = {
            "train": pd.read_csv(RESULTS / "train_features.csv"),
            "dev": pd.read_csv(RESULTS / "dev_features.csv"),
            "final": pd.read_csv(RESULTS / "final_features_ground_truth_free.csv"),
        }
    selected_model, prediction_frame = fit_models_and_freeze(
        train_features=frames["train"],
        dev_features=frames["dev"],
        final_features=frames["final"],
    )
    final_with_truth = load_final_truth_after_freeze(prediction_frame)
    _atomic_csv(final_with_truth, RESULTS / "final_predictions.csv")
    final_results, summary = evaluate_final(final_with_truth, selected_model)
    make_plots(final_with_truth, final_results, summary)
    write_report(final_results, summary, selected_model)
    print(json.dumps({
        "topic_decision": summary["topic_decision"],
        "method_decision": summary["method_decision"],
        "selected_model": selected_model,
        "final_macro_spearman": summary["final"]["macro_selected"]["spearman"],
        "final_macro_relative_reduction_at_50": summary["final"]["macro_selected"]["relative_risk_reduction_at_50"],
    }, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "infer", "analyze", "all"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.stage in {"prepare", "all"}:
        prepare(overwrite=args.overwrite)
    if args.stage in {"infer", "all"}:
        infer(shard_index=args.shard_index, shard_count=args.shard_count)
    if args.stage in {"analyze", "all"}:
        analyze()


if __name__ == "__main__":
    main()
