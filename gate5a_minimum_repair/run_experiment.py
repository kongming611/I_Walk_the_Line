"""Gate-5A: ground-truth-free graph-error lower-bound certification.

The implementation is intentionally isolated from the historical Gate
directories.  CDFM sees only the discovery half of each synthetic task.  CI
tests and all certificate construction use only the validation half.  The
truth graph is loaded by a separate final-stage function after the pre-truth
receipt exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE2_ROOT = REPO_ROOT / "gate2_reliable_routing"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(GATE2_ROOT) not in sys.path:
    sys.path.insert(0, str(GATE2_ROOT))

from gate2_reliable_routing.generate_tasks import (  # noqa: E402
    N_VARIABLES,
    RFF_FEATURES,
    _edge_transform,
    _graph,
    _noise,
    _normalize,
    _rng,
)


ANALYSIS_SEED = 20260921
N_SAMPLES_TOTAL = 1000
N_DISCOVERY = 500
N_VALIDATION = 500
N_DOMAINS = 6
TASKS_PER_DOMAIN = 20
GRAPH_SEED_START = 850000
HOLM_ALPHA = 0.05
MAX_REPAIR_K = 3
RCIT_NUM_F = 100
RCIT_NUM_F2 = 5

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASETS = CACHE / "datasets"
RAW = CACHE / "raw_predictions"
TASK_WITNESSES = CACHE / "witnesses"
TASK_CERTIFICATES = CACHE / "certificates"
PROTOCOL_PATH = RESULTS / "frozen_protocol.json"
MANIFEST_PATH = RESULTS / "manifest.csv"


# Six fixed families.  The first is a linear SCM obtained by setting the
# nonlinear mixture weight to zero; the remaining families exercise distinct
# nonlinear/noise combinations already supported by the repository generator.
DOMAINS: tuple[dict[str, object], ...] = (
    {"domain": "linear_gaussian", "generator_mechanism": "rff", "noise": "gaussian", "lambda": 0.0},
    {"domain": "rff_studentt3", "generator_mechanism": "rff", "noise": "student_t3", "lambda": 1.0},
    {"domain": "tanh_laplace", "generator_mechanism": "tanh", "noise": "laplace", "lambda": 1.0},
    {"domain": "softsign_studentt8", "generator_mechanism": "softsign", "noise": "student_t8", "lambda": 1.0},
    {"domain": "interaction_gaussian", "generator_mechanism": "interaction", "noise": "gaussian", "lambda": 1.0},
    {"domain": "sine_exponential", "generator_mechanism": "sine", "noise": "exponential", "lambda": 1.0},
)


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


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
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


def _repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _append_error(task_id: str, stage: str, exc: BaseException) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "errors.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp_epoch": time.time(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _is_dag(adjacency: np.ndarray) -> bool:
    adjacency = (np.asarray(adjacency) != 0).astype(np.int8)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return False
    if np.any(np.diag(adjacency)):
        return False
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


def _edge_count(adjacency: np.ndarray) -> int:
    return int(np.asarray(adjacency, dtype=np.int8).sum())


def _generate_raw_split(
    spec: object,
    generator_mechanism: str,
    noise_kind: str,
    lambda_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a task and standardize using discovery statistics only."""
    rng = _rng(seed, 2)
    exogenous = _noise(noise_kind, rng)
    x = np.zeros((N_SAMPLES_TOTAL, N_VARIABLES), dtype=np.float64)
    incoming = {node: np.flatnonzero(spec.targets == node) for node in range(N_VARIABLES)}
    for child_value in spec.order:
        child = int(child_value)
        indices = incoming[child]
        if len(indices) == 0:
            x[:, child] = exogenous[:, child]
            continue
        linear = np.zeros(N_SAMPLES_TOTAL)
        nonlinear = np.zeros(N_SAMPLES_TOTAL)
        parents: list[np.ndarray] = []
        for edge_index in indices:
            source = int(spec.sources[edge_index])
            parents.append(x[:, source])
            weight = float(spec.weights[edge_index])
            linear += weight * x[:, source]
            nonlinear += weight * _edge_transform(
                generator_mechanism,
                x[:, source],
                spec.rff_omega[edge_index],
                spec.rff_phase[edge_index],
                spec.rff_coeff[edge_index],
                float(spec.slopes[edge_index]),
                float(spec.biases[edge_index]),
            )
        if generator_mechanism == "interaction" and len(parents) >= 2:
            for left in range(len(parents)):
                for right in range(left + 1, len(parents)):
                    nonlinear += 0.35 * np.tanh(parents[left]) * np.tanh(parents[right])
        x[:, child] = (
            (1.0 - lambda_value) * _normalize(linear)
            + lambda_value * _normalize(nonlinear)
            + exogenous[:, child]
        )
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) <= 1e-12):
        raise ValueError("generated X is invalid")

    discovery = x[:N_DISCOVERY]
    validation = x[N_DISCOVERY:]
    mean = discovery.mean(axis=0)
    scale = discovery.std(axis=0)
    if np.any(scale <= 1e-12) or not np.isfinite(scale).all():
        raise ValueError("discovery standardization is invalid")
    discovery = (discovery - mean) / scale
    validation = (validation - mean) / scale
    return discovery.astype(np.float32), validation.astype(np.float32)


def _write_dataset(path: Path, discovery: np.ndarray, validation: np.ndarray, truth: np.ndarray, graph_seed: int) -> None:
    if path.exists():
        with np.load(path) as existing:
            if int(existing["graph_seed"]) != graph_seed:
                raise RuntimeError(f"graph seed mismatch in {path}")
            if existing["X_discovery"].shape != (N_DISCOVERY, N_VARIABLES):
                raise RuntimeError(f"unexpected discovery shape in {path}")
            if existing["X_validation"].shape != (N_VALIDATION, N_VARIABLES):
                raise RuntimeError(f"unexpected validation shape in {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            X_discovery=np.asarray(discovery, dtype=np.float32),
            X_validation=np.asarray(validation, dtype=np.float32),
            truth_adjacency=np.asarray(truth, dtype=np.int8),
            graph_seed=np.int64(graph_seed),
        )
    os.replace(temporary, path)


def _domain_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for domain_spec in DOMAINS:
        for task_index in range(TASKS_PER_DOMAIN):
            graph_seed = GRAPH_SEED_START + ordinal
            task_id = f"{domain_spec['domain']}_{task_index:03d}"
            dataset_path = DATASETS / f"{task_id}.npz"
            spec = _graph(graph_seed)
            discovery, validation = _generate_raw_split(
                spec,
                str(domain_spec["generator_mechanism"]),
                str(domain_spec["noise"]),
                float(domain_spec["lambda"]),
                graph_seed,
            )
            _write_dataset(dataset_path, discovery, validation, spec.adjacency, graph_seed)
            rows.append(
                {
                    "task_id": task_id,
                    "domain": str(domain_spec["domain"]),
                    "generator_mechanism": str(domain_spec["generator_mechanism"]),
                    "noise": str(domain_spec["noise"]),
                    "lambda": float(domain_spec["lambda"]),
                    "task_index": task_index,
                    "graph_seed": graph_seed,
                    "n_total": N_SAMPLES_TOTAL,
                    "n_discovery": N_DISCOVERY,
                    "n_validation": N_VALIDATION,
                    "n_variables": N_VARIABLES,
                    "dataset_path": _repo_relative(dataset_path),
                    "raw_path": _repo_relative(RAW / f"{task_id}.npz"),
                    "witness_path": _repo_relative(TASK_WITNESSES / f"{task_id}.json"),
                    "certificate_path": _repo_relative(TASK_CERTIFICATES / f"{task_id}.json"),
                }
            )
            ordinal += 1
    return rows


def protocol_payload() -> dict[str, object]:
    return {
        "schema_version": "gate5a_minimum_repair_protocol_v1",
        "freeze_date": "2026-09-21",
        "scientific_question": "Can held-out causal contradictions produce a non-vacuous ground-truth-free lower bound on CDFM graph error?",
        "formal_target": "ground-truth-free minimum structural repair / graph-error lower-bound certificate",
        "data": {
            "domains": [dict(item) for item in DOMAINS],
            "domain_count": N_DOMAINS,
            "tasks_per_domain": TASKS_PER_DOMAIN,
            "task_count": N_DOMAINS * TASKS_PER_DOMAIN,
            "D": N_VARIABLES,
            "N_total": N_SAMPLES_TOTAL,
            "N_discovery": N_DISCOVERY,
            "N_validation": N_VALIDATION,
            "split": "first 500 rows discovery, last 500 rows validation; standardization fit on discovery only",
            "graph_seed_range": [GRAPH_SEED_START, GRAPH_SEED_START + N_DOMAINS * TASKS_PER_DOMAIN - 1],
        },
        "cdfm": {
            "model": "DMIRLAB/CDFM",
            "input": "X_discovery only",
            "training": False,
            "fine_tuning": False,
            "source_modification": False,
        },
        "candidate_graph": {
            "raw_output": "CDFM adjacency and edge probabilities from X_discovery",
            "projection": "For each unordered pair retain the selected orientation with larger probability; sort retained edges by descending probability and greedily reject any edge that creates a directed cycle.",
            "tie_break": "higher probability, then smaller source index, then smaller target index",
            "candidate_is_dag": True,
            "validation_or_truth_used": False,
        },
        "witnesses": {
            "statements": "For every candidate node Y, test Y independent of each non-descendant non-parent X given Pa_candidate(Y).",
            "ci_test": "causal-learn RCIT",
            "rcit_parameters": {"approx": "lpd4", "num_f": RCIT_NUM_F, "num_f2": RCIT_NUM_F2, "rcit": True},
            "test_data": "X_validation only",
            "multiple_testing": "Holm step-down over every local-Markov statement generated for one task",
            "alpha": HOLM_ALPHA,
            "witness": "a statement with Holm-adjusted rejection at alpha",
        },
        "repair": {
            "distance": "directed pairwise SHD; add, delete, or reverse one unordered-pair state costs one",
            "space": "DAGs on the same ten observed nodes",
            "search": "exhaustive pair-state enumeration with DAG pruning for k=0,1,2,3",
            "no_solution_by_max_k": "report RR_lower_bound=4 and rr_exact=false; claim only RR >= 4",
            "witness_resolution": "a repaired DAG resolves a witness only when it does not d-separate the tested pair given the fixed candidate parent set",
        },
        "evaluation": {
            "truth": "opened only after certificate predictions and pre_truth_hashes.json",
            "true_error": "SHD(candidate_DAG, ground_truth_DAG)",
            "validity": "RR_lower_bound <= true SHD",
            "bad_graph": "true SHD >= 1",
            "severe_bad_graph": "true SHD >= 4",
            "tightness": "RR_lower_bound / true SHD for true SHD > 0",
        },
        "gate": {
            "GO": {
                "validity_rate_min": 0.95,
                "bad_graph_nonzero_rate_min": 0.40,
                "severe_graph_rr_ge_2_rate_min": 0.50,
            },
            "BORDERLINE": "validity passes but one or more non-vacuity conditions fail",
            "STOP": "validity fails, including any frequent RR_lower_bound > true SHD pattern",
            "INCONCLUSIVE": "technical failure rate exceeds 10 percent or an evaluation denominator is empty",
        },
        "truth_isolation": {
            "cdfm_input_is_discovery_only": True,
            "witness_builder_accepts_truth": False,
            "certificate_builder_accepts_truth": False,
            "final_truth_loaded_before_certificate": False,
            "final_truth_loader": "open_final_truth called only after pre_truth_hashes.json",
        },
    }


def prepare() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    TASK_WITNESSES.mkdir(parents=True, exist_ok=True)
    TASK_CERTIFICATES.mkdir(parents=True, exist_ok=True)
    atomic_json(protocol_payload(), PROTOCOL_PATH)
    rows = _domain_rows()
    manifest = pd.DataFrame(rows)
    if len(manifest) != N_DOMAINS * TASKS_PER_DOMAIN:
        raise AssertionError("unexpected task count")
    if manifest["graph_seed"].duplicated().any():
        raise AssertionError("graph seeds are not unique")
    atomic_csv(manifest, MANIFEST_PATH)
    atomic_json(
        {
            "schema_version": "gate5a_prepare_receipt_v1",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "task_count": len(manifest),
            "domain_counts": manifest.groupby("domain").size().to_dict(),
            "all_graph_seeds_unique": True,
            "truth_graph_written_but_not_loaded": True,
            "cdfm_input_keys": ["X_discovery"],
            "certificate_input_keys": ["X_validation", "candidate_adjacency", "rejected_witnesses"],
        },
        RESULTS / "prepare_receipt.json",
    )
    print(f"prepared Gate-5A: {len(manifest)} tasks across {len(DOMAINS)} domains", flush=True)


def _load_protocol() -> dict[str, object]:
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload["schema_version"] != "gate5a_minimum_repair_protocol_v1":
        raise RuntimeError("unexpected Gate-5A protocol")
    return payload


def _load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("missing manifest; run --stage prepare first")
    return pd.read_csv(MANIFEST_PATH)


def _load_discovery(row: dict[str, object]) -> np.ndarray:
    with np.load(REPO_ROOT / str(row["dataset_path"])) as data:
        return np.asarray(data["X_discovery"], dtype=np.float64)


def _load_validation(row: dict[str, object]) -> np.ndarray:
    with np.load(REPO_ROOT / str(row["dataset_path"])) as data:
        return np.asarray(data["X_validation"], dtype=np.float64)


def _edge_state(graph: np.ndarray, left: int, right: int) -> int:
    if graph[left, right] and not graph[right, left]:
        return 1
    if graph[right, left] and not graph[left, right]:
        return 2
    if not graph[left, right] and not graph[right, left]:
        return 0
    raise ValueError("candidate graph contains a reciprocal pair")


def _set_edge_state(graph: np.ndarray, left: int, right: int, state: int) -> None:
    graph[left, right] = 0
    graph[right, left] = 0
    if state == 1:
        graph[left, right] = 1
    elif state == 2:
        graph[right, left] = 1
    elif state != 0:
        raise ValueError(f"invalid edge state {state}")


def project_to_dag(raw_adjacency: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Project raw CDFM selected edges to a deterministic discovery-only DAG."""
    raw = (np.asarray(raw_adjacency) != 0).astype(np.int8)
    probs = np.asarray(probabilities, dtype=float)
    if raw.shape != (N_VARIABLES, N_VARIABLES) or probs.shape != raw.shape:
        raise ValueError("CDFM graph arrays have unexpected shape")
    if np.any(np.diag(raw)):
        raise ValueError("CDFM returned a self loop")

    selected: list[tuple[float, int, int]] = []
    for left in range(N_VARIABLES):
        for right in range(left + 1, N_VARIABLES):
            choices: list[tuple[float, int, int]] = []
            if raw[left, right]:
                choices.append((float(probs[left, right]), left, right))
            if raw[right, left]:
                choices.append((float(probs[right, left]), right, left))
            if choices:
                # max() with this key implements the frozen tie-break rule.
                selected.append(max(choices, key=lambda item: (item[0], -item[1], -item[2])))

    selected.sort(key=lambda item: (-item[0], item[1], item[2]))
    candidate = np.zeros((N_VARIABLES, N_VARIABLES), dtype=np.int8)
    for _, source, target in selected:
        trial = candidate.copy()
        trial[source, target] = 1
        if _is_dag(trial):
            candidate = trial
    if not _is_dag(candidate):
        raise AssertionError("DAG projection returned a cyclic graph")
    return candidate


def _predict_and_save(model: object, x: np.ndarray, path: Path) -> dict[str, object]:
    started = time.perf_counter()
    result = model.predict(x)
    raw_adjacency = np.asarray(result.adjacency, dtype=np.int8)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    candidate = project_to_dag(raw_adjacency, probabilities)
    payload: dict[str, object] = {
        "raw_adjacency": raw_adjacency,
        "probabilities": probabilities,
        "threshold": np.float64(result.threshold),
        "candidate_adjacency": candidate,
        "runtime_sec": np.float64(time.perf_counter() - started),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
    return payload


def infer(*, shard_index: int = 0, shard_count: int = 1) -> None:
    _load_protocol()
    manifest = _load_manifest()
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("invalid inference shard")
    selected = manifest.iloc[shard_index::shard_count].reset_index(drop=True)

    import torch
    from cdfm import CDFM

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // max(1, shard_count)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    atomic_json(
        {
            "schema_version": "gate5a_environment_v1",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "packages": {
                name: package_version(name)
                for name in ("cdfm-base", "numpy", "pandas", "scipy", "scikit-learn", "causal-learn", "torch")
            },
            "cdfm_info": getattr(model, "info", "unavailable"),
            "cdfm_frozen": True,
            "discovery_only": True,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        RESULTS / "environment.json",
    )

    for ordinal, row in enumerate(selected.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = REPO_ROOT / str(row["raw_path"])
        if destination.exists():
            with np.load(destination) as cached:
                if not _is_dag(cached["candidate_adjacency"]):
                    raise RuntimeError(f"cached candidate is not a DAG: {task_id}")
            print(f"[{ordinal}/{len(selected)}] skip cached {task_id}", flush=True)
            continue
        try:
            x = _load_discovery(row)
            _predict_and_save(model, x, destination)
            with np.load(destination) as saved:
                raw_edges = _edge_count(saved["raw_adjacency"])
                candidate_edges = _edge_count(saved["candidate_adjacency"])
                runtime = float(saved["runtime_sec"])
            print(
                f"[{ordinal}/{len(selected)}] {task_id}: raw_edges={raw_edges}, "
                f"candidate_edges={candidate_edges}, CDFM={runtime:.2f}s",
                flush=True,
            )
        except Exception as exc:
            _append_error(task_id, "inference", exc)
            print(f"[{ordinal}/{len(selected)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)

    completed = sum((REPO_ROOT / str(path)).exists() for path in manifest["raw_path"])
    if completed == len(manifest):
        atomic_json(
            {
                "schema_version": "gate5a_inference_receipt_v1",
                "task_count": len(manifest),
                "completed": completed,
                "all_candidates_are_dags": True,
                "cdfm_input": "X_discovery only",
                "truth_loaded": False,
                "raw_prediction_hashes": {
                    str(row["task_id"]): sha256(REPO_ROOT / str(row["raw_path"]))
                    for row in manifest.to_dict(orient="records")
                },
            },
            RESULTS / "inference_receipt.json",
        )
        print(f"Gate-5A inference complete: {completed}/{len(manifest)}", flush=True)
    else:
        print(f"Gate-5A inference shard complete: {completed}/{len(manifest)} total cached", flush=True)


def _parents(graph: np.ndarray, node: int) -> set[int]:
    return set(np.flatnonzero(graph[:, node]).astype(int).tolist())


def _descendants(graph: np.ndarray, node: int) -> set[int]:
    seen: set[int] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child_value in np.flatnonzero(graph[current]):
            child = int(child_value)
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def local_markov_statements(candidate: np.ndarray) -> list[dict[str, object]]:
    candidate = np.asarray(candidate, dtype=np.int8)
    if not _is_dag(candidate):
        raise ValueError("local Markov statements require a DAG candidate")
    statements: list[dict[str, object]] = []
    all_nodes = set(range(candidate.shape[0]))
    for y in range(candidate.shape[0]):
        parents = _parents(candidate, y)
        descendants = _descendants(candidate, y)
        non_desc_non_parent = sorted(all_nodes - {y} - descendants - parents)
        for x in non_desc_non_parent:
            statements.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "conditioning_set": sorted(int(item) for item in parents),
                }
            )
    return statements


def _stable_test_seed(graph_seed: int, x: int, y: int, conditioning_set: list[int]) -> int:
    value = ANALYSIS_SEED + 31 * int(graph_seed) + 101 * int(x) + 1009 * int(y)
    for item in conditioning_set:
        value = value * 65537 + int(item) + 1
    return int(value % (2**32 - 1))


def holm_adjust(p_values: Iterable[float], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be finite and one-dimensional")
    if len(values) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    factors = np.arange(len(values), 0, -1, dtype=float)
    adjusted_sorted = np.maximum.accumulate(sorted_values * factors)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    reject_sorted = np.zeros(len(values), dtype=bool)
    still_rejecting = True
    for index, p_value in enumerate(sorted_values):
        if still_rejecting and p_value <= alpha / (len(values) - index):
            reject_sorted[index] = True
        else:
            still_rejecting = False
    rejected = np.empty_like(reject_sorted)
    rejected[order] = reject_sorted
    return rejected, adjusted


def build_task_witnesses(row: dict[str, object]) -> dict[str, object]:
    """Build validation-only CI witnesses; this function accepts no truth graph."""
    from causallearn.utils.cit import CIT

    task_id = str(row["task_id"])
    raw_path = REPO_ROOT / str(row["raw_path"])
    with np.load(raw_path) as raw:
        candidate = np.asarray(raw["candidate_adjacency"], dtype=np.int8)
    validation = _load_validation(row)
    statements = local_markov_statements(candidate)
    ci_test = CIT(
        validation,
        method="rcit",
        approx="lpd4",
        num_f=RCIT_NUM_F,
        num_f2=RCIT_NUM_F2,
        rcit=True,
    )
    p_values: list[float] = []
    enriched: list[dict[str, object]] = []
    for statement in statements:
        x = int(statement["x"])
        y = int(statement["y"])
        conditioning_set = [int(item) for item in statement["conditioning_set"]]
        np.random.seed(_stable_test_seed(int(row["graph_seed"]), x, y, conditioning_set))
        p_value = float(ci_test(x, y, conditioning_set))
        if not np.isfinite(p_value):
            raise ValueError(f"non-finite RCIT p-value for {task_id}: {statement}")
        p_values.append(p_value)
        enriched.append(
            {
                "x": x,
                "y": y,
                "conditioning_set": conditioning_set,
                "p_value": p_value,
            }
        )
    rejected, adjusted = holm_adjust(p_values, HOLM_ALPHA)
    for statement, is_rejected, adjusted_p in zip(enriched, rejected, adjusted):
        statement["holm_adjusted_p_value"] = float(adjusted_p)
        statement["rejected"] = bool(is_rejected)
    result = {
        "schema_version": "gate5a_task_witnesses_v1",
        "task_id": task_id,
        "domain": str(row["domain"]),
        "candidate_edges": _edge_count(candidate),
        "test_count": len(enriched),
        "rejected_witness_count": int(np.sum(rejected)),
        "ci_test": "RCIT",
        "validation_rows": N_VALIDATION,
        "truth_loaded": False,
        "statements": enriched,
    }
    destination = REPO_ROOT / str(row["witness_path"])
    atomic_json(result, destination)
    return result


def build_witnesses() -> None:
    _load_protocol()
    manifest = _load_manifest()
    TASK_WITNESSES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    completed = 0
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        try:
            destination = REPO_ROOT / str(row["witness_path"])
            if destination.exists():
                result = json.loads(destination.read_text(encoding="utf-8"))
            else:
                result = build_task_witnesses(row)
            for statement in result["statements"]:
                rows.append({
                    "task_id": task_id,
                    "domain": str(row["domain"]),
                    "x": statement["x"],
                    "y": statement["y"],
                    "conditioning_set": json.dumps(statement["conditioning_set"]),
                    "p_value": statement["p_value"],
                    "holm_adjusted_p_value": statement["holm_adjusted_p_value"],
                    "rejected": statement["rejected"],
                })
            completed += 1
            print(
                f"[{ordinal}/{len(manifest)}] {task_id}: tests={result['test_count']}, "
                f"witnesses={result['rejected_witness_count']}",
                flush=True,
            )
        except Exception as exc:
            _append_error(task_id, "witnesses", exc)
            print(f"[{ordinal}/{len(manifest)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    atomic_csv(pd.DataFrame(rows), RESULTS / "witnesses.csv")
    atomic_json(
        {
            "schema_version": "gate5a_witness_receipt_v1",
            "task_count": len(manifest),
            "completed": completed,
            "error_count": len(manifest) - completed,
            "alpha": HOLM_ALPHA,
            "multiple_testing": "Holm over all local-Markov tests within task",
            "validation_only": True,
            "truth_loaded": False,
            "witnesses_sha256": sha256(RESULTS / "witnesses.csv"),
        },
        RESULTS / "witness_receipt.json",
    )
    if completed != len(manifest):
        raise RuntimeError(f"witness construction incomplete: {completed}/{len(manifest)}")


def _ancestral_set(graph: np.ndarray, nodes: set[int]) -> set[int]:
    ancestors = set(nodes)
    changed = True
    while changed:
        changed = False
        for node in tuple(ancestors):
            for parent in np.flatnonzero(graph[:, node]):
                parent_int = int(parent)
                if parent_int not in ancestors:
                    ancestors.add(parent_int)
                    changed = True
    return ancestors


def d_separated(graph: np.ndarray, x: int, y: int, conditioning_set: Iterable[int]) -> bool:
    """D-separation by ancestral moralization for a DAG."""
    graph = np.asarray(graph, dtype=np.int8)
    if not _is_dag(graph):
        raise ValueError("d-separation requires a DAG")
    conditioning = set(int(item) for item in conditioning_set)
    if x == y or x in conditioning or y in conditioning:
        return False
    ancestral = _ancestral_set(graph, {int(x), int(y), *conditioning})
    undirected: dict[int, set[int]] = {node: set() for node in ancestral}
    for child in ancestral:
        parents = [int(parent) for parent in np.flatnonzero(graph[:, child]) if int(parent) in ancestral]
        for parent in parents:
            undirected[parent].add(child)
            undirected[child].add(parent)
        for left_index, left in enumerate(parents):
            for right in parents[left_index + 1:]:
                undirected[left].add(right)
                undirected[right].add(left)
    active = set(ancestral) - conditioning
    if x not in active or y not in active:
        return False
    stack = [int(x)]
    seen = {int(x)}
    while stack:
        current = stack.pop()
        if current == y:
            return False
        for neighbor in undirected[current]:
            if neighbor in active and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return True


def _pair_indices(dimension: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(dimension) for right in range(left + 1, dimension)]


def _graphs_at_distance(candidate: np.ndarray, distance: int) -> Iterable[np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.int8)
    pairs = _pair_indices(candidate.shape[0])
    current_states = [_edge_state(candidate, left, right) for left, right in pairs]
    if distance == 0:
        yield candidate.copy()
        return
    for changed_indices in itertools.combinations(range(len(pairs)), distance):
        alternative_states = [
            tuple(state for state in (0, 1, 2) if state != current_states[index])
            for index in changed_indices
        ]
        for replacement in itertools.product(*alternative_states):
            graph = candidate.copy()
            for pair_index, state in zip(changed_indices, replacement):
                _set_edge_state(graph, *pairs[pair_index], state)
            if _is_dag(graph):
                yield graph


def minimum_repair_radius(
    candidate: np.ndarray,
    rejected_witnesses: list[dict[str, object]],
    max_k: int = MAX_REPAIR_K,
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.int8)
    if not _is_dag(candidate):
        raise ValueError("repair search requires a DAG candidate")
    if not rejected_witnesses:
        return {
            "rr_lower_bound": 0,
            "rr_exact": True,
            "searched_max_k": 0,
            "dag_candidates_checked": 1,
        }

    checked = 0
    for distance in range(max_k + 1):
        for graph in _graphs_at_distance(candidate, distance):
            checked += 1
            if all(
                not d_separated(
                    graph,
                    int(witness["x"]),
                    int(witness["y"]),
                    [int(item) for item in witness["conditioning_set"]],
                )
                for witness in rejected_witnesses
            ):
                return {
                    "rr_lower_bound": distance,
                    "rr_exact": True,
                    "searched_max_k": distance,
                    "dag_candidates_checked": checked,
                }
    return {
        "rr_lower_bound": max_k + 1,
        "rr_exact": False,
        "searched_max_k": max_k,
        "dag_candidates_checked": checked,
    }


def build_certificates() -> None:
    _load_protocol()
    manifest = _load_manifest()
    rows: list[dict[str, object]] = []
    completed = 0
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        try:
            witness_path = REPO_ROOT / str(row["witness_path"])
            raw_path = REPO_ROOT / str(row["raw_path"])
            certificate_path = REPO_ROOT / str(row["certificate_path"])
            witness_result = json.loads(witness_path.read_text(encoding="utf-8"))
            with np.load(raw_path) as raw:
                candidate = np.asarray(raw["candidate_adjacency"], dtype=np.int8)
            rejected = [item for item in witness_result["statements"] if item["rejected"]]
            repair = minimum_repair_radius(candidate, rejected, MAX_REPAIR_K)
            result = {
                "schema_version": "gate5a_task_certificate_v1",
                "task_id": task_id,
                "domain": str(row["domain"]),
                "candidate_edges": _edge_count(candidate),
                "test_count": int(witness_result["test_count"]),
                "witness_count": len(rejected),
                "rr_lower_bound": int(repair["rr_lower_bound"]),
                "rr_exact": bool(repair["rr_exact"]),
                "searched_max_k": int(repair["searched_max_k"]),
                "dag_candidates_checked": int(repair["dag_candidates_checked"]),
                "truth_loaded": False,
            }
            atomic_json(result, certificate_path)
            rows.append(result)
            completed += 1
            print(
                f"[{ordinal}/{len(manifest)}] {task_id}: witnesses={len(rejected)}, "
                f"RR_lb={repair['rr_lower_bound']} exact={repair['rr_exact']}",
                flush=True,
            )
        except Exception as exc:
            _append_error(task_id, "certificates", exc)
            print(f"[{ordinal}/{len(manifest)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    frame = pd.DataFrame(rows)
    atomic_csv(frame, RESULTS / "certificate_predictions.csv")
    atomic_json(
        {
            "schema_version": "gate5a_certificate_receipt_v1",
            "task_count": len(manifest),
            "completed": completed,
            "error_count": len(manifest) - completed,
            "max_repair_k": MAX_REPAIR_K,
            "certificate_predictions_sha256": sha256(RESULTS / "certificate_predictions.csv"),
            "validation_only": True,
            "truth_loaded": False,
        },
        RESULTS / "certificate_receipt.json",
    )
    if completed != len(manifest):
        raise RuntimeError(f"certificate construction incomplete: {completed}/{len(manifest)}")

    pretruth_files = {
        "protocol": PROTOCOL_PATH,
        "manifest": MANIFEST_PATH,
        "inference_receipt": RESULTS / "inference_receipt.json",
        "witness_receipt": RESULTS / "witness_receipt.json",
        "witnesses": RESULTS / "witnesses.csv",
        "certificate_receipt": RESULTS / "certificate_receipt.json",
        "certificate_predictions": RESULTS / "certificate_predictions.csv",
    }
    atomic_json(
        {
            "schema_version": "gate5a_pre_truth_hashes_v1",
            "files": {name: {"path": _repo_relative(path), "sha256": sha256(path)} for name, path in pretruth_files.items()},
            "final_truth_loaded": False,
            "hashes_sealed_before_truth": True,
        },
        RESULTS / "pre_truth_hashes.json",
    )


def _check_pretruth_hashes() -> dict[str, object]:
    receipt = json.loads((RESULTS / "pre_truth_hashes.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for name, item in receipt["files"].items():
        path = REPO_ROOT / str(item["path"])
        checks[name] = path.exists() and sha256(path) == item["sha256"]
    if not all(checks.values()):
        raise RuntimeError(f"pre-truth hash mismatch: {checks}")
    return {"checks": checks, "all_match": True}


def _clopper_pearson_lower(successes: int, total: int, confidence: float = 0.95) -> float | None:
    if total <= 0:
        return None
    from scipy.stats import beta

    if successes <= 0:
        return 0.0
    upper_shape = 1 if successes == total else total - successes + 1
    return float(beta.ppf((1.0 - confidence) / 2.0, successes, upper_shape))


def _metric_row(frame: pd.DataFrame, scope: str, domain: str) -> dict[str, object]:
    subset = frame if scope == "pooled" else frame[frame["domain"] == domain]
    complete = subset[subset["status"] == "complete"]
    if complete.empty:
        return {"scope": scope, "domain": domain, "n_attempted": len(subset), "n_complete": 0}
    valid = complete["rr_lower_bound"] <= complete["true_shd"]
    bad = complete["true_shd"] >= 1
    severe = complete["true_shd"] >= 4
    tight = complete.loc[complete["true_shd"] > 0, "rr_lower_bound"] / complete.loc[complete["true_shd"] > 0, "true_shd"]
    return {
        "scope": scope,
        "domain": domain,
        "n_attempted": len(subset),
        "n_complete": len(complete),
        "n_errors": int((subset["status"] != "complete").sum()),
        "validity_count": int(valid.sum()),
        "validity_rate": float(valid.mean()),
        "validity_lower_95": _clopper_pearson_lower(int(valid.sum()), len(valid)),
        "bad_graph_count": int(bad.sum()),
        "bad_graph_rr_ge_1_count": int((complete.loc[bad, "rr_lower_bound"] >= 1).sum()),
        "bad_graph_nonzero_rate": float((complete.loc[bad, "rr_lower_bound"] >= 1).mean()) if bad.any() else None,
        "severe_graph_count": int(severe.sum()),
        "severe_graph_rr_ge_2_count": int((complete.loc[severe, "rr_lower_bound"] >= 2).sum()),
        "severe_graph_rr_ge_2_rate": float((complete.loc[severe, "rr_lower_bound"] >= 2).mean()) if severe.any() else None,
        "mean_tightness": float(tight.mean()) if len(tight) else None,
        "exact_radius_rate": float(complete["rr_exact"].mean()),
        "mean_witness_count": float(complete["witness_count"].mean()),
    }


def open_final_truth() -> None:
    _load_protocol()
    hash_status = _check_pretruth_hashes()
    manifest = _load_manifest()
    certificates = pd.read_csv(RESULTS / "certificate_predictions.csv")
    certificate_by_task = {str(row["task_id"]): row for row in certificates.to_dict(orient="records")}
    rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        task_id = str(row["task_id"])
        certificate = certificate_by_task.get(task_id)
        if certificate is None:
            rows.append({"task_id": task_id, "domain": str(row["domain"]), "status": "error"})
            continue
        raw_path = REPO_ROOT / str(row["raw_path"])
        dataset_path = REPO_ROOT / str(row["dataset_path"])
        with np.load(raw_path) as raw:
            candidate = np.asarray(raw["candidate_adjacency"], dtype=np.int8)
        # This is the sole truth load in the pipeline.
        with np.load(dataset_path) as data:
            truth = np.asarray(data["truth_adjacency"], dtype=np.int8)
        if not _is_dag(truth):
            raise AssertionError(f"ground-truth graph is not a DAG: {task_id}")
        true_shd = sum(
            _edge_state(candidate, left, right) != _edge_state(truth, left, right)
            for left, right in _pair_indices(N_VARIABLES)
        )
        rows.append(
            {
                **certificate,
                "status": "complete",
                "true_shd": int(true_shd),
                "bad_graph": bool(true_shd >= 1),
                "severe_bad_graph": bool(true_shd >= 4),
                "validity": bool(int(certificate["rr_lower_bound"]) <= true_shd),
            }
        )
    final = pd.DataFrame(rows)
    atomic_csv(final, RESULTS / "final_predictions.csv")
    metric_rows = [_metric_row(final, "pooled", "ALL")]
    metric_rows.extend(_metric_row(final, "domain", domain) for domain in sorted(final["domain"].dropna().unique()))
    metrics = pd.DataFrame(metric_rows)
    atomic_csv(metrics, RESULTS / "final_metrics.csv")

    pooled = metric_rows[0]
    error_rate = float((len(final) - int(pooled.get("n_complete", 0))) / len(final)) if len(final) else 1.0
    denominators_ok = int(pooled.get("bad_graph_count", 0)) > 0 and int(pooled.get("severe_graph_count", 0)) > 0
    technical_ok = error_rate <= 0.10 and int(pooled.get("n_complete", 0)) > 0 and denominators_ok
    validity_ok = technical_ok and float(pooled.get("validity_rate", 0.0)) >= 0.95
    nonvacuity_ok = (
        validity_ok
        and float(pooled.get("bad_graph_nonzero_rate", 0.0) or 0.0) >= 0.40
        and float(pooled.get("severe_graph_rr_ge_2_rate", 0.0) or 0.0) >= 0.50
    )
    if not technical_ok:
        decision = "GATE5A_INCONCLUSIVE"
    elif not validity_ok:
        decision = "GATE5A_STOP"
    elif nonvacuity_ok:
        decision = "GATE5A_GO"
    else:
        decision = "GATE5A_BORDERLINE"
    summary = {
        "schema_version": "gate5a_summary_v1",
        "decision": decision,
        "validity_ok": bool(validity_ok),
        "nonvacuity_ok": bool(nonvacuity_ok),
        "technical_ok": bool(technical_ok),
        "error_rate": error_rate,
        "pooled_metrics": pooled,
        "gate_thresholds": protocol_payload()["gate"],
        "truth_loaded_after_pretruth_hashes": True,
        "pretruth_hashes_unchanged": hash_status,
        "post_truth_certificate_changes": False,
    }
    atomic_json(summary, RESULTS / "final_summary.json")
    atomic_json(
        {
            "schema_version": "gate5a_final_truth_open_receipt_v1",
            "truth_loaded_after": ["certificate_predictions.csv", "pre_truth_hashes.json"],
            "task_count": len(final),
            "complete_count": int((final["status"] == "complete").sum()),
            "pretruth_hashes_unchanged": hash_status,
            "truth_used_for_witnesses_or_certificates": False,
        },
        RESULTS / "final_truth_open_receipt.json",
    )
    print(decision, flush=True)


def report() -> None:
    summary = json.loads((RESULTS / "final_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "final_metrics.csv")
    lines = [
        f"# Gate-5A result: **{summary['decision']}**",
        "",
        "> Can held-out causal contradictions produce a non-vacuous ground-truth-free lower bound on CDFM graph error?",
        "",
        "The formal certificate is a minimum structural repair lower bound. CDFM sees only `X_discovery`; CI tests, Holm correction, and repair search use only `X_validation`; the true DAG is opened only in the final evaluation stage.",
        "",
        "## Pooled metrics",
        "",
        metrics[metrics["scope"] == "pooled"].to_markdown(index=False),
        "",
        "## Domain metrics",
        "",
        metrics[metrics["scope"] == "domain"].to_markdown(index=False),
        "",
        "## Gate interpretation",
        "",
        f"`{json.dumps(summary, ensure_ascii=False)}`",
        "",
        "`RR_lower_bound` is exact when `rr_exact=true`. A value of 4 with `rr_exact=false` means the search found no valid repair at k=0,1,2,3 and claims only `RR >= 4`.",
        "",
        "All pre-truth hashes and detailed task-level certificates are retained under `results/` and `cache/`.",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary["decision"], flush=True)


def verify() -> None:
    required = [
        PROTOCOL_PATH,
        MANIFEST_PATH,
        RESULTS / "prepare_receipt.json",
        RESULTS / "inference_receipt.json",
        RESULTS / "witnesses.csv",
        RESULTS / "witness_receipt.json",
        RESULTS / "certificate_predictions.csv",
        RESULTS / "certificate_receipt.json",
        RESULTS / "pre_truth_hashes.json",
        RESULTS / "final_predictions.csv",
        RESULTS / "final_metrics.csv",
        RESULTS / "final_summary.json",
        RESULTS / "final_truth_open_receipt.json",
        RESULTS / "REPORT.md",
    ]
    hash_status = _check_pretruth_hashes()
    manifest = _load_manifest()
    raw_complete = all((REPO_ROOT / str(path)).exists() for path in manifest["raw_path"])
    witness_complete = all((REPO_ROOT / str(path)).exists() for path in manifest["witness_path"])
    certificate_complete = all((REPO_ROOT / str(path)).exists() for path in manifest["certificate_path"])
    payload = {
        "schema_version": "gate5a_verification_v1",
        "required_artifacts": {_repo_relative(path): path.exists() for path in required},
        "all_required_artifacts": all(path.exists() for path in required),
        "raw_predictions_complete": raw_complete,
        "task_witnesses_complete": witness_complete,
        "task_certificates_complete": certificate_complete,
        "pretruth_hashes_unchanged": hash_status,
        "truth_open_receipt_exists": (RESULTS / "final_truth_open_receipt.json").exists(),
    }
    atomic_json(payload, RESULTS / "verification.json")
    if not payload["all_required_artifacts"] or not raw_complete or not witness_complete or not certificate_complete or not hash_status["all_match"]:
        raise RuntimeError("Gate-5A verification failed")
    print("Gate-5A verification passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "infer", "build-witnesses", "build-certificates", "open-final-truth", "report", "verify", "all"),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "infer":
        infer(shard_index=args.shard_index, shard_count=args.shard_count)
    elif args.stage == "build-witnesses":
        build_witnesses()
    elif args.stage == "build-certificates":
        build_certificates()
    elif args.stage == "open-final-truth":
        open_final_truth()
    elif args.stage == "report":
        report()
    elif args.stage == "verify":
        verify()
    else:
        prepare()
        infer(shard_index=args.shard_index, shard_count=args.shard_count)
        build_witnesses()
        build_certificates()
        open_final_truth()
        report()
        verify()


if __name__ == "__main__":
    main()
