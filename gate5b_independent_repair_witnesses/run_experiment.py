"""Gate-5B: independent-repair witnesses for graph-error certificates.

This experiment is intentionally isolated from ``gate5a_minimum_repair``.
Gate-5A is imported only for read-only pure helpers and for the frozen CDFM
projection/data generator.  No Gate-5A file is written by this module.

The experiment reports two certificates separately:

* RR_CI: the Gate-5A local-Markov/RCIT certificate, used as a paired baseline.
* RR_ANM: a mechanism certificate based on held-out residual-independence
  witnesses and an explicit one-edit repair signature.

The mechanism certificate is empirical and is only interpreted under the
additive independent-noise mechanism family used by the synthetic generator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
GATE5A_ROOT = REPO_ROOT / "gate5a_minimum_repair"
GATE5A_RUNNER = GATE5A_ROOT / "run_experiment.py"

if not GATE5A_RUNNER.exists():
    raise FileNotFoundError(f"missing read-only Gate-5A runner: {GATE5A_RUNNER}")

_spec = importlib.util.spec_from_file_location("gate5a_read_only", GATE5A_RUNNER)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot import read-only Gate-5A runner: {GATE5A_RUNNER}")
gate5a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate5a)


ANALYSIS_SEED = 20260921
ANM_SPLIT_SEED = 510000
ANM_MODEL_SEED = 610000
N_VARIABLES = 10
N_DOMAINS = 6
TASKS_PER_DOMAIN = 20
N_TASKS = N_DOMAINS * TASKS_PER_DOMAIN
N_SAMPLES_TOTAL = 1000
N_DISCOVERY = 500
N_VALIDATION = 500
FINAL_GRAPH_SEED_START = 860000
FINAL_GRAPH_SEED_END = FINAL_GRAPH_SEED_START + N_TASKS - 1
ALPHA = 0.05
MAX_REPAIR_K = 3
RCIT_NUM_F = 100
RCIT_NUM_F2 = 5
ANM_N_ESTIMATORS = 300
ANM_MIN_SAMPLES_LEAF = 5

RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
DATASETS = CACHE / "datasets"
RAW = CACHE / "raw_predictions"

DEV_RESULTS = RESULTS / "dev"
DEV_CACHE = CACHE / "dev"
FINAL_CACHE = CACHE / "final"

DEV_MANIFEST = DEV_RESULTS / "manifest.csv"
FINAL_MANIFEST = RESULTS / "manifest.csv"
DEV_PROTOCOL = DEV_RESULTS / "dev_protocol.json"
FROZEN_PROTOCOL = RESULTS / "frozen_protocol.json"


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
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
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


def repo_relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def append_error(task_id: str, stage: str, exc: BaseException, phase: str) -> None:
    destination = RESULTS / phase / "errors.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task_id": task_id,
                    "stage": stage,
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp_epoch": time.time(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _source_files() -> list[Path]:
    return [
        ROOT / "run_experiment.py",
        ROOT / "test_gate5b.py",
        ROOT / "README.md",
        ROOT / "LITERATURE_BOUNDARY.md",
    ]


def source_hashes() -> dict[str, str]:
    return {repo_relative(path): sha256(path) for path in _source_files() if path.exists()}


def snapshot_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def gate5a_snapshot_path() -> Path:
    return DEV_RESULTS / "gate5a_read_only_snapshot.json"


def snapshot_gate5a() -> None:
    atomic_json(
        {
            "schema_version": "gate5b_gate5a_snapshot_v1",
            "root": repo_relative(GATE5A_ROOT),
            "files": snapshot_tree(GATE5A_ROOT),
            "captured_before_gate5b_writes": True,
        },
        gate5a_snapshot_path(),
    )


def verify_gate5a_snapshot() -> dict[str, object]:
    receipt = json.loads(gate5a_snapshot_path().read_text(encoding="utf-8"))
    expected = {str(key): str(value) for key, value in receipt["files"].items()}
    actual = snapshot_tree(GATE5A_ROOT)
    keys_match = set(expected) == set(actual)
    mismatches = sorted(
        key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key)
    )
    result = {
        "all_match": keys_match and not mismatches,
        "file_count": len(actual),
        "mismatches": mismatches,
    }
    if not result["all_match"]:
        raise RuntimeError(f"Gate-5A changed during Gate-5B: {result}")
    return result


def protocol_payload(*, phase: str = "dev_and_final") -> dict[str, object]:
    return {
        "schema_version": "gate5b_independent_repair_witnesses_v1",
        "freeze_date": "2026-09-21",
        "phase": phase,
        "scientific_question": "Can held-out causal contradictions produce a non-vacuous ground-truth-free lower bound on CDFM graph error?",
        "formal_target": "ground-truth-free minimum structural repair / graph-error lower-bound certification",
        "data": {
            "domains": [dict(item) for item in gate5a.DOMAINS],
            "domain_count": N_DOMAINS,
            "tasks_per_domain": TASKS_PER_DOMAIN,
            "task_count": N_TASKS,
            "D": N_VARIABLES,
            "N_total": N_SAMPLES_TOTAL,
            "N_discovery": N_DISCOVERY,
            "N_validation": N_VALIDATION,
            "split": "first 500 rows discovery, last 500 rows validation; standardization fit on discovery only",
            "dev_source": "Gate-5A manifest, raw CDFM predictions and validation data, all read-only",
            "final_graph_seed_range": [FINAL_GRAPH_SEED_START, FINAL_GRAPH_SEED_END],
        },
        "cdfm": {
            "model": "DMIRLAB/CDFM",
            "input": "X_discovery only",
            "training": False,
            "fine_tuning": False,
            "source_modification": False,
        },
        "certificates": {
            "rr_ci": {
                "role": "paired Gate-5A pure Markov/CI baseline",
                "statements": "local Markov statements for every candidate node",
                "ci_test": "causal-learn RCIT",
                "multiple_testing": "Holm over all local-Markov statements per task",
                "alpha": ALPHA,
                "test_data": "X_validation only",
            },
            "rr_anm": {
                "role": "additive independent-noise mechanism certificate",
                "assumption": "Y=f(Pa_candidate(Y))+epsilon with epsilon independent of candidate non-descendants when the parent set is correct",
                "parent_set": "Pa_candidate(Y)",
                "tested_variables": "all candidate non-descendants other than Y, including candidate parents",
                "regressor": {
                    "class": "sklearn.ensemble.ExtraTreesRegressor",
                    "n_estimators": ANM_N_ESTIMATORS,
                    "min_samples_leaf": ANM_MIN_SAMPLES_LEAF,
                    "max_features": 1.0,
                    "bootstrap": False,
                    "n_jobs": 1,
                },
                "validation_split": "deterministic 250/250 A/B split from graph seed",
                "residual_test": "RCIT residual versus one tested non-descendant, unconditional",
                "fold_p_value": "min(1, 2*min(p_A_to_B, p_B_to_A))",
                "multiple_testing": "Holm over all atomic (Y,Z,Pa_candidate(Y)) witnesses per task",
                "alpha": ALPHA,
                "test_data": "X_validation only",
                "ground_truth_used": False,
            },
        },
        "repair": {
            "distance": "directed pairwise SHD; add, delete, or reverse one unordered-pair state costs one",
            "space": "DAGs on the same ten observed nodes",
            "search": "exhaustive pair-state enumeration with DAG pruning for k=0,1,2,3",
            "no_solution_by_max_k": "report lower bound 4 and exact=false",
            "ci_witness_persists": "repaired DAG d-separates the tested pair given the fixed candidate conditioning set",
            "anm_witness_persists": "repaired DAG has the same candidate parent set for Y and Z remains a non-descendant of Y",
            "certificates_separate": True,
        },
        "evaluation": {
            "truth": "opened only after phase pre-truth hashes",
            "true_error": "SHD(candidate_DAG, ground_truth_DAG)",
            "validity": "RR_CI and RR_ANM lower bounds do not exceed true SHD",
            "bad_graph": "true SHD >= 1",
            "severe_bad_graph": "true SHD >= 4",
            "tightness": "RR / true SHD for true SHD > 0",
            "primary": "RR_ANM severe-graph RR>=2 rate",
            "paired_gain": "RR_ANM severe RR>=2 rate minus RR_CI severe RR>=2 rate",
        },
        "gate": {
            "validity_rate_min": 0.95,
            "bad_graph_nonzero_rate_min": 0.40,
            "severe_graph_rr_ge_2_rate_min": 0.50,
            "paired_severe_gain_min": 0.10,
            "technical_error_rate_max": 0.10,
            "dev_failure": "any DEV condition fails: stop before independent FINAL",
            "stop": "validity fails",
            "borderline": "validity passes but an efficacy condition fails",
            "inconclusive": "technical failure rate exceeds 10 percent or an evaluation denominator is empty",
        },
        "truth_isolation": {
            "cdfm_input_is_discovery_only": True,
            "witness_builder_accepts_truth": False,
            "certificate_builder_accepts_truth": False,
            "final_truth_loaded_before_certificate": False,
            "final_truth_loader": "open-final-truth only after pre_truth_hashes.json",
        },
    }


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


def _final_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ordinal = 0
    for domain_spec in gate5a.DOMAINS:
        for task_index in range(TASKS_PER_DOMAIN):
            graph_seed = FINAL_GRAPH_SEED_START + ordinal
            task_id = f"{domain_spec['domain']}_{task_index:03d}"
            dataset_path = DATASETS / f"{task_id}.npz"
            raw_path = RAW / f"{task_id}.npz"
            spec = gate5a._graph(graph_seed)
            discovery, validation = gate5a._generate_raw_split(
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
                    "dataset_path": repo_relative(dataset_path),
                    "raw_path": repo_relative(raw_path),
                }
            )
            ordinal += 1
    return rows


def prepare_dev() -> None:
    DEV_RESULTS.mkdir(parents=True, exist_ok=True)
    DEV_CACHE.mkdir(parents=True, exist_ok=True)
    snapshot_gate5a()
    source_manifest = pd.read_csv(gate5a.RESULTS / "manifest.csv")
    required = [
        gate5a.RESULTS / "manifest.csv",
        gate5a.RESULTS / "certificate_predictions.csv",
        gate5a.RESULTS / "final_predictions.csv",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Gate-5A DEV inputs are incomplete")
    source_manifest = source_manifest.copy()
    source_manifest["phase"] = "dev"
    source_manifest["source_experiment"] = "gate5a_minimum_repair"
    atomic_csv(source_manifest, DEV_MANIFEST)
    atomic_json(protocol_payload(phase="dev"), DEV_PROTOCOL)
    atomic_json(
        {
            "schema_version": "gate5b_prepare_dev_receipt_v1",
            "task_count": len(source_manifest),
            "domain_counts": source_manifest.groupby("domain").size().to_dict(),
            "source_manifest": repo_relative(gate5a.RESULTS / "manifest.csv"),
            "cdfm_reused_read_only": True,
            "truth_loaded": False,
            "gate5a_snapshot": repo_relative(gate5a_snapshot_path()),
        },
        DEV_RESULTS / "prepare_receipt.json",
    )
    print(f"prepared Gate-5B DEV: {len(source_manifest)} read-only Gate-5A tasks", flush=True)


def prepare_final() -> None:
    frozen = json.loads(FROZEN_PROTOCOL.read_text(encoding="utf-8"))
    if frozen.get("dev_decision") != "GATE5B_DEV_GO":
        raise RuntimeError("FINAL preparation requires a passing Gate-5B DEV-SCREEN")
    DATASETS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    FINAL_CACHE.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(_final_rows())
    if len(rows) != N_TASKS or rows["graph_seed"].duplicated().any():
        raise AssertionError("FINAL manifest task or seed count is invalid")
    atomic_csv(rows, FINAL_MANIFEST)
    atomic_json(
        {
            "schema_version": "gate5b_prepare_final_receipt_v1",
            "task_count": len(rows),
            "domain_counts": rows.groupby("domain").size().to_dict(),
            "graph_seed_range": [int(rows["graph_seed"].min()), int(rows["graph_seed"].max())],
            "truth_graph_written_but_not_loaded": True,
            "cdfm_input_keys": ["X_discovery"],
            "certificate_input_keys": ["X_validation", "candidate_adjacency"],
        },
        RESULTS / "prepare_final_receipt.json",
    )
    print(f"prepared Gate-5B FINAL: {len(rows)} tasks", flush=True)


def _load_manifest(phase: str) -> pd.DataFrame:
    path = DEV_MANIFEST if phase == "dev" else FINAL_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"missing {phase} manifest: {path}")
    return pd.read_csv(path)


def _load_dataset(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(REPO_ROOT / str(row["dataset_path"])) as data:
        return np.asarray(data["X_discovery"], dtype=np.float64), np.asarray(data["X_validation"], dtype=np.float64)


def _load_candidate(row: dict[str, object]) -> np.ndarray:
    with np.load(REPO_ROOT / str(row["raw_path"])) as raw:
        candidate = np.asarray(raw["candidate_adjacency"], dtype=np.int8)
    if not gate5a._is_dag(candidate):
        raise ValueError(f"candidate is not a DAG: {row['task_id']}")
    return candidate


def _predict_and_save(model: object, discovery: np.ndarray, destination: Path) -> None:
    started = time.perf_counter()
    result = model.predict(discovery)
    raw_adjacency = np.asarray(result.adjacency, dtype=np.int8)
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    candidate = gate5a.project_to_dag(raw_adjacency, probabilities)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            raw_adjacency=raw_adjacency,
            probabilities=probabilities,
            threshold=np.float64(result.threshold),
            candidate_adjacency=candidate,
            runtime_sec=np.float64(time.perf_counter() - started),
        )
    os.replace(temporary, destination)


def infer(*, shard_index: int = 0, shard_count: int = 1) -> None:
    if not FROZEN_PROTOCOL.exists():
        raise RuntimeError("FINAL inference requires frozen_protocol.json")
    manifest = _load_manifest("final")
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
            "schema_version": "gate5b_environment_v1",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "packages": {
                name: package_version(name)
                for name in ("cdfm-base", "numpy", "pandas", "scipy", "scikit-learn", "causal-learn", "torch")
            },
            "cdfm_frozen": True,
            "discovery_only": True,
            "shard_index": shard_index,
            "shard_count": shard_count,
        },
        RESULTS / "environment.json",
    )
    errors = 0
    for ordinal, row in enumerate(selected.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = REPO_ROOT / str(row["raw_path"])
        if destination.exists():
            with np.load(destination) as cached:
                if not gate5a._is_dag(cached["candidate_adjacency"]):
                    raise RuntimeError(f"cached candidate is not a DAG: {task_id}")
            print(f"[{ordinal}/{len(selected)}] skip cached {task_id}", flush=True)
            continue
        try:
            discovery, _ = _load_dataset(row)
            _predict_and_save(model, discovery, destination)
            with np.load(destination) as saved:
                print(
                    f"[{ordinal}/{len(selected)}] {task_id}: candidate_edges={int(saved['candidate_adjacency'].sum())}",
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            append_error(task_id, "inference", exc, "final")
            print(f"[{ordinal}/{len(selected)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    completed = sum((REPO_ROOT / str(path)).exists() for path in manifest["raw_path"])
    atomic_json(
        {
            "schema_version": "gate5b_inference_receipt_v1",
            "task_count": len(manifest),
            "completed": completed,
            "shard_errors": errors,
            "all_candidates_are_dags": completed == len(manifest),
            "truth_loaded": False,
        },
        RESULTS / "inference_receipt.json",
    )
    if completed != len(manifest):
        raise RuntimeError(f"FINAL inference incomplete: {completed}/{len(manifest)}")


def _stable_seed(graph_seed: int, x: int, y: int, conditioning_set: Iterable[int], offset: int = 0) -> int:
    value = int(ANALYSIS_SEED + offset + 31 * int(graph_seed) + 101 * int(x) + 1009 * int(y))
    for item in conditioning_set:
        value = value * 65537 + int(item) + 1
    return int(value % (2**32 - 1))


def _ci_test(data: np.ndarray, graph_seed: int, x: int, y: int, conditioning_set: list[int], offset: int) -> float:
    from causallearn.utils.cit import CIT

    if data.shape[0] < 5 or np.std(data[:, x]) <= 1e-12 or np.std(data[:, y]) <= 1e-12:
        return 1.0
    ci_test = CIT(
        data,
        method="rcit",
        approx="lpd4",
        num_f=RCIT_NUM_F,
        num_f2=RCIT_NUM_F2,
        rcit=True,
    )
    np.random.seed(_stable_seed(graph_seed, x, y, conditioning_set, offset))
    p_value = float(ci_test(x, y, conditioning_set))
    if not np.isfinite(p_value):
        raise ValueError("RCIT returned a non-finite p-value")
    return min(1.0, max(0.0, p_value))


def _write_task_json(payload: dict[str, object], path: Path) -> None:
    atomic_json(payload, path)


def _ci_witnesses_for_task(row: dict[str, object], destination: Path) -> dict[str, object]:
    candidate = _load_candidate(row)
    _, validation = _load_dataset(row)
    statements = gate5a.local_markov_statements(candidate)
    p_values: list[float] = []
    enriched: list[dict[str, object]] = []
    graph_seed = int(row["graph_seed"])
    for statement in statements:
        x = int(statement["x"])
        y = int(statement["y"])
        conditioning_set = [int(item) for item in statement["conditioning_set"]]
        p_value = _ci_test(validation, graph_seed, x, y, conditioning_set, 0)
        p_values.append(p_value)
        enriched.append(
            {
                "kind": "ci",
                "x": x,
                "y": y,
                "conditioning_set": conditioning_set,
                "p_value": p_value,
            }
        )
    rejected, adjusted = gate5a.holm_adjust(p_values, ALPHA)
    for statement, is_rejected, adjusted_p in zip(enriched, rejected, adjusted):
        statement["holm_adjusted_p_value"] = float(adjusted_p)
        statement["rejected"] = bool(is_rejected)
    result = {
        "schema_version": "gate5b_ci_witnesses_v1",
        "task_id": str(row["task_id"]),
        "domain": str(row["domain"]),
        "candidate_edges": int(candidate.sum()),
        "test_count": len(enriched),
        "rejected_witness_count": int(np.sum(rejected)),
        "ci_test": "RCIT",
        "validation_rows": N_VALIDATION,
        "truth_loaded": False,
        "statements": enriched,
    }
    _write_task_json(result, destination)
    return result


def build_ci() -> None:
    manifest = _load_manifest("final")
    destination_root = FINAL_CACHE / "ci_witnesses"
    destination_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    completed = 0
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = destination_root / f"{task_id}.json"
        try:
            if destination.exists():
                result = json.loads(destination.read_text(encoding="utf-8"))
            else:
                result = _ci_witnesses_for_task(row, destination)
            rejected = [item for item in result["statements"] if item["rejected"]]
            candidate = _load_candidate(row)
            repair = minimum_repair_radius(candidate, rejected, MAX_REPAIR_K)
            rows.append(
                {
                    "task_id": task_id,
                    "domain": str(row["domain"]),
                    "rr_ci": int(repair["rr_lower_bound"]),
                    "rr_ci_exact": bool(repair["rr_exact"]),
                    "ci_witness_count": len(rejected),
                    "ci_test_count": int(result["test_count"]),
                    "ci_dag_candidates_checked": int(repair["dag_candidates_checked"]),
                    "status": "complete",
                    "truth_loaded": False,
                }
            )
            completed += 1
            print(f"[CI {ordinal}/{len(manifest)}] {task_id}: RR={repair['rr_lower_bound']}", flush=True)
        except Exception as exc:
            append_error(task_id, "ci", exc, "final")
            rows.append({"task_id": task_id, "domain": str(row["domain"]), "status": "error", "truth_loaded": False})
            print(f"[CI {ordinal}/{len(manifest)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    frame = pd.DataFrame(rows)
    atomic_csv(frame, RESULTS / "final_ci_predictions.csv")
    atomic_json(
        {
            "schema_version": "gate5b_ci_receipt_v1",
            "task_count": len(manifest),
            "completed": completed,
            "error_count": len(manifest) - completed,
            "validation_only": True,
            "truth_loaded": False,
        },
        RESULTS / "final_ci_receipt.json",
    )
    if completed != len(manifest):
        raise RuntimeError(f"CI baseline incomplete: {completed}/{len(manifest)}")


def _fit_residual(train: np.ndarray, test: np.ndarray, y: int, parents: list[int], seed: int) -> np.ndarray:
    if not parents:
        return test[:, y] - float(np.mean(train[:, y]))
    from sklearn.ensemble import ExtraTreesRegressor

    regressor = ExtraTreesRegressor(
        n_estimators=ANM_N_ESTIMATORS,
        min_samples_leaf=ANM_MIN_SAMPLES_LEAF,
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        random_state=int(seed % (2**31 - 1)),
    )
    regressor.fit(train[:, parents], train[:, y])
    return np.asarray(test[:, y] - regressor.predict(test[:, parents]), dtype=np.float64)


def _anm_witnesses_for_task(row: dict[str, object], destination: Path) -> dict[str, object]:
    candidate = _load_candidate(row)
    _, validation = _load_dataset(row)
    graph_seed = int(row["graph_seed"])
    split_rng = np.random.default_rng(ANM_SPLIT_SEED + graph_seed)
    order = split_rng.permutation(len(validation))
    left = validation[order[: N_VALIDATION // 2]]
    right = validation[order[N_VALIDATION // 2 :]]
    atoms: list[dict[str, object]] = []
    for y in range(N_VARIABLES):
        parents = sorted(int(item) for item in gate5a._parents(candidate, y))
        descendants = gate5a._descendants(candidate, y)
        tested = sorted(set(range(N_VARIABLES)) - {y} - descendants)
        residual_right = _fit_residual(left, right, y, parents, ANM_MODEL_SEED + graph_seed + 17 * y)
        residual_left = _fit_residual(right, left, y, parents, ANM_MODEL_SEED + graph_seed + 17 * y + 1)
        for z in tested:
            p_ab = _ci_test(
                np.column_stack([residual_right, right[:, z]]),
                graph_seed,
                0,
                1,
                [],
                1000003 + 19 * y + z,
            )
            p_ba = _ci_test(
                np.column_stack([residual_left, left[:, z]]),
                graph_seed,
                0,
                1,
                [],
                2000003 + 19 * y + z,
            )
            atoms.append(
                {
                    "kind": "anm",
                    "y": int(y),
                    "z": int(z),
                    "candidate_parents": parents,
                    "p_a_to_b": p_ab,
                    "p_b_to_a": p_ba,
                    "p_atom": min(1.0, 2.0 * min(p_ab, p_ba)),
                }
            )
    p_values = [float(item["p_atom"]) for item in atoms]
    rejected, adjusted = gate5a.holm_adjust(p_values, ALPHA)
    for atom, is_rejected, adjusted_p in zip(atoms, rejected, adjusted):
        atom["holm_adjusted_p_value"] = float(adjusted_p)
        atom["rejected"] = bool(is_rejected)
        atom["repair_signature_rule"] = "same candidate parents for Y and Z remains non-descendant"
    result = {
        "schema_version": "gate5b_anm_witnesses_v1",
        "task_id": str(row["task_id"]),
        "domain": str(row["domain"]),
        "candidate_edges": int(candidate.sum()),
        "validation_rows": N_VALIDATION,
        "split": "deterministic 250/250 A/B",
        "regressor": {
            "class": "ExtraTreesRegressor",
            "n_estimators": ANM_N_ESTIMATORS,
            "min_samples_leaf": ANM_MIN_SAMPLES_LEAF,
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": 1,
        },
        "test_count": len(atoms),
        "rejected_witness_count": int(np.sum(rejected)),
        "truth_loaded": False,
        "witnesses": atoms,
    }
    _write_task_json(result, destination)
    return result


def witness_persists(graph: np.ndarray, witness: dict[str, object]) -> bool:
    kind = str(witness["kind"])
    if kind == "ci":
        return gate5a.d_separated(
            graph,
            int(witness["x"]),
            int(witness["y"]),
            [int(item) for item in witness["conditioning_set"]],
        )
    if kind == "anm":
        y = int(witness["y"])
        z = int(witness["z"])
        parents = {int(item) for item in witness["candidate_parents"]}
        return gate5a._parents(graph, y) == parents and z not in gate5a._descendants(graph, y)
    raise ValueError(f"unknown witness kind: {kind}")


def minimum_repair_radius(
    candidate: np.ndarray,
    rejected_witnesses: list[dict[str, object]],
    max_k: int = MAX_REPAIR_K,
) -> dict[str, object]:
    candidate = np.asarray(candidate, dtype=np.int8)
    if not gate5a._is_dag(candidate):
        raise ValueError("repair search requires a DAG candidate")
    if not rejected_witnesses:
        return {"rr_lower_bound": 0, "rr_exact": True, "searched_max_k": 0, "dag_candidates_checked": 1}
    checked = 0
    for distance in range(max_k + 1):
        for graph in gate5a._graphs_at_distance(candidate, distance):
            checked += 1
            if all(not witness_persists(graph, witness) for witness in rejected_witnesses):
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


def build_anm(phase: str) -> None:
    manifest = _load_manifest(phase)
    destination_root = (DEV_CACHE if phase == "dev" else FINAL_CACHE) / "anm_witnesses"
    destination_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    completed = 0
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        destination = destination_root / f"{task_id}.json"
        try:
            if destination.exists():
                result = json.loads(destination.read_text(encoding="utf-8"))
            else:
                result = _anm_witnesses_for_task(row, destination)
            rejected = [item for item in result["witnesses"] if item["rejected"]]
            repair = minimum_repair_radius(_load_candidate(row), rejected, MAX_REPAIR_K)
            rows.append(
                {
                    "task_id": task_id,
                    "domain": str(row["domain"]),
                    "rr_anm": int(repair["rr_lower_bound"]),
                    "rr_anm_exact": bool(repair["rr_exact"]),
                    "anm_witness_count": len(rejected),
                    "anm_test_count": int(result["test_count"]),
                    "anm_dag_candidates_checked": int(repair["dag_candidates_checked"]),
                    "status": "complete",
                    "truth_loaded": False,
                }
            )
            completed += 1
            print(f"[ANM {phase} {ordinal}/{len(manifest)}] {task_id}: RR={repair['rr_lower_bound']}", flush=True)
        except Exception as exc:
            append_error(task_id, "anm", exc, phase)
            rows.append({"task_id": task_id, "domain": str(row["domain"]), "status": "error", "truth_loaded": False})
            print(f"[ANM {phase} {ordinal}/{len(manifest)}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
    output = RESULTS / ("dev_anm_predictions.csv" if phase == "dev" else "final_anm_predictions.csv")
    atomic_csv(pd.DataFrame(rows), output)
    atomic_json(
        {
            "schema_version": "gate5b_anm_receipt_v1",
            "phase": phase,
            "task_count": len(manifest),
            "completed": completed,
            "error_count": len(manifest) - completed,
            "validation_only": True,
            "truth_loaded": False,
        },
        RESULTS / ("dev_anm_receipt.json" if phase == "dev" else "final_anm_receipt.json"),
    )
    if completed != len(manifest):
        raise RuntimeError(f"ANM witness construction incomplete: {completed}/{len(manifest)}")


def _dev_ci_baseline() -> pd.DataFrame:
    source = pd.read_csv(gate5a.RESULTS / "certificate_predictions.csv")
    return source[["task_id", "rr_lower_bound", "rr_exact", "witness_count", "test_count"]].rename(
        columns={
            "rr_lower_bound": "rr_ci",
            "rr_exact": "rr_ci_exact",
            "witness_count": "ci_witness_count",
            "test_count": "ci_test_count",
        }
    ).assign(status="complete", truth_loaded=False)


def build_certificates(phase: str) -> None:
    manifest = _load_manifest(phase)
    anm_path = RESULTS / ("dev_anm_predictions.csv" if phase == "dev" else "final_anm_predictions.csv")
    if not anm_path.exists():
        raise FileNotFoundError(f"missing ANM predictions: {anm_path}")
    anm = pd.read_csv(anm_path)
    if phase == "dev":
        ci = _dev_ci_baseline()
    else:
        ci_path = RESULTS / "final_ci_predictions.csv"
        if not ci_path.exists():
            raise FileNotFoundError(f"missing CI predictions: {ci_path}")
        ci = pd.read_csv(ci_path)
    merged = manifest[["task_id", "domain"]].merge(ci, on=["task_id"], how="left", suffixes=("", "_ci"))
    merged = merged.merge(anm, on=["task_id", "domain"], how="left", suffixes=("", "_anm"))
    if merged["rr_ci"].isna().any() or merged["rr_anm"].isna().any():
        raise RuntimeError("certificate component is incomplete")
    merged["truth_loaded"] = False
    output = RESULTS / ("dev_certificate_predictions.csv" if phase == "dev" else "certificate_predictions.csv")
    atomic_csv(merged, output)
    atomic_json(
        {
            "schema_version": "gate5b_certificate_receipt_v1",
            "phase": phase,
            "task_count": len(merged),
            "completed": int(len(merged)),
            "error_count": 0,
            "rr_ci_and_rr_anm_separate": True,
            "validation_only": True,
            "truth_loaded": False,
        },
        RESULTS / ("dev_certificate_receipt.json" if phase == "dev" else "certificate_receipt.json"),
    )
    pretruth_files = {
        "protocol": DEV_PROTOCOL if phase == "dev" else FROZEN_PROTOCOL,
        "manifest": DEV_MANIFEST if phase == "dev" else FINAL_MANIFEST,
        "anm_receipt": RESULTS / ("dev_anm_receipt.json" if phase == "dev" else "final_anm_receipt.json"),
        "anm_predictions": RESULTS / ("dev_anm_predictions.csv" if phase == "dev" else "final_anm_predictions.csv"),
        "certificate_receipt": RESULTS / ("dev_certificate_receipt.json" if phase == "dev" else "certificate_receipt.json"),
        "certificate_predictions": output,
    }
    if phase == "final":
        pretruth_files["ci_receipt"] = RESULTS / "final_ci_receipt.json"
        pretruth_files["ci_predictions"] = RESULTS / "final_ci_predictions.csv"
        pretruth_files["inference_receipt"] = RESULTS / "inference_receipt.json"
    atomic_json(
        {
            "schema_version": "gate5b_pre_truth_hashes_v1",
            "phase": phase,
            "files": {name: {"path": repo_relative(path), "sha256": sha256(path)} for name, path in pretruth_files.items()},
            "final_truth_loaded": False,
            "hashes_sealed_before_truth": True,
        },
        RESULTS / ("dev_pre_truth_hashes.json" if phase == "dev" else "pre_truth_hashes.json"),
    )


def _check_pretruth_hashes(phase: str) -> dict[str, object]:
    path = RESULTS / ("dev_pre_truth_hashes.json" if phase == "dev" else "pre_truth_hashes.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    for name, item in receipt["files"].items():
        file_path = REPO_ROOT / str(item["path"])
        checks[name] = file_path.exists() and sha256(file_path) == item["sha256"]
    result = {"checks": checks, "all_match": all(checks.values())}
    if not result["all_match"]:
        raise RuntimeError(f"pre-truth hash mismatch: {result}")
    return result


def _clopper_pearson_lower(successes: int, total: int, confidence: float = 0.95) -> float | None:
    if total <= 0:
        return None
    from scipy.stats import beta

    if successes <= 0:
        return 0.0
    upper_shape = 1 if successes == total else total - successes + 1
    return float(beta.ppf((1.0 - confidence) / 2.0, successes, upper_shape))


def _metric_row(frame: pd.DataFrame, certificate: str, scope: str, domain: str) -> dict[str, object]:
    subset = frame if scope == "pooled" else frame[frame["domain"] == domain]
    complete = subset[subset["status"] == "complete"]
    if complete.empty:
        return {"certificate": certificate, "scope": scope, "domain": domain, "n_attempted": len(subset), "n_complete": 0}
    values = complete[f"rr_{certificate}"]
    valid = values <= complete["true_shd"]
    bad = complete["true_shd"] >= 1
    severe = complete["true_shd"] >= 4
    tight = complete.loc[complete["true_shd"] > 0, f"rr_{certificate}"] / complete.loc[complete["true_shd"] > 0, "true_shd"]
    return {
        "certificate": certificate,
        "scope": scope,
        "domain": domain,
        "n_attempted": len(subset),
        "n_complete": len(complete),
        "n_errors": int((subset["status"] != "complete").sum()),
        "validity_count": int(valid.sum()),
        "validity_rate": float(valid.mean()),
        "validity_lower_95": _clopper_pearson_lower(int(valid.sum()), len(valid)),
        "bad_graph_count": int(bad.sum()),
        "bad_graph_rr_ge_1_count": int((complete.loc[bad, f"rr_{certificate}"] >= 1).sum()),
        "bad_graph_nonzero_rate": float((complete.loc[bad, f"rr_{certificate}"] >= 1).mean()) if bad.any() else None,
        "severe_graph_count": int(severe.sum()),
        "severe_graph_rr_ge_2_count": int((complete.loc[severe, f"rr_{certificate}"] >= 2).sum()),
        "severe_graph_rr_ge_2_rate": float((complete.loc[severe, f"rr_{certificate}"] >= 2).mean()) if severe.any() else None,
        "mean_tightness": float(tight.mean()) if len(tight) else None,
        "exact_radius_rate": float(complete[f"rr_{certificate}_exact"].mean()),
        "mean_witness_count": float(complete[f"{certificate}_witness_count"].mean()),
    }


def _evaluate_frame(frame: pd.DataFrame, phase: str) -> tuple[pd.DataFrame, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    for certificate in ("ci", "anm"):
        metric_rows.append(_metric_row(frame, certificate, "pooled", "ALL"))
        metric_rows.extend(_metric_row(frame, certificate, "domain", domain) for domain in sorted(frame["domain"].unique()))
    metrics = pd.DataFrame(metric_rows)
    pooled = metrics[metrics["scope"] == "pooled"].set_index("certificate")
    errors = int((frame["status"] != "complete").sum())
    error_rate = float(errors / len(frame)) if len(frame) else 1.0
    denominators_ok = all(int(pooled.loc[c, "bad_graph_count"]) > 0 and int(pooled.loc[c, "severe_graph_count"]) > 0 for c in ("ci", "anm"))
    technical_ok = error_rate <= 0.10 and denominators_ok and len(frame) > 0
    ci_validity = float(pooled.loc["ci", "validity_rate"])
    anm_validity = float(pooled.loc["anm", "validity_rate"])
    validity_ok = technical_ok and ci_validity >= 0.95 and anm_validity >= 0.95
    anm_bad = float(pooled.loc["anm", "bad_graph_nonzero_rate"] or 0.0)
    anm_severe = float(pooled.loc["anm", "severe_graph_rr_ge_2_rate"] or 0.0)
    ci_severe = float(pooled.loc["ci", "severe_graph_rr_ge_2_rate"] or 0.0)
    severe_gain = anm_severe - ci_severe
    efficacy_ok = (
        validity_ok
        and anm_bad >= 0.40
        and anm_severe >= 0.50
        and severe_gain >= 0.10
    )
    if not technical_ok:
        decision = "GATE5B_INCONCLUSIVE" if phase == "final" else "GATE5B_DEV_INCONCLUSIVE"
    elif not validity_ok:
        decision = "GATE5B_STOP" if phase == "final" else "GATE5B_DEV_STOP"
    elif efficacy_ok:
        decision = "GATE5B_GO" if phase == "final" else "GATE5B_DEV_GO"
    else:
        decision = "GATE5B_BORDERLINE" if phase == "final" else "GATE5B_DEV_STOP"
    summary = {
        "schema_version": "gate5b_summary_v1",
        "phase": phase,
        "decision": decision,
        "technical_ok": bool(technical_ok),
        "validity_ok": bool(validity_ok),
        "efficacy_ok": bool(efficacy_ok),
        "error_rate": error_rate,
        "ci_validity_rate": ci_validity,
        "anm_validity_rate": anm_validity,
        "anm_bad_graph_nonzero_rate": anm_bad,
        "ci_severe_rr_ge_2_rate": ci_severe,
        "anm_severe_rr_ge_2_rate": anm_severe,
        "severe_gain": severe_gain,
        "gate_thresholds": protocol_payload()["gate"],
        "truth_loaded_after_pretruth_hashes": True,
        "metrics": metrics.to_dict(orient="records"),
    }
    return metrics, summary


def run_dev() -> None:
    if not DEV_MANIFEST.exists():
        prepare_dev()
    dev_anm_predictions = RESULTS / "dev_anm_predictions.csv"
    if not dev_anm_predictions.exists():
        build_anm("dev")
    else:
        cached = pd.read_csv(dev_anm_predictions)
        if len(cached) != N_TASKS or not (cached["status"] == "complete").all():
            build_anm("dev")
    build_certificates("dev")
    print("Gate-5B DEV certificate predictions sealed; run --stage dev-screen", flush=True)


def dev_screen() -> None:
    hash_status = _check_pretruth_hashes("dev")
    predictions = pd.read_csv(RESULTS / "dev_certificate_predictions.csv")
    truth = pd.read_csv(gate5a.RESULTS / "final_predictions.csv")[["task_id", "true_shd"]]
    frame = predictions.merge(truth, on="task_id", how="left", validate="one_to_one")
    frame["status"] = frame["status"].fillna("error")
    frame["validity_ci"] = frame["rr_ci"] <= frame["true_shd"]
    frame["validity_anm"] = frame["rr_anm"] <= frame["true_shd"]
    atomic_csv(frame, DEV_RESULTS / "dev_final_predictions.csv")
    metrics, summary = _evaluate_frame(frame, "dev")
    summary["pretruth_hashes_unchanged"] = hash_status
    atomic_csv(metrics, DEV_RESULTS / "dev_metrics.csv")
    atomic_json(summary, DEV_RESULTS / "dev_summary.json")
    atomic_json(
        {
            "schema_version": "gate5b_dev_truth_open_receipt_v1",
            "truth_source": repo_relative(gate5a.RESULTS / "final_predictions.csv"),
            "truth_loaded_after": ["dev_certificate_predictions.csv", "dev_pre_truth_hashes.json"],
            "truth_used_for_witnesses_or_certificates": False,
            "pretruth_hashes_unchanged": hash_status,
        },
        DEV_RESULTS / "dev_truth_open_receipt.json",
    )
    print(summary["decision"], flush=True)
    if summary["decision"] != "GATE5B_DEV_GO":
        raise RuntimeError(f"Gate-5B DEV-SCREEN failed: {summary}")


def freeze() -> None:
    summary_path = DEV_RESULTS / "dev_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run --stage dev-screen before freeze")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("decision") != "GATE5B_DEV_GO":
        raise RuntimeError("cannot freeze a failed Gate-5B DEV-SCREEN")
    verify_gate5a_snapshot()
    payload = protocol_payload(phase="final")
    payload["dev_decision"] = summary["decision"]
    payload["dev_summary_sha256"] = sha256(summary_path)
    payload["source_hashes"] = source_hashes()
    payload["gate5a_snapshot_sha256"] = sha256(gate5a_snapshot_path())
    atomic_json(payload, FROZEN_PROTOCOL)
    atomic_json(
        {
            "schema_version": "gate5b_freeze_receipt_v1",
            "dev_decision": summary["decision"],
            "protocol_sha256": sha256(FROZEN_PROTOCOL),
            "source_hashes": source_hashes(),
            "gate5a_read_only_verified": True,
            "truth_loaded": True,
        },
        RESULTS / "freeze_receipt.json",
    )
    print("Gate-5B protocol frozen after DEV-SCREEN", flush=True)


def open_final_truth() -> None:
    hash_status = _check_pretruth_hashes("final")
    manifest = _load_manifest("final")
    certificates = pd.read_csv(RESULTS / "certificate_predictions.csv")
    rows: list[dict[str, object]] = []
    for row in manifest.to_dict(orient="records"):
        task_id = str(row["task_id"])
        matches = certificates[certificates["task_id"] == task_id]
        if len(matches) != 1:
            rows.append({"task_id": task_id, "domain": str(row["domain"]), "status": "error"})
            continue
        certificate = matches.iloc[0].to_dict()
        candidate = _load_candidate(row)
        with np.load(REPO_ROOT / str(row["dataset_path"])) as data:
            truth = np.asarray(data["truth_adjacency"], dtype=np.int8)
        if not gate5a._is_dag(truth):
            raise AssertionError(f"ground-truth graph is not a DAG: {task_id}")
        true_shd = sum(
            gate5a._edge_state(candidate, left, right) != gate5a._edge_state(truth, left, right)
            for left, right in gate5a._pair_indices(N_VARIABLES)
        )
        rows.append(
            {
                **certificate,
                "status": "complete",
                "true_shd": int(true_shd),
                "bad_graph": bool(true_shd >= 1),
                "severe_bad_graph": bool(true_shd >= 4),
                "validity_ci": bool(int(certificate["rr_ci"]) <= true_shd),
                "validity_anm": bool(int(certificate["rr_anm"]) <= true_shd),
            }
        )
    final = pd.DataFrame(rows)
    atomic_csv(final, RESULTS / "final_predictions.csv")
    metrics, summary = _evaluate_frame(final, "final")
    summary["pretruth_hashes_unchanged"] = hash_status
    atomic_csv(metrics, RESULTS / "final_metrics.csv")
    atomic_json(summary, RESULTS / "final_summary.json")
    atomic_json(
        {
            "schema_version": "gate5b_final_truth_open_receipt_v1",
            "truth_loaded_after": ["certificate_predictions.csv", "pre_truth_hashes.json"],
            "task_count": len(final),
            "complete_count": int((final["status"] == "complete").sum()),
            "pretruth_hashes_unchanged": hash_status,
            "truth_used_for_witnesses_or_certificates": False,
        },
        RESULTS / "final_truth_open_receipt.json",
    )
    print(summary["decision"], flush=True)


def report() -> None:
    summary = json.loads((RESULTS / "final_summary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "final_metrics.csv")
    lines = [
        f"# Gate-5B result: **{summary['decision']}**",
        "",
        "> Primary certificate: RR_ANM; RR_CI is a separate paired Gate-5A baseline.",
        "",
        "CDFM reads only X_discovery. Both certificate builders use only X_validation and the candidate DAG. The true DAG is opened only after pre-truth hashes.",
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
        "RR_CI and RR_ANM are reported separately. RR_ANM is an additive-noise mechanism certificate and is not a distribution-free CI guarantee.",
    ]
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary["decision"], flush=True)


def verify() -> None:
    gate5a_status = verify_gate5a_snapshot()
    dev_required = [
        DEV_PROTOCOL,
        DEV_MANIFEST,
        DEV_RESULTS / "prepare_receipt.json",
        RESULTS / "dev_anm_predictions.csv",
        RESULTS / "dev_certificate_predictions.csv",
        RESULTS / "dev_pre_truth_hashes.json",
        DEV_RESULTS / "dev_final_predictions.csv",
        DEV_RESULTS / "dev_metrics.csv",
        DEV_RESULTS / "dev_summary.json",
        DEV_RESULTS / "dev_truth_open_receipt.json",
    ]
    final_required = [
        FROZEN_PROTOCOL,
        FINAL_MANIFEST,
        RESULTS / "prepare_final_receipt.json",
        RESULTS / "inference_receipt.json",
        RESULTS / "final_ci_predictions.csv",
        RESULTS / "final_anm_predictions.csv",
        RESULTS / "certificate_predictions.csv",
        RESULTS / "pre_truth_hashes.json",
        RESULTS / "final_predictions.csv",
        RESULTS / "final_metrics.csv",
        RESULTS / "final_summary.json",
        RESULTS / "final_truth_open_receipt.json",
        RESULTS / "REPORT.md",
    ]
    dev_summary = json.loads((DEV_RESULTS / "dev_summary.json").read_text(encoding="utf-8"))
    if dev_summary["decision"] != "GATE5B_DEV_GO":
        payload = {
            "schema_version": "gate5b_verification_v1",
            "scope": "dev_stop",
            "gate5a_unchanged": gate5a_status,
            "required_artifacts": {repo_relative(path): path.exists() for path in dev_required},
            "all_required_artifacts": all(path.exists() for path in dev_required),
            "dev_decision": dev_summary["decision"],
        }
    else:
        hash_status = _check_pretruth_hashes("final")
        payload = {
            "schema_version": "gate5b_verification_v1",
            "scope": "final",
            "gate5a_unchanged": gate5a_status,
            "required_artifacts": {repo_relative(path): path.exists() for path in final_required},
            "all_required_artifacts": all(path.exists() for path in final_required),
            "final_pretruth_hashes_unchanged": hash_status,
            "final_decision": json.loads((RESULTS / "final_summary.json").read_text(encoding="utf-8"))["decision"],
        }
    atomic_json(payload, RESULTS / "verification.json")
    if not payload["all_required_artifacts"]:
        raise RuntimeError(f"Gate-5B verification failed: {payload}")
    print("Gate-5B verification passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "prepare-dev",
            "run-dev",
            "dev-screen",
            "freeze",
            "prepare-final",
            "infer",
            "build-ci",
            "build-anm",
            "build-certificates",
            "open-final-truth",
            "report",
            "verify",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare-dev":
        prepare_dev()
    elif args.stage == "run-dev":
        run_dev()
    elif args.stage == "dev-screen":
        dev_screen()
    elif args.stage == "freeze":
        freeze()
    elif args.stage == "prepare-final":
        prepare_final()
    elif args.stage == "infer":
        infer(shard_index=args.shard_index, shard_count=args.shard_count)
    elif args.stage == "build-ci":
        build_ci()
    elif args.stage == "build-anm":
        build_anm("final")
    elif args.stage == "build-certificates":
        build_certificates("final")
    elif args.stage == "open-final-truth":
        open_final_truth()
    elif args.stage == "report":
        report()
    elif args.stage == "verify":
        verify()
    else:
        prepare_dev()
        run_dev()
        dev_screen()
        freeze()
        prepare_final()
        infer(shard_index=args.shard_index, shard_count=args.shard_count)
        build_ci()
        build_anm("final")
        build_certificates("final")
        open_final_truth()
        report()
        verify()


if __name__ == "__main__":
    main()
