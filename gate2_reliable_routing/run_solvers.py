"""Gate-2 的算法运行器。

本文件只复用 Gate-0 的 CDFM/DirectLiNGAM 入口与 directed-F1 计分器，
不修改生产代码或既有 Gate-0/Gate-1 结果。每个任务一个 raw ``npz``，CSV
是可恢复的账本；技术失败写入单独的 jsonl，并仍留在预期分母中。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causallearn.search.FCMBased import lingam
from cdfm import CDFM

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_DIR = RESULTS_DIR / "raw_predictions"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
FREEZE_MANIFEST = ROOT / "frozen" / "freeze_manifest.json"
RESULTS_PATH = RESULTS_DIR / "solver_results.csv"
ERRORS_PATH = RESULTS_DIR / "solver_errors.jsonl"
ENVIRONMENT_PATH = RESULTS_DIR / "solver_environment.json"

GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from metrics import (  # noqa: E402
    directed_graph_metrics,
    lingam_target_source_to_source_target,
    validate_adjacency,
)

RESULT_COLUMNS = [
    "task_id", "split", "domain", "mechanism", "noise", "task_index",
    "graph_seed", "lambda", "n_samples", "n_variables", "edge_count",
    "cdfm_f1", "lingam_f1", "cdfm_precision", "lingam_precision",
    "cdfm_recall", "lingam_recall", "cdfm_shd", "lingam_shd",
    "cdfm_runtime_sec", "lingam_runtime_sec", "lingam_even_runtime_sec",
    "lingam_odd_runtime_sec", "cdfm_threshold", "winner", "winner_margin",
    "raw_prediction_path", "status", "error_type",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, RESULTS_PATH)


def _append_error(task_id: str, exc: BaseException) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task_id,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp_epoch": time.time(),
    }
    with ERRORS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _write_environment(device: str, model: CDFM) -> None:
    payload = {
        "schema_version": "gate2_solver_environment_v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "packages": {name: _package_version(name) for name in [
            "cdfm-base", "causal-learn", "numpy", "pandas", "scipy",
            "scikit-learn", "torch", "joblib",
        ]},
        "cdfm_info": model.info,
        "matrix_convention": "A[source,target]; DirectLiNGAM B[target,source] converted by transpose",
        "cdfm_threshold_policy": "official auto calibration; model.predict(X) without override",
        "direct_lingam": {"measure": "pwling", "nonzero_tolerance": 1e-8},
        "task_manifest_sha256": _sha256(MANIFEST_PATH),
    }
    temporary = ENVIRONMENT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, ENVIRONMENT_PATH)


def _run_lingam(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    model = lingam.DirectLiNGAM(measure="pwling")
    model.fit(x)
    runtime = time.perf_counter() - start
    coefficients = np.asarray(model.adjacency_matrix_, dtype=float)
    adjacency = lingam_target_source_to_source_target(coefficients)
    order = np.asarray(model.causal_order_, dtype=int)
    return coefficients, adjacency, order, float(runtime)


def _empty_row(row: dict[str, object], exc: BaseException) -> dict[str, object]:
    result = {column: "" for column in RESULT_COLUMNS}
    for key in ["task_id", "split", "domain", "mechanism", "noise", "task_index",
                "graph_seed", "lambda", "n_samples", "n_variables"]:
        result[key] = row.get(key, "")
    result.update({"status": "error", "error_type": type(exc).__name__})
    return result


def _load_existing() -> dict[str, dict[str, object]]:
    if not RESULTS_PATH.exists():
        return {}
    frame = pd.read_csv(RESULTS_PATH, dtype={"task_id": str})
    return {str(item["task_id"]): item for item in frame.to_dict(orient="records")}


def run(*, manifest_path: Path = MANIFEST_PATH, split: str | None = None,
        limit: int | None = None, offset: int = 0, rerun: bool = False) -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    if split == "test" and not FREEZE_MANIFEST.exists():
        raise RuntimeError("test solver is locked until frozen/freeze_manifest.json exists")
    manifest = pd.read_csv(manifest_path)
    if split is not None:
        manifest = manifest[manifest["split"] == split]
    if offset:
        manifest = manifest.iloc[offset:]
    if limit is not None:
        manifest = manifest.iloc[:limit]
    if manifest.empty:
        raise ValueError("selected solver manifest is empty")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_id = {} if rerun else _load_existing()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CDFM on {device}; selected tasks={len(manifest)}", flush=True)
    cdfm_model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _write_environment(device, cdfm_model)
    total = len(manifest)

    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        if task_id in rows_by_id and not rerun and rows_by_id[task_id].get("status") == "ok":
            saved_raw = rows_by_id[task_id].get("raw_prediction_path", "")
            raw_exists = bool(saved_raw) and (ROOT / str(saved_raw)).exists()
            if raw_exists:
                print(f"[{ordinal}/{total}] skip {task_id}", flush=True)
                continue
            print(f"[{ordinal}/{total}] repair missing raw {task_id}", flush=True)
        try:
            task_path = ROOT / str(row["path"])
            # The generator writes one file per task.  On Windows a freshly
            # renamed file can become visible a few milliseconds after the
            # manifest; retry the read before counting a transient visibility
            # race as a technical failure.
            for _ in range(20):
                if task_path.exists():
                    break
                time.sleep(0.1)
            if not task_path.exists():
                raise FileNotFoundError(task_path)
            with np.load(task_path) as payload:
                x = np.asarray(payload["X"], dtype=np.float64)
                truth = validate_adjacency(payload["truth_adjacency"], name="ground truth")
            if x.shape != (int(row["n_samples"]), int(row["n_variables"])):
                raise ValueError("task shape does not match manifest")
            if not np.isfinite(x).all() or np.any(x.std(axis=0) <= 0):
                raise ValueError("task X failed finite/nonconstant validation")

            cdfm_start = time.perf_counter()
            cdfm_result = cdfm_model.predict(x)
            cdfm_runtime = time.perf_counter() - cdfm_start
            cdfm_adjacency = validate_adjacency(cdfm_result.adjacency, name="CDFM adjacency")
            cdfm_probabilities = np.asarray(cdfm_result.probabilities, dtype=float)
            cdfm_logits = np.asarray(cdfm_result.logits, dtype=float)
            if cdfm_probabilities.shape != truth.shape or not np.isfinite(cdfm_probabilities).all():
                raise ValueError("CDFM probabilities are invalid")
            cdfm_metrics = directed_graph_metrics(cdfm_adjacency, truth)

            coefficients, lingam_adjacency, order, lingam_runtime = _run_lingam(x)
            even_coeff, even_adj, even_order, even_runtime = _run_lingam(x[::2])
            odd_coeff, odd_adj, odd_order, odd_runtime = _run_lingam(x[1::2])
            lingam_metrics = directed_graph_metrics(lingam_adjacency, truth)
            winner = "cdfm" if cdfm_metrics["f1"] > lingam_metrics["f1"] else (
                "lingam" if lingam_metrics["f1"] > cdfm_metrics["f1"] else "tie"
            )
            raw_path = RAW_DIR / f"{task_id}.npz"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = raw_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle, X=x, truth_adjacency=truth,
                    cdfm_adjacency=cdfm_adjacency,
                    cdfm_probabilities=cdfm_probabilities,
                    cdfm_logits=cdfm_logits,
                    lingam_coefficients_target_source=coefficients,
                    lingam_adjacency_source_target=lingam_adjacency,
                    lingam_causal_order=order,
                    lingam_even_coefficients_target_source=even_coeff,
                    lingam_even_adjacency_source_target=even_adj,
                    lingam_even_causal_order=even_order,
                    lingam_odd_coefficients_target_source=odd_coeff,
                    lingam_odd_adjacency_source_target=odd_adj,
                    lingam_odd_causal_order=odd_order,
                )
            os.replace(temporary, raw_path)
            result = {
                "task_id": task_id, "split": str(row["split"]), "domain": str(row["domain"]),
                "mechanism": str(row["mechanism"]), "noise": str(row["noise"]),
                "task_index": int(row["task_index"]), "graph_seed": int(row["graph_seed"]),
                "lambda": float(row["lambda"]), "n_samples": int(row["n_samples"]),
                "n_variables": int(row["n_variables"]), "edge_count": int(truth.sum()),
                "cdfm_f1": float(cdfm_metrics["f1"]), "lingam_f1": float(lingam_metrics["f1"]),
                "cdfm_precision": float(cdfm_metrics["precision"]), "lingam_precision": float(lingam_metrics["precision"]),
                "cdfm_recall": float(cdfm_metrics["recall"]), "lingam_recall": float(lingam_metrics["recall"]),
                "cdfm_shd": int(cdfm_metrics["shd"]), "lingam_shd": int(lingam_metrics["shd"]),
                "cdfm_runtime_sec": float(cdfm_runtime), "lingam_runtime_sec": float(lingam_runtime),
                "lingam_even_runtime_sec": float(even_runtime), "lingam_odd_runtime_sec": float(odd_runtime),
                "cdfm_threshold": float(cdfm_result.threshold), "winner": winner,
                "winner_margin": abs(float(cdfm_metrics["f1"]) - float(lingam_metrics["f1"])),
                "raw_prediction_path": raw_path.relative_to(ROOT).as_posix(),
                "status": "ok", "error_type": "",
            }
            rows_by_id[task_id] = result
            _atomic_csv([rows_by_id[key] for key in sorted(rows_by_id)])
            print(
                f"[{ordinal}/{total}] {task_id}: CDFM={cdfm_metrics['f1']:.4f}, "
                f"LiNGAM={lingam_metrics['f1']:.4f}, winner={winner}", flush=True,
            )
        except Exception as exc:  # technical failure is retained in denominator
            _append_error(task_id, exc)
            rows_by_id[task_id] = _empty_row(row, exc)
            _atomic_csv([rows_by_id[key] for key in sorted(rows_by_id)])
            print(f"[{ordinal}/{total}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)

    selected_ids = set(manifest["task_id"].astype(str))
    completed = sum(key in rows_by_id and rows_by_id[key].get("status") == "ok" for key in selected_ids)
    failed = sum(key in rows_by_id and rows_by_id[key].get("status") == "error" for key in selected_ids)
    print(f"Solver stage complete: ok={completed}, technical_failures={failed}, expected={len(selected_ids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split", choices=["train", "dev", "calibration", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(manifest_path=args.manifest, split=args.split, limit=args.limit, offset=args.offset, rerun=args.rerun)


if __name__ == "__main__":
    main()
