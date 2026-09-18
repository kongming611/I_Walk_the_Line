"""真实运行 CDFM 与 DirectLiNGAM，并保存逐数据集原始结果。"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from causallearn.search.FCMBased import lingam
from cdfm import CDFM

from generate_data import MANIFEST_PATH, ROOT, generate_all
from metrics import (
    directed_graph_metrics,
    lingam_target_source_to_source_target,
    validate_adjacency,
)


RESULTS_DIR = ROOT / "results"
RAW_PREDICTIONS_DIR = RESULTS_DIR / "raw_predictions"
RESULTS_PATH = RESULTS_DIR / "per_dataset_results.csv"
ERRORS_PATH = RESULTS_DIR / "errors.jsonl"
ENVIRONMENT_PATH = RESULTS_DIR / "environment.json"

RESULT_COLUMNS = [
    "dataset_id",
    "base_seed",
    "lambda",
    "n_samples",
    "n_variables",
    "edge_count",
    "cdfm_f1",
    "lingam_f1",
    "cdfm_precision",
    "lingam_precision",
    "cdfm_recall",
    "lingam_recall",
    "cdfm_shd",
    "lingam_shd",
    "cdfm_runtime_sec",
    "lingam_runtime_sec",
    "cdfm_threshold",
    "winner",
    "winner_margin",
    "raw_prediction_path",
]


def _atomic_write_csv(rows: list[dict[str, object]]) -> None:
    temp_path = RESULTS_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, RESULTS_PATH)


def _append_error(dataset_id: str, exc: BaseException) -> None:
    record = {
        "dataset_id": dataset_id,
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
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in [
                "cdfm-base",
                "causal-learn",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "torch",
            ]
        },
        "cdfm_info": model.info,
        "lingam_embedded_version": str(getattr(lingam, "__version__", "unknown")),
        "matrix_convention": "A[source,target]",
        "cdfm_threshold_policy": "official auto calibration; model.predict(X) without override",
        "direct_lingam": {"measure": "pwling", "nonzero_tolerance": 1e-8},
    }
    ENVIRONMENT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_rows() -> list[dict[str, object]]:
    if not RESULTS_PATH.exists():
        return []
    frame = pd.read_csv(RESULTS_PATH)
    return frame.to_dict(orient="records")


def run_benchmark(*, limit: int | None = None, rerun: bool = False) -> None:
    if not MANIFEST_PATH.exists():
        generate_all()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CDFM on {device} with official auto threshold...")
    cdfm_model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _write_environment(device, cdfm_model)

    manifest = pd.read_csv(MANIFEST_PATH)
    if limit is not None:
        manifest = manifest.iloc[:limit]
    existing_rows = [] if rerun else _load_existing_rows()
    completed = {str(row["dataset_id"]) for row in existing_rows}
    rows_by_id = {str(row["dataset_id"]): row for row in existing_rows}
    total = len(manifest)

    for ordinal, manifest_row in enumerate(manifest.to_dict(orient="records"), start=1):
        dataset_id = str(manifest_row["dataset_id"])
        if dataset_id in completed:
            print(f"[{ordinal}/{total}] skip {dataset_id}")
            continue
        try:
            dataset_path = ROOT / str(manifest_row["dataset_path"])
            with np.load(dataset_path) as payload:
                x = np.asarray(payload["X"], dtype=np.float64)
                truth = validate_adjacency(payload["adjacency"], name="ground truth")
            if x.shape != (int(manifest_row["n_samples"]), int(manifest_row["n_variables"])):
                raise ValueError("dataset shape does not match manifest")
            if not np.isfinite(x).all() or np.any(x.std(axis=0) <= 0):
                raise ValueError("dataset failed finite/nonconstant validation")

            cdfm_result = cdfm_model.predict(x)
            cdfm_adjacency = validate_adjacency(cdfm_result.adjacency, name="CDFM adjacency")
            cdfm_probabilities = np.asarray(cdfm_result.probabilities, dtype=float)
            if cdfm_probabilities.shape != truth.shape or not np.isfinite(cdfm_probabilities).all():
                raise ValueError("CDFM probabilities are invalid")
            cdfm_metrics = directed_graph_metrics(cdfm_adjacency, truth)

            lingam_model = lingam.DirectLiNGAM(measure="pwling")
            lingam_start = time.perf_counter()
            lingam_model.fit(x)
            lingam_runtime = time.perf_counter() - lingam_start
            lingam_coefficients = np.asarray(lingam_model.adjacency_matrix_, dtype=float)
            lingam_adjacency = lingam_target_source_to_source_target(lingam_coefficients)
            lingam_metrics = directed_graph_metrics(lingam_adjacency, truth)

            if cdfm_metrics["f1"] > lingam_metrics["f1"]:
                winner = "cdfm"
            elif lingam_metrics["f1"] > cdfm_metrics["f1"]:
                winner = "lingam"
            else:
                winner = "tie"

            raw_path = RAW_PREDICTIONS_DIR / f"{dataset_id}.npz"
            np.savez_compressed(
                raw_path,
                truth_adjacency=truth,
                cdfm_adjacency=cdfm_adjacency,
                cdfm_probabilities=cdfm_probabilities,
                cdfm_logits=np.asarray(cdfm_result.logits, dtype=float),
                lingam_coefficients_target_source=lingam_coefficients,
                lingam_adjacency_source_target=lingam_adjacency,
                lingam_causal_order=np.asarray(lingam_model.causal_order_, dtype=int),
            )
            result_row = {
                "dataset_id": dataset_id,
                "base_seed": int(manifest_row["base_seed"]),
                "lambda": float(manifest_row["lambda"]),
                "n_samples": int(manifest_row["n_samples"]),
                "n_variables": int(manifest_row["n_variables"]),
                "edge_count": int(manifest_row["edge_count"]),
                "cdfm_f1": cdfm_metrics["f1"],
                "lingam_f1": lingam_metrics["f1"],
                "cdfm_precision": cdfm_metrics["precision"],
                "lingam_precision": lingam_metrics["precision"],
                "cdfm_recall": cdfm_metrics["recall"],
                "lingam_recall": lingam_metrics["recall"],
                "cdfm_shd": cdfm_metrics["shd"],
                "lingam_shd": lingam_metrics["shd"],
                "cdfm_runtime_sec": float(cdfm_result.runtime_sec),
                "lingam_runtime_sec": float(lingam_runtime),
                "cdfm_threshold": float(cdfm_result.threshold),
                "winner": winner,
                "winner_margin": abs(float(cdfm_metrics["f1"]) - float(lingam_metrics["f1"])),
                "raw_prediction_path": raw_path.relative_to(ROOT).as_posix(),
            }
            rows_by_id[dataset_id] = result_row
            ordered_rows = [rows_by_id[key] for key in sorted(rows_by_id)]
            _atomic_write_csv(ordered_rows)
            print(
                f"[{ordinal}/{total}] {dataset_id}: "
                f"CDFM={cdfm_metrics['f1']:.4f}, LiNGAM={lingam_metrics['f1']:.4f}, "
                f"winner={winner}"
            )
        except Exception as exc:
            _append_error(dataset_id, exc)
            print(f"[{ordinal}/{total}] ERROR {dataset_id}: {type(exc).__name__}: {exc}")

    expected = len(manifest)
    succeeded = sum(str(row["dataset_id"]) in rows_by_id for row in manifest.to_dict(orient="records"))
    print(f"Benchmark complete: {succeeded}/{expected} successful datasets")
    if succeeded != expected:
        raise RuntimeError(f"benchmark incomplete: {succeeded}/{expected}; inspect {ERRORS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run_benchmark(limit=args.limit, rerun=args.rerun)


if __name__ == "__main__":
    main()

