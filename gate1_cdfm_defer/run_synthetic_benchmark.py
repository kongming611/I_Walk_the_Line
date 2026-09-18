"""运行 Gate-1 synthetic strata，并只用冻结 Router 做推理。"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from causallearn.search.FCMBased import lingam
from cdfm import CDFM

from freeze_router import MANIFEST_PATH as FREEZE_MANIFEST_PATH
from freeze_router import MODEL_PATH as FROZEN_MODEL_PATH
from freeze_router import ROUTER_THRESHOLD, verify as verify_frozen_router
from generate_synthetic import MANIFEST_PATH, ROOT, generate_all


REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from meta_features import FEATURE_COLUMNS, extract_meta_features  # noqa: E402
from metrics import (  # noqa: E402
    directed_graph_metrics,
    lingam_target_source_to_source_target,
    validate_adjacency,
)


RESULTS_DIR = ROOT / "results" / "synthetic"
RAW_DIR = RESULTS_DIR / "raw_predictions"
RESULTS_PATH = RESULTS_DIR / "per_dataset_results.csv"
ERRORS_PATH = RESULTS_DIR / "errors.jsonl"
ENVIRONMENT_PATH = RESULTS_DIR / "environment.json"
DEFER_MARGIN = 0.03

RESULT_COLUMNS = [
    "dataset_id",
    "stratum",
    "base_seed",
    "lambda",
    "n_samples",
    "n_variables",
    "mechanism",
    "noise",
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
    "feature_runtime_sec",
    "cdfm_threshold",
    "winner",
    "winner_margin",
    *FEATURE_COLUMNS,
    "router_defer_probability",
    "router_prediction",
    "router_target",
    "selected_solver",
    "hybrid_f1",
    "oracle_f1",
    "raw_prediction_path",
]


def _atomic_write(rows: list[dict[str, object]]) -> None:
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


def _write_environment(device: str, cdfm_model: CDFM) -> None:
    freeze_manifest = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [
                "cdfm-base",
                "causal-learn",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "torch",
                "joblib",
            ]
        },
        "cdfm_info": cdfm_model.info,
        "cdfm_threshold_policy": "official auto calibration; model.predict(X) without override",
        "direct_lingam": {"measure": "pwling", "nonzero_tolerance": 1e-8},
        "frozen_router_model_sha256": freeze_manifest["model_sha256"],
        "frozen_router_refit": False,
    }
    ENVIRONMENT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(*, stratum: str | None = None, limit: int | None = None, rerun: bool = False) -> None:
    verify_frozen_router()
    if not MANIFEST_PATH.exists():
        generate_all()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    if stratum is not None:
        manifest = manifest[manifest["stratum"] == stratum]
        if manifest.empty:
            raise ValueError(f"unknown or empty stratum: {stratum}")
    if limit is not None:
        manifest = manifest.iloc[:limit]

    if rerun or not RESULTS_PATH.exists():
        rows_by_id: dict[str, dict[str, object]] = {}
    else:
        existing = pd.read_csv(RESULTS_PATH).to_dict(orient="records")
        rows_by_id = {str(row["dataset_id"]): row for row in existing}

    router = joblib.load(FROZEN_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CDFM on {device}; frozen Router refit is disabled.", flush=True)
    cdfm_model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _write_environment(device, cdfm_model)
    total = len(manifest)

    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        dataset_id = str(row["dataset_id"])
        if dataset_id in rows_by_id:
            print(f"[{ordinal}/{total}] skip {dataset_id}", flush=True)
            continue
        try:
            with np.load(ROOT / str(row["dataset_path"])) as payload:
                x = np.asarray(payload["X"], dtype=np.float64)
                truth = validate_adjacency(payload["adjacency"], name="ground truth")
            if x.shape != (int(row["n_samples"]), int(row["n_variables"])):
                raise ValueError("dataset shape does not match manifest")
            if not np.isfinite(x).all() or np.any(x.std(axis=0) <= 0):
                raise ValueError("dataset failed finite/nonconstant validation")

            cdfm_result = cdfm_model.predict(x)
            cdfm_adjacency = validate_adjacency(cdfm_result.adjacency, name="CDFM adjacency")
            cdfm_metrics = directed_graph_metrics(cdfm_adjacency, truth)

            lingam_model = lingam.DirectLiNGAM(measure="pwling")
            lingam_start = time.perf_counter()
            lingam_model.fit(x)
            lingam_runtime = time.perf_counter() - lingam_start
            lingam_coefficients = np.asarray(lingam_model.adjacency_matrix_, dtype=float)
            lingam_adjacency = lingam_target_source_to_source_target(lingam_coefficients)
            lingam_metrics = directed_graph_metrics(lingam_adjacency, truth)

            feature_start = time.perf_counter()
            meta = extract_meta_features(x)
            feature_runtime = time.perf_counter() - feature_start
            router_x = np.array([[meta[name] for name in FEATURE_COLUMNS]], dtype=float)
            defer_probability = float(router.predict_proba(router_x)[0, 1])
            router_prediction = int(defer_probability >= ROUTER_THRESHOLD)
            selected_solver = "DirectLiNGAM" if router_prediction else "CDFM"
            hybrid_f1 = float(lingam_metrics["f1"] if router_prediction else cdfm_metrics["f1"])
            oracle_f1 = max(float(cdfm_metrics["f1"]), float(lingam_metrics["f1"]))
            router_target = int(
                float(lingam_metrics["f1"]) >= float(cdfm_metrics["f1"]) + DEFER_MARGIN
            )
            if cdfm_metrics["f1"] > lingam_metrics["f1"]:
                winner = "cdfm"
            elif lingam_metrics["f1"] > cdfm_metrics["f1"]:
                winner = "lingam"
            else:
                winner = "tie"

            raw_path = RAW_DIR / f"{dataset_id}.npz"
            np.savez_compressed(
                raw_path,
                truth_adjacency=truth,
                cdfm_adjacency=cdfm_adjacency,
                cdfm_probabilities=np.asarray(cdfm_result.probabilities, dtype=float),
                cdfm_logits=np.asarray(cdfm_result.logits, dtype=float),
                lingam_coefficients_target_source=lingam_coefficients,
                lingam_adjacency_source_target=lingam_adjacency,
                lingam_causal_order=np.asarray(lingam_model.causal_order_, dtype=int),
                router_features=router_x[0],
                router_defer_probability=np.float64(defer_probability),
            )
            result_row = {
                "dataset_id": dataset_id,
                "stratum": str(row["stratum"]),
                "base_seed": int(row["base_seed"]),
                "lambda": float(row["lambda"]),
                "n_samples": int(row["n_samples"]),
                "n_variables": int(row["n_variables"]),
                "mechanism": str(row["mechanism"]),
                "noise": str(row["noise"]),
                "edge_count": int(row["edge_count"]),
                "cdfm_f1": float(cdfm_metrics["f1"]),
                "lingam_f1": float(lingam_metrics["f1"]),
                "cdfm_precision": float(cdfm_metrics["precision"]),
                "lingam_precision": float(lingam_metrics["precision"]),
                "cdfm_recall": float(cdfm_metrics["recall"]),
                "lingam_recall": float(lingam_metrics["recall"]),
                "cdfm_shd": int(cdfm_metrics["shd"]),
                "lingam_shd": int(lingam_metrics["shd"]),
                "cdfm_runtime_sec": float(cdfm_result.runtime_sec),
                "lingam_runtime_sec": float(lingam_runtime),
                "feature_runtime_sec": float(feature_runtime),
                "cdfm_threshold": float(cdfm_result.threshold),
                "winner": winner,
                "winner_margin": abs(float(cdfm_metrics["f1"]) - float(lingam_metrics["f1"])),
                **meta,
                "router_defer_probability": defer_probability,
                "router_prediction": router_prediction,
                "router_target": router_target,
                "selected_solver": selected_solver,
                "hybrid_f1": hybrid_f1,
                "oracle_f1": oracle_f1,
                "raw_prediction_path": raw_path.relative_to(ROOT).as_posix(),
            }
            rows_by_id[dataset_id] = result_row
            _atomic_write([rows_by_id[key] for key in sorted(rows_by_id)])
            print(
                f"[{ordinal}/{total}] {dataset_id}: CDFM={cdfm_metrics['f1']:.4f}, "
                f"LiNGAM={lingam_metrics['f1']:.4f}, router={selected_solver}",
                flush=True,
            )
        except Exception as exc:
            _append_error(dataset_id, exc)
            print(
                f"[{ordinal}/{total}] ERROR {dataset_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )

    expected_ids = set(manifest["dataset_id"].astype(str))
    succeeded = len(expected_ids & set(rows_by_id))
    print(f"Synthetic benchmark complete: {succeeded}/{len(expected_ids)}", flush=True)
    if succeeded != len(expected_ids):
        raise RuntimeError(f"synthetic benchmark incomplete; inspect {ERRORS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stratum", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(stratum=args.stratum, limit=args.limit, rerun=args.rerun)


if __name__ == "__main__":
    main()
