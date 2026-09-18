"""运行 Tübingen strict-95 官方加权二元因果方向评测。"""

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

from freeze_router import MODEL_PATH as FROZEN_MODEL_PATH
from freeze_router import ROUTER_THRESHOLD, verify as verify_frozen_router
from prepare_real_data import ROOT


REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from meta_features import FEATURE_COLUMNS, extract_meta_features  # noqa: E402


DATA_DIR = ROOT / "data" / "real" / "tuebingen"
MANIFEST_PATH = DATA_DIR / "strict95_manifest.csv"
RESULTS_DIR = ROOT / "results" / "tuebingen"
RAW_DIR = RESULTS_DIR / "raw_predictions"
RESULTS_PATH = RESULTS_DIR / "per_pair_results.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
ERRORS_PATH = RESULTS_DIR / "errors.jsonl"
ENVIRONMENT_PATH = RESULTS_DIR / "environment.json"

# Aggregate sanity targets are from the CDFM paper. No pair-level reference or
# decoder was published, so the frozen decoders below are never changed based
# on observed accuracy. A difference over 0.05 triggers diagnosis, not tuning.
SANITY_TARGETS = {"cdfm": 0.671, "direct_lingam": 0.508}
SANITY_TOLERANCE = 0.05

RESULT_COLUMNS = [
    "pair_id",
    "n_samples",
    "raw_column_count",
    "selected_raw_column_0",
    "selected_raw_column_1",
    "true_source_local",
    "true_target_local",
    "weight",
    "cdfm_direction",
    "cdfm_correct",
    "cdfm_failure",
    "cdfm_runtime_sec",
    "lingam_direction",
    "lingam_correct",
    "lingam_failure",
    "lingam_runtime_sec",
    *FEATURE_COLUMNS,
    "feature_runtime_sec",
    "router_defer_probability",
    "router_prediction",
    "selected_solver",
    "hybrid_correct",
    "oracle_correct",
    "raw_prediction_path",
]


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_rows(rows: list[dict[str, object]]) -> None:
    temporary = RESULTS_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, RESULTS_PATH)


def _append_error(pair_id: str, solver: str, exc: BaseException) -> None:
    record = {
        "pair_id": pair_id,
        "solver": solver,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp_epoch": time.time(),
    }
    with ERRORS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_pair(row: dict[str, object]) -> tuple[np.ndarray, list[int], int, int]:
    raw = np.loadtxt(ROOT / str(row["path"]))
    cause = int(row["cause_column"])
    effect = int(row["effect_column"])
    selected_columns = sorted([cause, effect])
    x = np.asarray(raw[:, selected_columns], dtype=np.float64)
    if x.shape != (int(row["n_samples"]), 2) or not np.isfinite(x).all():
        raise RuntimeError("invalid selected Tübingen pair matrix")
    if np.any(x.std(axis=0) <= 0):
        raise RuntimeError("constant Tübingen pair column")
    return x, selected_columns, selected_columns.index(cause), selected_columns.index(effect)


def _direction_from_cdfm_scores(scores: np.ndarray) -> str:
    values = np.asarray(scores, dtype=float)
    if values.shape != (2, 2) or not np.isfinite(values).all():
        raise RuntimeError("CDFM bivariate score matrix is invalid")
    if values[0, 1] > values[1, 0]:
        return "0->1"
    if values[1, 0] > values[0, 1]:
        return "1->0"
    raise RuntimeError("CDFM bivariate off-diagonal scores are tied")


def _direction_from_lingam_order(order: object) -> str:
    values = [int(value) for value in order]
    if sorted(values) != [0, 1]:
        raise RuntimeError(f"DirectLiNGAM returned invalid bivariate order: {values}")
    return f"{values[0]}->{values[1]}"


def _weighted_accuracy(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column].astype(float), weights=frame["weight"].astype(float)))


def _write_environment(cdfm_model: CDFM, device: str) -> None:
    packages = {
        name: importlib.metadata.version(name)
        for name in ["cdfm-base", "causal-learn", "numpy", "pandas", "scikit-learn", "torch"]
    }
    _atomic_json(
        ENVIRONMENT_PATH,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "cuda_available": bool(torch.cuda.is_available()),
            "packages": packages,
            "cdfm_info": cdfm_model.info,
            "strict_pair_count": 95,
            "denominator_policy": "all 95 weighted pairs; failures and ties count as incorrect",
            "cdfm_decoder": "argmax between score[0,1] and score[1,0]",
            "cdfm_decoder_score": "probabilities",
            "direct_lingam_decoder": "first-to-second variable in causal_order_",
            "router_refit": False,
            "sanity_targets": SANITY_TARGETS,
            "sanity_tolerance": SANITY_TOLERANCE,
        },
    )


def run(*, limit: int | None = None, rerun: bool = False) -> None:
    verify_frozen_router()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    if len(manifest) != 95:
        raise RuntimeError("Tübingen strict manifest must contain 95 pairs")
    if limit is not None:
        manifest = manifest.iloc[:limit]
    rows_by_id = {} if rerun or not RESULTS_PATH.exists() else {
        str(row["pair_id"]): row for row in pd.read_csv(RESULTS_PATH).to_dict(orient="records")
    }

    router = joblib.load(FROZEN_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cdfm_model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _write_environment(cdfm_model, device)

    for ordinal, manifest_row in enumerate(manifest.to_dict(orient="records"), start=1):
        pair_id = str(manifest_row["pair_id"])
        if pair_id in rows_by_id:
            print(f"[{ordinal}/{len(manifest)}] skip {pair_id}", flush=True)
            continue
        x, selected_columns, true_source, true_target = _load_pair(manifest_row)
        true_direction = f"{true_source}->{true_target}"
        raw_payload: dict[str, np.ndarray] = {"X": x}

        cdfm_direction = "failure"
        cdfm_correct = 0
        cdfm_failure = ""
        cdfm_runtime = float("nan")
        try:
            result = cdfm_model.predict(x)
            cdfm_runtime = float(result.runtime_sec)
            cdfm_direction = _direction_from_cdfm_scores(result.probabilities)
            cdfm_correct = int(cdfm_direction == true_direction)
            raw_payload.update(
                cdfm_probabilities=np.asarray(result.probabilities, dtype=float),
                cdfm_logits=np.asarray(result.logits, dtype=float),
                cdfm_adjacency=np.asarray(result.adjacency, dtype=np.int8),
                cdfm_threshold=np.asarray(result.threshold, dtype=float),
            )
        except Exception as exc:
            cdfm_failure = f"{type(exc).__name__}: {exc}"
            _append_error(pair_id, "CDFM", exc)

        lingam_direction = "failure"
        lingam_correct = 0
        lingam_failure = ""
        lingam_runtime = float("nan")
        try:
            model = lingam.DirectLiNGAM(measure="pwling")
            started = time.perf_counter()
            model.fit(x)
            lingam_runtime = time.perf_counter() - started
            lingam_direction = _direction_from_lingam_order(model.causal_order_)
            lingam_correct = int(lingam_direction == true_direction)
            raw_payload.update(
                lingam_causal_order=np.asarray(model.causal_order_, dtype=int),
                lingam_coefficients_target_source=np.asarray(model.adjacency_matrix_, dtype=float),
            )
        except Exception as exc:
            lingam_failure = f"{type(exc).__name__}: {exc}"
            _append_error(pair_id, "DirectLiNGAM", exc)

        feature_started = time.perf_counter()
        features = extract_meta_features(x)
        feature_runtime = time.perf_counter() - feature_started
        router_x = np.array([[features[name] for name in FEATURE_COLUMNS]], dtype=float)
        defer_probability = float(router.predict_proba(router_x)[0, 1])
        router_prediction = int(defer_probability >= ROUTER_THRESHOLD)
        selected_solver = "DirectLiNGAM" if router_prediction else "CDFM"
        hybrid_correct = lingam_correct if router_prediction else cdfm_correct
        oracle_correct = max(cdfm_correct, lingam_correct)
        raw_payload.update(
            router_features=router_x[0],
            router_defer_probability=np.asarray(defer_probability),
            true_direction=np.asarray(true_direction),
        )
        raw_path = RAW_DIR / f"{pair_id}.npz"
        np.savez_compressed(raw_path, **raw_payload)
        rows_by_id[pair_id] = {
            "pair_id": pair_id,
            "n_samples": int(manifest_row["n_samples"]),
            "raw_column_count": int(manifest_row["raw_column_count"]),
            "selected_raw_column_0": selected_columns[0],
            "selected_raw_column_1": selected_columns[1],
            "true_source_local": true_source,
            "true_target_local": true_target,
            "weight": float(manifest_row["weight"]),
            "cdfm_direction": cdfm_direction,
            "cdfm_correct": cdfm_correct,
            "cdfm_failure": cdfm_failure,
            "cdfm_runtime_sec": cdfm_runtime,
            "lingam_direction": lingam_direction,
            "lingam_correct": lingam_correct,
            "lingam_failure": lingam_failure,
            "lingam_runtime_sec": lingam_runtime,
            **features,
            "feature_runtime_sec": feature_runtime,
            "router_defer_probability": defer_probability,
            "router_prediction": router_prediction,
            "selected_solver": selected_solver,
            "hybrid_correct": hybrid_correct,
            "oracle_correct": oracle_correct,
            "raw_prediction_path": raw_path.relative_to(ROOT).as_posix(),
        }
        _atomic_rows([rows_by_id[key] for key in sorted(rows_by_id)])
        print(
            f"[{ordinal}/{len(manifest)}] {pair_id}: true={true_direction}, "
            f"CDFM={cdfm_direction}, LiNGAM={lingam_direction}, Router={selected_solver}",
            flush=True,
        )

    expected = set(manifest["pair_id"].astype(str))
    if not expected.issubset(rows_by_id):
        raise RuntimeError("Tübingen benchmark is incomplete")
    if limit is not None:
        print(f"Tübingen smoke complete: {len(expected)} pairs", flush=True)
        return

    frame = pd.DataFrame([rows_by_id[key] for key in sorted(expected)])
    scores = {
        "always_cdfm": _weighted_accuracy(frame, "cdfm_correct"),
        "always_direct_lingam": _weighted_accuracy(frame, "lingam_correct"),
        "frozen_hybrid": _weighted_accuracy(frame, "hybrid_correct"),
        "oracle": _weighted_accuracy(frame, "oracle_correct"),
    }
    scores["best_single"] = max(scores["always_cdfm"], scores["always_direct_lingam"])
    scores["hybrid_minus_best_single"] = scores["frozen_hybrid"] - scores["best_single"]
    scores["oracle_gap"] = scores["oracle"] - scores["best_single"]
    scores["captured_gap"] = (
        scores["hybrid_minus_best_single"] / scores["oracle_gap"]
        if scores["oracle_gap"] > 0
        else None
    )
    sanity = {
        "cdfm_absolute_delta": abs(scores["always_cdfm"] - SANITY_TARGETS["cdfm"]),
        "direct_lingam_absolute_delta": abs(
            scores["always_direct_lingam"] - SANITY_TARGETS["direct_lingam"]
        ),
    }
    sanity["passed"] = bool(
        sanity["cdfm_absolute_delta"] <= SANITY_TOLERANCE
        and sanity["direct_lingam_absolute_delta"] <= SANITY_TOLERANCE
    )
    _atomic_json(
        SUMMARY_PATH,
        {
            "pair_count": len(frame),
            "total_weight": float(frame["weight"].sum()),
            "technical_failures": {
                "cdfm": int((frame["cdfm_direction"] == "failure").sum()),
                "direct_lingam": int((frame["lingam_direction"] == "failure").sum()),
            },
            "scores": scores,
            "sanity_targets": SANITY_TARGETS,
            "sanity_tolerance": SANITY_TOLERANCE,
            "sanity": sanity,
        },
    )
    print(
        f"Tübingen: CDFM={scores['always_cdfm']:.4f}, "
        f"LiNGAM={scores['always_direct_lingam']:.4f}, Hybrid={scores['frozen_hybrid']:.4f}, "
        f"sanity={'PASS' if sanity['passed'] else 'FAIL'}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    run(limit=args.limit, rerun=args.rerun)


if __name__ == "__main__":
    main()
