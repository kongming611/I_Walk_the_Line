"""从 X 及已保存的重采样图提取 Gate-2 路由诊断特征。

七个 Gate-0 特征原样复用，新增特征不读取真实图或算法 F1。偶/奇样本的
DirectLiNGAM 图仅用于预注册的重采样稳定性诊断；所有方法共享同一张特征表。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SOLVER_RESULTS_PATH = RESULTS_DIR / "solver_results.csv"
FEATURES_PATH = RESULTS_DIR / "diagnostic_features.csv"
ERRORS_PATH = RESULTS_DIR / "diagnostic_errors.jsonl"
FEATURE_SCHEMA_PATH = RESULTS_DIR / "diagnostic_feature_schema.json"

GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from meta_features import FEATURE_COLUMNS as GATE0_FEATURE_COLUMNS  # noqa: E402
from meta_features import extract_meta_features  # noqa: E402

EXTRA_FEATURE_COLUMNS = [
    "robust_skew_abs_median",
    "robust_tail_q99_iqr_median",
    "max_abs_z_q995",
    "crossfit_residual_dependence_mean",
    "crossfit_residual_dependence_max",
    "residual_skew_abs_median",
    "residual_excess_kurtosis_abs_median",
    "split_graph_stability_jaccard",
    "split_graph_disagreement",
]
FEATURE_COLUMNS = [*GATE0_FEATURE_COLUMNS, *EXTRA_FEATURE_COLUMNS]
CSV_COLUMNS = ["task_id", "split", "domain", "graph_seed", *FEATURE_COLUMNS, "diagnostic_failure", "error_type", "runtime_sec"]
CV_SEED = 2026091702
EPS = 1e-12


def _atomic_csv(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FEATURES_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, FEATURES_PATH)


def _append_error(task_id: str, exc: BaseException) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "task_id": task_id, "error_type": type(exc).__name__,
            "message": str(exc), "timestamp_epoch": time.time(),
        }, ensure_ascii=False) + "\n")


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3 or np.std(left) <= EPS or np.std(right) <= EPS:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return 0.0 if not np.isfinite(value) else value


def _robust_features(x: np.ndarray) -> dict[str, float]:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    robust_z = (x - median) / (1.4826 * mad[None, :] + EPS)
    robust_skew = np.asarray(skew(np.clip(robust_z, -20.0, 20.0), axis=0, bias=False), dtype=float)
    q = np.quantile(robust_z, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)
    iqr = q[3] - q[1]
    tail = (q[4] - q[0]) / (iqr + EPS)
    return {
        "robust_skew_abs_median": float(np.median(np.abs(np.nan_to_num(robust_skew)))),
        "robust_tail_q99_iqr_median": float(np.median(np.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0))),
        "max_abs_z_q995": float(np.quantile(np.max(np.abs(robust_z), axis=1), 0.995)),
    }


def _residual_features(x: np.ndarray) -> dict[str, float]:
    correlation = np.corrcoef(x, rowvar=False)
    pairs = [
        (left, right)
        for left in range(x.shape[1])
        for right in range(left + 1, x.shape[1])
    ]
    pairs.sort(key=lambda pair: abs(float(correlation[pair])), reverse=True)
    pairs = pairs[:5]
    folds = KFold(n_splits=3, shuffle=True, random_state=CV_SEED).split(x)
    folds = list(folds)
    dependencies: list[float] = []
    residual_skews: list[float] = []
    residual_kurtoses: list[float] = []
    for left, right in pairs:
        for predictor_index, outcome_index in [(left, right), (right, left)]:
            predictor = x[:, predictor_index]
            outcome = x[:, outcome_index]
            heldout_residual = np.empty(x.shape[0], dtype=float)
            for train, test in folds:
                model = LinearRegression().fit(predictor[train, None], outcome[train])
                heldout_residual[test] = outcome[test] - model.predict(predictor[test, None])
            # Rank correlation catches residual dependence even when Pearson is small.
            ranked_predictor = rankdata(predictor)
            ranked_residual = rankdata(heldout_residual)
            dependencies.append(abs(_safe_corr(ranked_predictor, ranked_residual)))
            residual_skews.append(abs(float(skew(heldout_residual, bias=False))))
            residual_kurtoses.append(abs(float(kurtosis(heldout_residual, fisher=True, bias=False))))
    return {
        "crossfit_residual_dependence_mean": float(np.mean(dependencies)),
        "crossfit_residual_dependence_max": float(np.max(dependencies)),
        "residual_skew_abs_median": float(np.median(residual_skews)),
        "residual_excess_kurtosis_abs_median": float(np.median(residual_kurtoses)),
    }


def _split_features(raw_path: Path) -> dict[str, float]:
    with np.load(raw_path) as raw:
        even = np.asarray(raw["lingam_even_adjacency_source_target"], dtype=np.int8)
        odd = np.asarray(raw["lingam_odd_adjacency_source_target"], dtype=np.int8)
    even_set = {(int(a), int(b)) for a, b in zip(*np.nonzero(even))}
    odd_set = {(int(a), int(b)) for a, b in zip(*np.nonzero(odd))}
    union = even_set | odd_set
    intersection = even_set & odd_set
    jaccard = len(intersection) / len(union) if union else 1.0
    disagreement = 1.0 - jaccard
    return {
        "split_graph_stability_jaccard": float(jaccard),
        "split_graph_disagreement": float(disagreement),
    }


def _failure_row(row: dict[str, object], exc: BaseException) -> dict[str, object]:
    result = {column: 0.0 for column in FEATURE_COLUMNS}
    result.update({
        "task_id": str(row["task_id"]), "split": str(row["split"]),
        "domain": str(row["domain"]), "graph_seed": int(row["graph_seed"]),
        "diagnostic_failure": 1, "error_type": type(exc).__name__, "runtime_sec": 0.0,
    })
    return result


def _write_schema() -> None:
    payload = {
        "schema_version": "gate2_diagnostic_features_v1",
        "gate0_feature_columns": list(GATE0_FEATURE_COLUMNS),
        "extra_feature_columns": list(EXTRA_FEATURE_COLUMNS),
        "forbidden_inputs": ["truth_adjacency", "cdfm_f1", "lingam_f1", "mechanism", "noise", "lambda", "task_id"],
        "notes": {
            "crossfit_residual": "3-fold held-out linear residual rank dependence over top-5 absolute Pearson pairs",
            "split_stability": "Jaccard of DirectLiNGAM edge sets from even/odd rows; no truth used",
            "failure_policy": "numeric safe zeros plus diagnostic_failure=1; evaluator forces CDFM fallback",
        },
    }
    temporary = FEATURE_SCHEMA_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, FEATURE_SCHEMA_PATH)


def run(*, manifest_path: Path = MANIFEST_PATH, split: str | None = None,
        limit: int | None = None, offset: int = 0, rerun: bool = False,
        checkpoint_every: int = 10) -> None:
    if not SOLVER_RESULTS_PATH.exists():
        raise FileNotFoundError(SOLVER_RESULTS_PATH)
    manifest = pd.read_csv(manifest_path)
    if split is not None:
        manifest = manifest[manifest["split"] == split]
    if offset:
        manifest = manifest.iloc[offset:]
    if limit is not None:
        manifest = manifest.iloc[:limit]
    solver = pd.read_csv(SOLVER_RESULTS_PATH, dtype={"task_id": str})
    solver_by_id = {str(item["task_id"]): item for item in solver.to_dict(orient="records")}
    existing = {} if rerun or not FEATURES_PATH.exists() else {
        str(item["task_id"]): item for item in pd.read_csv(FEATURES_PATH).to_dict(orient="records")
    }
    _write_schema()
    total = len(manifest)
    dirty = 0
    for ordinal, row in enumerate(manifest.to_dict(orient="records"), start=1):
        task_id = str(row["task_id"])
        if task_id in existing and not rerun and int(existing[task_id].get("diagnostic_failure", 1)) == 0:
            solver_row = solver_by_id.get(task_id)
            saved_raw = solver_row.get("raw_prediction_path", "") if solver_row else ""
            if saved_raw and (ROOT / str(saved_raw)).exists():
                print(f"[{ordinal}/{total}] skip {task_id}", flush=True)
                continue
            print(f"[{ordinal}/{total}] repair missing raw {task_id}", flush=True)
        start = time.perf_counter()
        try:
            solver_row = solver_by_id.get(task_id)
            if solver_row is None or str(solver_row.get("status", "")) != "ok":
                raise RuntimeError("solver result is missing or technically failed")
            raw_path = ROOT / str(solver_row["raw_prediction_path"])
            with np.load(raw_path) as raw:
                x = np.asarray(raw["X"], dtype=float)
            base = extract_meta_features(x)
            extras = {**_robust_features(x), **_residual_features(x), **_split_features(raw_path)}
            if not np.isfinite(np.asarray([*base.values(), *extras.values()], dtype=float)).all():
                raise ValueError("diagnostic feature contains non-finite value")
            result = {
                "task_id": task_id, "split": str(row["split"]), "domain": str(row["domain"]),
                "graph_seed": int(row["graph_seed"]), **base, **extras,
                "diagnostic_failure": 0, "error_type": "",
                "runtime_sec": float(time.perf_counter() - start),
            }
            print(f"[{ordinal}/{total}] {task_id}: gain={base['nonlinearity_gain_mean']:.4f}, stability={extras['split_graph_stability_jaccard']:.3f}", flush=True)
        except Exception as exc:
            _append_error(task_id, exc)
            result = _failure_row(row, exc)
            result["runtime_sec"] = float(time.perf_counter() - start)
            print(f"[{ordinal}/{total}] ERROR {task_id}: {type(exc).__name__}: {exc}", flush=True)
        existing[task_id] = result
        dirty += 1
        if dirty >= checkpoint_every:
            _atomic_csv([existing[key] for key in sorted(existing)])
            dirty = 0
    if dirty or not FEATURES_PATH.exists():
        _atomic_csv([existing[key] for key in sorted(existing)])
    selected_ids = set(manifest["task_id"].astype(str))
    failures = sum(int(existing[key].get("diagnostic_failure", 1)) for key in selected_ids if key in existing)
    print(f"Diagnostics complete: {len(selected_ids) - failures} successful, {failures} failures, expected={len(selected_ids)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split", choices=["train", "dev", "calibration", "test"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    run(manifest_path=args.manifest, split=args.split, limit=args.limit, offset=args.offset,
        rerun=args.rerun, checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
