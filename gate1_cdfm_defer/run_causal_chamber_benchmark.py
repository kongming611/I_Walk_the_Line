"""运行 Causal Chamber A1 sanity gate 与 A2 逐环境/ensemble 评测。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from causallearn.search.FCMBased import lingam
from cdfm import CDFM
from sklearn.metrics import roc_auc_score

from freeze_router import MODEL_PATH as FROZEN_MODEL_PATH
from freeze_router import ROUTER_THRESHOLD, verify as verify_frozen_router
from prepare_real_data import A2_ENVIRONMENTS, DAG_VARS, ROOT


REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from meta_features import FEATURE_COLUMNS, extract_meta_features  # noqa: E402
from metrics import (  # noqa: E402
    directed_graph_metrics,
    lingam_target_source_to_source_target,
    validate_adjacency,
)


DATA_DIR = ROOT / "data" / "real" / "causal_chamber"
RESULTS_DIR = ROOT / "results" / "causal_chamber"
RAW_DIR = RESULTS_DIR / "raw_predictions"
A1_RESULT_PATH = RESULTS_DIR / "a1_result.json"
A2_RESULTS_PATH = RESULTS_DIR / "a2_per_environment.csv"
A2_ENSEMBLE_PATH = RESULTS_DIR / "a2_ensemble.json"
ENVIRONMENT_PATH = RESULTS_DIR / "environment.json"
DEFER_MARGIN = 0.03

# 该门只核验公开 notebook 中不依赖阈值选择的图结构结果。模型、数据完全
# 对齐时应为 0.727272...；0.03 容忍数值/依赖版本差异，但不允许借此调阈值。
A1_EXPECTED_CDFM_F1 = 0.7272727272727273
A1_SANITY_TOLERANCE = 0.03


def _atomic_json(path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _truth() -> np.ndarray:
    edges = pd.read_csv(DATA_DIR / "dag.csv")
    index = {name: position for position, name in enumerate(DAG_VARS)}
    truth = np.zeros((len(DAG_VARS), len(DAG_VARS)), dtype=np.int8)
    for edge in edges.itertuples(index=False):
        if edge.source in index and edge.target in index:
            truth[index[edge.source], index[edge.target]] = 1
    truth = validate_adjacency(truth, name="Causal Chamber induced truth")
    if int(truth.sum()) != 39:
        raise RuntimeError("Causal Chamber induced truth must contain 39 edges")
    return truth


def _load_x(path: Path, expected_rows: int) -> np.ndarray:
    frame = pd.read_csv(path)
    x = frame[DAG_VARS].to_numpy(dtype=np.float64)
    if x.shape != (expected_rows, len(DAG_VARS)) or not np.isfinite(x).all():
        raise RuntimeError(f"invalid Causal Chamber matrix: {path}")
    # 强干预环境允许出现常数列。算法仍需接受官方输入；冻结的 Gate-0
    # feature pipeline 若拒绝该矩阵，则单独记录 Router 技术失败。
    return x


def _edge_auroc(scores: np.ndarray, truth: np.ndarray) -> float:
    mask = ~np.eye(truth.shape[0], dtype=bool)
    return float(roc_auc_score(truth[mask], np.asarray(scores, dtype=float)[mask]))


def _router_decision(
    router: object, x: np.ndarray
) -> tuple[dict[str, float], float, float, int, str, str]:
    started = time.perf_counter()
    try:
        features = extract_meta_features(x)
        feature_runtime = time.perf_counter() - started
        router_x = np.array([[features[name] for name in FEATURE_COLUMNS]], dtype=float)
        probability = float(router.predict_proba(router_x)[0, 1])
        prediction = int(probability >= ROUTER_THRESHOLD)
        solver = "DirectLiNGAM" if prediction else "CDFM"
        return features, feature_runtime, probability, prediction, solver, ""
    except Exception as exc:
        # 冻结 Router 的语义是“CDFM 默认、明确 defer 才用 specialist”。不能
        # 为真实数据修改 Gate-0 特征；特征失败时 fail closed 到 generalist。
        feature_runtime = time.perf_counter() - started
        features = {name: float("nan") for name in FEATURE_COLUMNS}
        failure = f"{type(exc).__name__}: {exc}"
        return features, feature_runtime, float("nan"), 0, "CDFM_FEATURE_FAILURE_FALLBACK", failure


def _evaluate_one(
    *,
    dataset_id: str,
    x: np.ndarray,
    truth: np.ndarray,
    cdfm_model: CDFM,
    router: object,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    cdfm_result = cdfm_model.predict(x)
    cdfm_adjacency = validate_adjacency(cdfm_result.adjacency, name="CDFM adjacency")
    cdfm_metrics = directed_graph_metrics(cdfm_adjacency, truth)

    lingam_model = lingam.DirectLiNGAM(measure="pwling")
    lingam_started = time.perf_counter()
    lingam_failure = ""
    try:
        lingam_model.fit(x)
        lingam_runtime = time.perf_counter() - lingam_started
        coefficients = np.asarray(lingam_model.adjacency_matrix_, dtype=float)
        lingam_adjacency = lingam_target_source_to_source_target(coefficients)
        lingam_order = np.asarray(lingam_model.causal_order_, dtype=int)
    except Exception as exc:
        lingam_runtime = time.perf_counter() - lingam_started
        lingam_failure = f"{type(exc).__name__}: {exc}"
        coefficients = np.full_like(truth, np.nan, dtype=float)
        lingam_adjacency = np.zeros_like(truth)
        lingam_order = np.array([], dtype=int)
    lingam_metrics = directed_graph_metrics(lingam_adjacency, truth)

    features, feature_runtime, probability, prediction, solver, router_failure = _router_decision(
        router, x
    )
    hybrid_f1 = float(lingam_metrics["f1"] if prediction else cdfm_metrics["f1"])
    oracle_f1 = max(float(cdfm_metrics["f1"]), float(lingam_metrics["f1"]))
    row: dict[str, object] = {
        "dataset_id": dataset_id,
        "n_samples": int(x.shape[0]),
        "n_variables": int(x.shape[1]),
        "cdfm_f1": float(cdfm_metrics["f1"]),
        "cdfm_precision": float(cdfm_metrics["precision"]),
        "cdfm_recall": float(cdfm_metrics["recall"]),
        "cdfm_shd": int(cdfm_metrics["shd"]),
        "cdfm_auroc": _edge_auroc(np.asarray(cdfm_result.logits), truth),
        "cdfm_threshold": float(cdfm_result.threshold),
        "cdfm_runtime_sec": float(cdfm_result.runtime_sec),
        "lingam_f1": float(lingam_metrics["f1"]),
        "lingam_precision": float(lingam_metrics["precision"]),
        "lingam_recall": float(lingam_metrics["recall"]),
        "lingam_shd": int(lingam_metrics["shd"]),
        "lingam_runtime_sec": float(lingam_runtime),
        "lingam_failure": lingam_failure,
        **features,
        "feature_runtime_sec": float(feature_runtime),
        "router_defer_probability": probability,
        "router_prediction": prediction,
        "selected_solver": solver,
        "router_failure": router_failure,
        "router_target": int(
            float(lingam_metrics["f1"]) >= float(cdfm_metrics["f1"]) + DEFER_MARGIN
        ),
        "hybrid_f1": hybrid_f1,
        "oracle_f1": oracle_f1,
    }
    raw = {
        "truth_adjacency": truth,
        "cdfm_adjacency": cdfm_adjacency,
        "cdfm_probabilities": np.asarray(cdfm_result.probabilities, dtype=float),
        "cdfm_logits": np.asarray(cdfm_result.logits, dtype=float),
        "lingam_coefficients_target_source": coefficients,
        "lingam_adjacency_source_target": lingam_adjacency,
        "lingam_causal_order": lingam_order,
        "router_features": np.array([features[name] for name in FEATURE_COLUMNS], dtype=float),
    }
    return row, raw


def _write_environment(cdfm_model: CDFM, device: str) -> None:
    packages = {}
    for name in ["cdfm-base", "causal-learn", "numpy", "pandas", "scikit-learn", "torch"]:
        packages[name] = importlib.metadata.version(name)
    _atomic_json(
        ENVIRONMENT_PATH,
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device,
            "cuda_available": bool(torch.cuda.is_available()),
            "packages": packages,
            "cdfm_info": cdfm_model.info,
            "cdfm_threshold_policy": "official auto calibration per environment",
            "direct_lingam": {"measure": "pwling", "matrix_input": "B[target,source]"},
            "router_refit": False,
            "a2_official_notebook_vote_threshold": 0.25,
            "a2_official_notebook_average_logits_threshold": 0.7,
        },
    )


def run_a1(cdfm_model: CDFM, router: object) -> bool:
    truth = _truth()
    x = _load_x(DATA_DIR / "task_a1" / "uniform_reference.csv", 10000)
    row, raw = _evaluate_one(
        dataset_id="uniform_reference", x=x, truth=truth, cdfm_model=cdfm_model, router=router
    )
    np.savez_compressed(RAW_DIR / "a1_uniform_reference.npz", **raw)
    sanity_delta = abs(float(row["cdfm_f1"]) - A1_EXPECTED_CDFM_F1)
    passed = sanity_delta <= A1_SANITY_TOLERANCE
    payload = {
        **row,
        "expected_cdfm_f1": A1_EXPECTED_CDFM_F1,
        "sanity_tolerance": A1_SANITY_TOLERANCE,
        "sanity_absolute_delta": sanity_delta,
        "sanity_passed": passed,
        "paper_reference": {
            "cdfm_f1": 0.727,
            "direct_lingam_f1": 0.495,
        },
        "official_notebook_reference": {"cdfm_f1": 0.7272727272727273, "cdfm_auroc": 0.9353},
    }
    _atomic_json(A1_RESULT_PATH, payload)
    print(
        f"A1: CDFM F1={row['cdfm_f1']:.4f}, DirectLiNGAM F1={row['lingam_f1']:.4f}, "
        f"Router={row['selected_solver']}, sanity={'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def _a1_is_valid() -> bool:
    if not A1_RESULT_PATH.exists():
        return False
    return bool(json.loads(A1_RESULT_PATH.read_text(encoding="utf-8"))["sanity_passed"])


def run_a2(cdfm_model: CDFM, router: object) -> None:
    if not _a1_is_valid():
        raise RuntimeError("A1 sanity gate has not passed; A2 execution is blocked")
    truth = _truth()
    existing_rows = (
        pd.read_csv(A2_RESULTS_PATH).to_dict(orient="records") if A2_RESULTS_PATH.exists() else []
    )
    rows_by_id = {str(row["dataset_id"]): row for row in existing_rows}
    rows: list[dict[str, object]] = []
    raw_by_environment: list[dict[str, np.ndarray]] = []
    for ordinal, environment in enumerate(A2_ENVIRONMENTS, start=1):
        raw_path = RAW_DIR / f"a2_{environment}.npz"
        if environment in rows_by_id and raw_path.exists():
            rows.append(rows_by_id[environment])
            with np.load(raw_path) as saved:
                raw_by_environment.append({name: saved[name] for name in saved.files})
            print(f"A2 [{ordinal}/{len(A2_ENVIRONMENTS)}] skip {environment}", flush=True)
            continue
        x = _load_x(DATA_DIR / "task_a2" / f"{environment}.csv", 1000)
        row, raw = _evaluate_one(
            dataset_id=environment, x=x, truth=truth, cdfm_model=cdfm_model, router=router
        )
        np.savez_compressed(raw_path, **raw)
        rows.append(row)
        raw_by_environment.append(raw)
        pd.DataFrame(rows).to_csv(A2_RESULTS_PATH.with_suffix(".csv.tmp"), index=False)
        os.replace(A2_RESULTS_PATH.with_suffix(".csv.tmp"), A2_RESULTS_PATH)
        print(
            f"A2 [{ordinal}/{len(A2_ENVIRONMENTS)}] {environment}: "
            f"CDFM={row['cdfm_f1']:.4f}, LiNGAM={row['lingam_f1']:.4f}, "
            f"Router={row['selected_solver']}",
            flush=True,
        )

    cdfm_vote = np.mean([raw["cdfm_adjacency"] for raw in raw_by_environment], axis=0)
    lingam_vote = np.mean([raw["lingam_adjacency_source_target"] for raw in raw_by_environment], axis=0)
    avg_logits = np.mean([raw["cdfm_logits"] for raw in raw_by_environment], axis=0)
    avg_probabilities = np.mean([raw["cdfm_probabilities"] for raw in raw_by_environment], axis=0)
    ensembles = {
        "cdfm_notebook_adjacency_vote_ge_5_of_20": (cdfm_vote >= 0.25).astype(np.int8),
        "cdfm_notebook_avg_logits_gt_0p7": (avg_logits > 0.7).astype(np.int8),
        "cdfm_paper_text_probability_average_gt_0p5": (avg_probabilities > 0.5).astype(np.int8),
        "direct_lingam_matched_adjacency_vote_ge_5_of_20": (lingam_vote >= 0.25).astype(np.int8),
    }
    for adjacency in ensembles.values():
        np.fill_diagonal(adjacency, 0)
    ensemble_payload: dict[str, object] = {
        "protocol_note": (
            "Current notebook vote, current notebook avg-logit, and paper-text probability-average "
            "are reported separately; none was selected after seeing performance."
        ),
        "paper_reference": {"cdfm_f1": 0.727, "direct_lingam_f1": 0.349},
        "cdfm_average_logits_auroc": _edge_auroc(avg_logits, truth),
        "metrics": {
            name: directed_graph_metrics(adjacency, truth) for name, adjacency in ensembles.items()
        },
        # DirectLiNGAM 论文没有公开 A2 聚合代码。完整保存 1..20 票门槛扫
        # 描述协议敏感性；正式口径仍是上面的 matched 5/20，不按论文数字择优。
        "direct_lingam_vote_threshold_sweep": {
            str(vote_count): directed_graph_metrics(
                (lingam_vote >= vote_count / len(A2_ENVIRONMENTS)).astype(np.int8), truth
            )
            for vote_count in range(1, len(A2_ENVIRONMENTS) + 1)
        },
    }
    _atomic_json(A2_ENSEMBLE_PATH, ensemble_payload)
    np.savez_compressed(
        RAW_DIR / "a2_ensembles.npz",
        truth_adjacency=truth,
        cdfm_vote=cdfm_vote,
        lingam_vote=lingam_vote,
        cdfm_average_logits=avg_logits,
        cdfm_average_probabilities=avg_probabilities,
        **ensembles,
    )
    print("A2 per-environment and ensemble benchmark complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a1", "a2", "all"], default="a1")
    args = parser.parse_args()
    verify_frozen_router()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    router = joblib.load(FROZEN_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cdfm_model = CDFM.from_pretrained("DMIRLAB/CDFM", device=device)
    _write_environment(cdfm_model, device)
    if args.stage in {"a1", "all"}:
        passed = run_a1(cdfm_model, router)
        if not passed:
            raise RuntimeError("A1 sanity gate failed; diagnose before running A2")
    if args.stage in {"a2", "all"}:
        run_a2(cdfm_model, router)


if __name__ == "__main__":
    main()
