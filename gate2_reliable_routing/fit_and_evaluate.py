"""冻结 Gate-2 路由器并在唯一一次测试上评估。

``--fit`` 只读取 train/dev/calibration 三个源域分割，校准风险阈值后写入
``frozen/``；``--evaluate`` 要求冻结文件存在，并只读取 test。测试标签只在
评估和报告阶段出现，绝不参与模型、特征或阈值选择。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FROZEN_DIR = ROOT / "frozen"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SOLVER_RESULTS_PATH = RESULTS_DIR / "solver_results.csv"
FEATURES_PATH = RESULTS_DIR / "diagnostic_features.csv"
MODELS_PATH = FROZEN_DIR / "models.joblib"
FREEZE_MANIFEST_PATH = FROZEN_DIR / "freeze_manifest.json"
PREDICTIONS_PATH = RESULTS_DIR / "router_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.json"
DOMAIN_SUMMARY_PATH = RESULTS_DIR / "domain_summary.csv"
BOOTSTRAP_PATH = RESULTS_DIR / "bootstrap_summary.csv"
FIGURE_PATH = RESULTS_DIR / "risk_gain_frontier.png"
REPORT_PATH = ROOT / "REPORT.md"

GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
GATE1_ROOT = REPO_ROOT / "experiments" / "gate1_cdfm_defer"
sys.path.insert(0, str(ROOT))
from compute_diagnostics import FEATURE_COLUMNS, GATE0_FEATURE_COLUMNS  # noqa: E402

RANDOM_SEED = 2026091703
DEFER_MARGIN = 0.03
RISK_BUDGET_H = 0.02
SEVERE_LIMIT = 0.05
RISK_ALPHA = 0.05
BOOTSTRAP_REPS = 2000
EXPECTED_COUNTS = {"train": 600, "dev": 300, "calibration": 600, "test": 1800}

METHODS = [
    "always_cdfm", "always_lingam", "old_gate0_router", "logistic_router",
    "gain_regression", "l2d_weighted", "gain_crc", "gain_ltt",
    "conformal_uncertain", "support_gate", "conservative_min_ltt", "support_gain_ltt",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    normalized = frame.sort_values("task_id").to_csv(index=False, lineterminator="\n").encode("utf-8")
    digest.update(normalized)
    return digest.hexdigest()


def _load_split(split: str, *, allow_failures: bool = False) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"task_id": str})
    manifest = manifest[manifest["split"] == split].copy()
    solver = pd.read_csv(SOLVER_RESULTS_PATH, dtype={"task_id": str})
    features = pd.read_csv(FEATURES_PATH, dtype={"task_id": str})
    if manifest["task_id"].duplicated().any() or solver["task_id"].duplicated().any() or features["task_id"].duplicated().any():
        raise RuntimeError(f"duplicate task_id in {split} input")
    frame = manifest.merge(solver, on="task_id", how="left", suffixes=("", "_solver"), validate="one_to_one")
    frame = frame.merge(features, on=["task_id", "split", "domain", "graph_seed"], how="left", suffixes=("", "_feature"), validate="one_to_one")
    if len(frame) != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"{split} requires exactly {EXPECTED_COUNTS[split]} tasks, found {len(frame)}")
    if not allow_failures and (frame["status"].fillna("missing") != "ok").any():
        bad = frame.loc[frame["status"].fillna("missing") != "ok", "task_id"].head(5).tolist()
        raise RuntimeError(f"{split} contains solver technical failures: {bad}")
    return frame


def _matrix(frame: pd.DataFrame, columns: list[str], medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    values = frame[columns].to_numpy(dtype=float)
    if medians is None:
        medians = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
        medians = np.nan_to_num(medians, nan=0.0)
    # The explicit column loop avoids fragile advanced-index broadcasting and is deterministic.
    for index, median in enumerate(medians):
        values[~np.isfinite(values[:, index]), index] = float(median)
    return values, np.asarray(medians, dtype=float)


def _fit_logistic(x: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
        model.fit(x, y)
        return model
    return Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED)),
    ]).fit(x, y)


def _fit_weighted_classifier(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
        model.fit(x, y, sample_weight=weights)
        return model
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return model.fit(x, y, sample_weight=weights)


def _fit_regressor(x: np.ndarray, y: np.ndarray):
    return RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=RANDOM_SEED, n_jobs=1,
    ).fit(x, y)


def _score_probability(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 in classes:
            return np.asarray(probabilities[:, classes.index(1)], dtype=float)
        return np.zeros(x.shape[0], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _harm(frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(frame["cdfm_f1"].to_numpy(float) - frame["lingam_f1"].to_numpy(float), 0.0)


def _bound_crc(harm: np.ndarray, selected: np.ndarray) -> float:
    n = len(harm)
    return (float(np.sum(harm[selected])) + 1.0) / (n + 1.0)


def _bound_ltt(harm: np.ndarray, selected: np.ndarray) -> float:
    n = len(harm)
    if n <= 1:
        return 1.0
    values = harm * selected.astype(float)
    mean = float(np.mean(values))
    variance = float(np.var(values, ddof=1))
    log_term = math.log(1.0 / RISK_ALPHA)
    return mean + math.sqrt(max(0.0, 2.0 * variance * log_term / n)) + 3.0 * log_term / (n - 1.0)


def _calibrate_threshold(scores: np.ndarray, frame: pd.DataFrame, bound_kind: str) -> dict[str, float | int | bool]:
    harm = _harm(frame)
    candidates = [float("inf"), *sorted({float(value) for value in scores if np.isfinite(value)})]
    feasible: list[tuple[int, float, float]] = []
    for threshold in candidates:
        selected = np.isfinite(scores) & (scores >= threshold)
        bound = _bound_crc(harm, selected) if bound_kind == "crc" else _bound_ltt(harm, selected)
        if bound <= RISK_BUDGET_H:
            feasible.append((int(selected.sum()), threshold, bound))
    if not feasible:
        return {"threshold": float("inf"), "selected_count": 0, "bound": 1.0, "kind": bound_kind}
    # Most permissive feasible threshold; threshold itself is fixed before test labels.
    count, threshold, bound = max(feasible, key=lambda item: (item[0], -item[1] if np.isfinite(item[1]) else -float("inf")))
    return {"threshold": float(threshold), "selected_count": int(count), "bound": float(bound), "kind": bound_kind}


def _threshold_for_json(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _threshold_from_json(value: object) -> float:
    return float("inf") if value is None else float(value)


def _save_joblib(bundle: dict[str, object]) -> str:
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODELS_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, MODELS_PATH)
    return _sha256(MODELS_PATH)


def _fit() -> None:
    if MODELS_PATH.exists() or FREEZE_MANIFEST_PATH.exists():
        raise FileExistsError("Gate-2 frozen models already exist; refusing overwrite")
    train = _load_split("train")
    dev = _load_split("dev")
    calibration = _load_split("calibration")
    if set(train["task_id"]) & set(dev["task_id"]) or set(train["task_id"]) & set(calibration["task_id"]) or set(dev["task_id"]) & set(calibration["task_id"]):
        raise RuntimeError("train/dev/calibration task overlap")
    x_train, medians = _matrix(train, FEATURE_COLUMNS)
    x_dev, _ = _matrix(dev, FEATURE_COLUMNS, medians)
    x_cal, _ = _matrix(calibration, FEATURE_COLUMNS, medians)
    delta_train = train["lingam_f1"].to_numpy(float) - train["cdfm_f1"].to_numpy(float)
    delta_cal = calibration["lingam_f1"].to_numpy(float) - calibration["cdfm_f1"].to_numpy(float)
    target_train = (delta_train >= DEFER_MARGIN).astype(int)

    old_gate0_model = joblib.load(GATE1_ROOT / "frozen" / "router.joblib")
    logistic = _fit_logistic(x_train, target_train)
    regression = _fit_regressor(x_train, delta_train)
    weighted = _fit_weighted_classifier(x_train, target_train, np.abs(delta_train) + DEFER_MARGIN)
    domain_regressors = {}
    for domain in sorted(train["domain"].unique()):
        subset = train["domain"] == domain
        domain_regressors[domain] = _fit_regressor(x_train[subset.to_numpy()], delta_train[subset.to_numpy()])

    train_scale = np.std(x_train, axis=0)
    train_scale[train_scale <= 1e-12] = 1.0
    train_center = np.mean(x_train, axis=0)
    train_distance = np.max(np.abs((x_train - train_center) / train_scale), axis=1)
    support_distance_threshold = float(np.quantile(train_distance, 0.95))

    regression_cal_scores = np.asarray(regression.predict(x_cal), dtype=float)
    conservative_cal_scores = np.column_stack([
        model.predict(x_cal) for model in domain_regressors.values()
    ]).min(axis=1)
    crc = _calibrate_threshold(regression_cal_scores, calibration, "crc")
    ltt = _calibrate_threshold(regression_cal_scores, calibration, "ltt")
    conservative_ltt = _calibrate_threshold(conservative_cal_scores, calibration, "ltt")
    support_cal_distance = np.max(np.abs((x_cal - train_center) / train_scale), axis=1)
    support_scores = np.where(support_cal_distance <= support_distance_threshold, regression_cal_scores, -np.inf)
    support_ltt = _calibrate_threshold(support_scores, calibration, "ltt")

    weighted_cal_prob = _score_probability(weighted, x_cal)
    true_prob = np.where((delta_cal >= DEFER_MARGIN), weighted_cal_prob, 1.0 - weighted_cal_prob)
    conformal_nonconformity = 1.0 - true_prob
    conformal_q = float(np.quantile(conformal_nonconformity, 1.0 - RISK_ALPHA, method="higher"))

    bundle = {
        "schema_version": "gate2_models_v1",
        "feature_columns": list(FEATURE_COLUMNS),
        "gate0_feature_columns": list(GATE0_FEATURE_COLUMNS),
        "feature_medians": medians,
        "old_gate0_model": old_gate0_model,
        "logistic_router": logistic,
        "gain_regression": regression,
        "l2d_weighted": weighted,
        "domain_regressors": domain_regressors,
        "train_center": train_center,
        "train_scale": train_scale,
        "support_distance_threshold": support_distance_threshold,
        "conformal_nonconformity_quantile": conformal_q,
        "thresholds": {
            "gain_crc": {**crc, "threshold": _threshold_for_json(float(crc["threshold"]))},
            "gain_ltt": {**ltt, "threshold": _threshold_for_json(float(ltt["threshold"]))},
            "conservative_min_ltt": {**conservative_ltt, "threshold": _threshold_for_json(float(conservative_ltt["threshold"]))},
            "support_gain_ltt": {**support_ltt, "threshold": _threshold_for_json(float(support_ltt["threshold"]))},
        },
        "fit_config": {
            "random_seed": RANDOM_SEED, "defer_margin": DEFER_MARGIN,
            "risk_budget_h": RISK_BUDGET_H, "risk_alpha": RISK_ALPHA,
            "severe_limit": SEVERE_LIMIT, "support_quantile": 0.95,
        },
    }
    model_hash = _save_joblib(bundle)
    solver_all = pd.read_csv(SOLVER_RESULTS_PATH, dtype={"task_id": str})
    feature_all = pd.read_csv(FEATURES_PATH, dtype={"task_id": str})
    source_splits = ["train", "dev", "calibration"]
    solver_source = solver_all[solver_all["split"].isin(source_splits)].sort_values("task_id")
    feature_source = feature_all[feature_all["split"].isin(source_splits)].sort_values("task_id")
    def _safe_calibration(item: dict[str, object]) -> dict[str, object]:
        output = dict(item)
        output["threshold"] = _threshold_for_json(float(item["threshold"]))
        return output
    manifest = {
        "schema_version": "gate2_freeze_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_seen_before_freeze": False,
        "model_path": MODELS_PATH.relative_to(ROOT).as_posix(),
        "model_sha256": model_hash,
        "feature_schema_sha256": _sha256(RESULTS_DIR / "diagnostic_feature_schema.json"),
        "input_identity": {
            "manifest_sha256": _sha256(MANIFEST_PATH),
            "source_solver_digest": _canonical_frame_hash(solver_source),
            "source_feature_digest": _canonical_frame_hash(feature_source),
            "train_task_digest": _canonical_frame_hash(train[["task_id", "graph_seed", "domain"]]),
            "dev_task_digest": _canonical_frame_hash(dev[["task_id", "graph_seed", "domain"]]),
            "calibration_task_digest": _canonical_frame_hash(calibration[["task_id", "graph_seed", "domain"]]),
        },
        "counts": {"train": len(train), "dev": len(dev), "calibration": len(calibration)},
        "class_counts": {"train_defer_margin": int(target_train.sum()), "train_cdfm": int((target_train == 0).sum())},
        "risk_calibration": {
            "gain_crc": _safe_calibration(crc), "gain_ltt": _safe_calibration(ltt),
            "conservative_min_ltt": _safe_calibration(conservative_ltt), "support_gain_ltt": _safe_calibration(support_ltt),
            "conformal_q": conformal_q,
        },
        "forbidden_at_fit": ["test solver metrics", "test feature rows", "test labels"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    temporary = FREEZE_MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, FREEZE_MANIFEST_PATH)
    dev_report = _quick_validation(dev, bundle, "dev")
    print(json.dumps({"frozen": True, "model_sha256": model_hash, "dev": dev_report}, ensure_ascii=False, indent=2))


def _predict_scores(frame: pd.DataFrame, bundle: dict[str, object]) -> pd.DataFrame:
    x, _ = _matrix(frame, list(bundle["feature_columns"]), np.asarray(bundle["feature_medians"], dtype=float))
    old_x = x[:, [list(bundle["feature_columns"]).index(name) for name in bundle["gate0_feature_columns"]]]
    old_probability = _score_probability(bundle["old_gate0_model"], old_x)
    logistic_probability = _score_probability(bundle["logistic_router"], x)
    weighted_probability = _score_probability(bundle["l2d_weighted"], x)
    gain = np.asarray(bundle["gain_regression"].predict(x), dtype=float)
    domain_scores = np.column_stack([model.predict(x) for model in bundle["domain_regressors"].values()])
    conservative = domain_scores.min(axis=1)
    center = np.asarray(bundle["train_center"], dtype=float)
    scale = np.asarray(bundle["train_scale"], dtype=float)
    support_distance = np.max(np.abs((x - center) / scale), axis=1)
    support_ok = support_distance <= float(bundle["support_distance_threshold"])
    thresholds = bundle["thresholds"]
    crc_t = _threshold_from_json(thresholds["gain_crc"]["threshold"])
    ltt_t = _threshold_from_json(thresholds["gain_ltt"]["threshold"])
    conservative_t = _threshold_from_json(thresholds["conservative_min_ltt"]["threshold"])
    support_t = _threshold_from_json(thresholds["support_gain_ltt"]["threshold"])
    conformal_q = float(bundle["conformal_nonconformity_quantile"])
    conformal_min_p = 1.0 - conformal_q
    diagnostic_ok = (frame["diagnostic_failure"].fillna(1).to_numpy(int) == 0) & (frame["status"].fillna("error").to_numpy() == "ok")
    out = pd.DataFrame({
        "task_id": frame["task_id"].astype(str).to_numpy(), "split": frame["split"].to_numpy(),
        "domain": frame["domain"].to_numpy(), "status": frame["status"].fillna("error").to_numpy(),
        "diagnostic_ok": diagnostic_ok.astype(int), "cdfm_f1": frame["cdfm_f1"].to_numpy(float),
        "lingam_f1": frame["lingam_f1"].to_numpy(float), "old_gate0_probability": old_probability,
        "logistic_probability": logistic_probability, "l2d_weighted_probability": weighted_probability,
        "gain_regression_score": gain, "conservative_min_score": conservative,
        "support_distance": support_distance, "support_ok": support_ok.astype(int),
    })
    selections = {
        "always_cdfm": np.zeros(len(frame), dtype=int),
        "always_lingam": np.ones(len(frame), dtype=int),
        "old_gate0_router": old_probability >= 0.5,
        "logistic_router": logistic_probability >= 0.5,
        "gain_regression": gain >= DEFER_MARGIN,
        "l2d_weighted": weighted_probability >= 0.5,
        "gain_crc": gain >= crc_t,
        "gain_ltt": gain >= ltt_t,
        "conformal_uncertain": (weighted_probability >= conformal_min_p) & (weighted_probability > 0.5),
        "support_gate": (gain >= DEFER_MARGIN) & support_ok,
        "conservative_min_ltt": conservative >= conservative_t,
        "support_gain_ltt": (gain >= support_t) & support_ok,
    }
    for method, selected in selections.items():
        selected = np.asarray(selected, dtype=int)
        if method not in {"always_cdfm", "always_lingam"}:
            selected = selected * diagnostic_ok.astype(int)
        out[f"{method}__selected"] = selected
        out[f"{method}__hybrid_f1"] = np.where(selected == 1, out["lingam_f1"], out["cdfm_f1"])
        out[f"{method}__harm"] = np.maximum(out["cdfm_f1"] - out["lingam_f1"], 0.0) * selected
        out[f"{method}__severe_harm"] = (out[f"{method}__harm"] > 0.10).astype(int)
        out[f"{method}__benefit"] = np.maximum(out["lingam_f1"] - out["cdfm_f1"], 0.0) * selected
        out[f"{method}__oracle_gap"] = np.maximum(out["cdfm_f1"], out["lingam_f1"]) - out[f"{method}__hybrid_f1"]
    return out


def _quick_validation(frame: pd.DataFrame, bundle: dict[str, object], split: str) -> dict[str, object]:
    prediction = _predict_scores(frame, bundle)
    return {
        "split": split,
        "n": int(len(prediction)),
        "methods": {method: {"switch_rate": float(prediction[f"{method}__selected"].mean()), "net_gain": float((prediction[f"{method}__hybrid_f1"] - prediction["cdfm_f1"]).mean())} for method in METHODS},
    }


def _bootstrap(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(values) == 0:
        return np.array([np.nan] * BOOTSTRAP_REPS)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    return values[indices].mean(axis=1)


def _evaluate() -> None:
    if not MODELS_PATH.exists() or not FREEZE_MANIFEST_PATH.exists():
        raise FileNotFoundError("frozen models are required before test evaluation")
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if freeze.get("test_seen_before_freeze"):
        raise RuntimeError("freeze manifest claims test data was seen before freezing")
    if _sha256(MODELS_PATH) != freeze["model_sha256"]:
        raise RuntimeError("frozen model hash mismatch")
    test = _load_split("test", allow_failures=True)
    bundle = joblib.load(MODELS_PATH)
    predictions = _predict_scores(test, bundle)
    prediction_columns = ["task_id", "split", "domain", "status", "diagnostic_ok", "cdfm_f1", "lingam_f1", "old_gate0_probability", "logistic_probability", "l2d_weighted_probability", "gain_regression_score", "conservative_min_score", "support_distance", "support_ok"]
    for method in METHODS:
        prediction_columns.extend([f"{method}__selected", f"{method}__hybrid_f1", f"{method}__harm", f"{method}__severe_harm", f"{method}__benefit", f"{method}__oracle_gap"])
    temporary = PREDICTIONS_PATH.with_suffix(".csv.tmp")
    predictions[prediction_columns].to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, PREDICTIONS_PATH)

    successful = predictions[ predictions["status"] == "ok" ].copy()
    n_total = len(predictions)
    n_success = len(successful)
    rng = np.random.default_rng(RANDOM_SEED)
    summary: dict[str, object] = {
        "schema_version": "gate2_evaluation_v1", "n_total_test_tasks": n_total,
        "n_successful_tasks": n_success, "technical_failure_count": n_total - n_success,
        "technical_failure_rate": (n_total - n_success) / max(1, n_total),
        "risk_thresholds": {"mean_harm_H": RISK_BUDGET_H, "severe_harm_rate_T": SEVERE_LIMIT},
        "methods": {}, "decision": {},
    }
    domain_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for method in METHODS:
        hybrid = successful[f"{method}__hybrid_f1"].to_numpy(float)
        harm = successful[f"{method}__harm"].to_numpy(float)
        severe = successful[f"{method}__severe_harm"].to_numpy(float)
        net = hybrid - successful["cdfm_f1"].to_numpy(float)
        oracle = np.maximum(successful["cdfm_f1"].to_numpy(float), successful["lingam_f1"].to_numpy(float))
        selected = successful[f"{method}__selected"].to_numpy(float)
        bootstrap_h = _bootstrap(harm, rng)
        bootstrap_t = _bootstrap(severe, rng)
        bootstrap_net = _bootstrap(net, rng)
        summary["methods"][method] = {
            "mean_harm_H": float(np.mean(harm)), "severe_harm_rate_T": float(np.mean(severe)),
            "net_gain_vs_cdfm": float(np.mean(net)), "mean_hybrid_f1": float(np.mean(hybrid)),
            "mean_oracle_gap": float(np.mean(oracle - hybrid)), "switch_rate": float(np.mean(selected)),
            "oracle_gain": float(np.mean(oracle - successful["cdfm_f1"].to_numpy(float))),
            "bootstrap_upper_harm_pointwise": float(np.quantile(bootstrap_h, 0.95)),
            "bootstrap_upper_severe_pointwise": float(np.quantile(bootstrap_t, 0.95)),
            "bootstrap_lower_net_pointwise": float(np.quantile(bootstrap_net, 0.05)),
        }
        for domain, subset in successful.groupby("domain"):
            local = subset
            local_h = local[f"{method}__harm"].to_numpy(float)
            local_t = local[f"{method}__severe_harm"].to_numpy(float)
            local_net = local[f"{method}__hybrid_f1"].to_numpy(float) - local["cdfm_f1"].to_numpy(float)
            local_oracle = np.maximum(local["cdfm_f1"].to_numpy(float), local["lingam_f1"].to_numpy(float))
            domain_rows.append({
                "method": method, "domain": domain, "n": len(local),
                "mean_harm_H": float(np.mean(local_h)), "severe_harm_rate_T": float(np.mean(local_t)),
                "net_gain_vs_cdfm": float(np.mean(local_net)), "mean_oracle_gap": float(np.mean(local_oracle - local[f"{method}__hybrid_f1"].to_numpy(float))),
                "switch_rate": float(local[f"{method}__selected"].mean()),
            })
            bh = _bootstrap(local_h, rng); bt = _bootstrap(local_t, rng); bn = _bootstrap(local_net, rng)
            bootstrap_rows.extend([
                {"method": method, "domain": domain, "metric": "harm_upper_simultaneous", "value": float(np.quantile(bh, 1.0 - RISK_ALPHA / (len(METHODS) * max(1, successful["domain"].nunique()))))},
                {"method": method, "domain": domain, "metric": "severe_upper_simultaneous", "value": float(np.quantile(bt, 1.0 - RISK_ALPHA / (len(METHODS) * max(1, successful["domain"].nunique()))))},
                {"method": method, "domain": domain, "metric": "net_lower_simultaneous", "value": float(np.quantile(bn, RISK_ALPHA / (len(METHODS) * max(1, successful["domain"].nunique()))))},
            ])
    pd.DataFrame(domain_rows).to_csv(DOMAIN_SUMMARY_PATH, index=False, lineterminator="\n")
    pd.DataFrame(bootstrap_rows).to_csv(BOOTSTRAP_PATH, index=False, lineterminator="\n")
    domain_frame = pd.DataFrame(domain_rows)
    boot_frame = pd.DataFrame(bootstrap_rows)
    for method in METHODS:
        local_boot = boot_frame[boot_frame["method"] == method]
        uppers_h = local_boot[local_boot["metric"] == "harm_upper_simultaneous"]["value"]
        uppers_t = local_boot[local_boot["metric"] == "severe_upper_simultaneous"]["value"]
        summary["methods"][method]["simultaneous_max_harm_upper"] = float(uppers_h.max()) if not uppers_h.empty else float("nan")
        summary["methods"][method]["simultaneous_max_severe_upper"] = float(uppers_t.max()) if not uppers_t.empty else float("nan")

    candidates = []
    for method, values in summary["methods"].items():
        method_domains = domain_frame[domain_frame["method"] == method]
        if method_domains.empty:
            continue
        safe = bool((method_domains["mean_harm_H"] <= RISK_BUDGET_H).all() and (method_domains["severe_harm_rate_T"] <= SEVERE_LIMIT).all())
        substantive = bool(values["net_gain_vs_cdfm"] >= 0.015 and (method_domains["net_gain_vs_cdfm"] > 0.0).sum() >= 3)
        candidates.append({"method": method, "safe_point": safe, "substantive": substantive, "switch_rate": values["switch_rate"], "net_gain": values["net_gain_vs_cdfm"]})
    qualifying = [item for item in candidates if item["safe_point"] and item["substantive"] and item["method"] not in {"always_cdfm", "always_lingam"}]
    if qualifying:
        best = max(qualifying, key=lambda item: item["net_gain"])
        decision = "GO" if best["method"] in {"conservative_min_ltt", "support_gain_ltt", "gain_crc", "gain_ltt"} else "BORDERLINE"
        rationale = f"safe substantive candidate={best['method']}"
    elif any(item["safe_point"] and item["switch_rate"] < 0.05 for item in candidates):
        decision = "STOP/PIVOT"
        rationale = "only safe routers are effectively always-CDFM or nearly never-switching"
    else:
        decision = "BORDERLINE"
        rationale = "complementarity or risk evidence is incomplete/insufficient"
    summary["decision"] = {"label": decision, "rationale": rationale, "qualifying_candidates": qualifying, "all_candidates": candidates}
    temporary = SUMMARY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, SUMMARY_PATH)
    _plot(domain_frame)
    _write_report(summary, domain_frame, boot_frame)
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


def _plot(domain_frame: pd.DataFrame) -> None:
    if domain_frame.empty:
        return
    figure, axis = plt.subplots(figsize=(10, 6))
    for method in METHODS:
        local = domain_frame[domain_frame["method"] == method]
        if local.empty:
            continue
        axis.scatter(local["mean_harm_H"], local["net_gain_vs_cdfm"], label=method, s=35)
    axis.axvline(RISK_BUDGET_H, color="red", linestyle="--", linewidth=1)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Mean harmful-switch loss H")
    axis.set_ylabel("Net gain vs CDFM")
    axis.set_title("Gate-2 risk–gain frontier by migration domain")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def _write_report(summary: dict[str, object], domain_frame: pd.DataFrame, boot_frame: pd.DataFrame) -> None:
    lines = [
        "# Gate-2：机制迁移下的安全整图路由实验",
        "",
        "文档职责：记录 Gate-2 的独立任务生成、求解、冻结、测试、风险校准与生死判据。",
        "适用范围：只覆盖 `experiments/gate2_reliable_routing/`；不修改或重解释 Gate-0/Gate-1 与生产 Agent。",
        "",
        f"最终决策：**{summary['decision']['label']}**。{summary['decision']['rationale']}",
        "",
        "## 固定协议",
        "",
        "任务单位是独立 DAG 图；源域 train/dev/calibration 为 600/300/600，六个未见迁移域各 300 个测试图。所有方法共享 X-only 诊断特征与 CDFM/DirectLiNGAM 计算结果，测试只运行一次。",
        "风险门槛固定为平均错误切换损失 H ≤ 0.02、严重损失 T（损失 > 0.10）≤ 5%；失败任务保留在技术失败账本和总分母中。",
        "",
        "## 测试汇总",
        "",
        f"测试任务总数 `{summary['n_total_test_tasks']}`，成功 `{summary['n_successful_tasks']}`，技术失败 `{summary['technical_failure_count']}`（失败率 `{summary['technical_failure_rate']:.4f}`）。",
        "",
        "| 方法 | H | T | 净收益 vs CDFM | 切换率 | 同时校正 H 上界 | 同时校正 T 上界 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        lines.append(f"| {method} | {values['mean_harm_H']:.4f} | {values['severe_harm_rate_T']:.4f} | {values['net_gain_vs_cdfm']:.4f} | {values['switch_rate']:.4f} | {values['simultaneous_max_harm_upper']:.4f} | {values['simultaneous_max_severe_upper']:.4f} |")
    lines.extend([
        "",
        "逐任务预测、路由分数和损失见 `results/router_predictions.csv`；域级统计见 `results/domain_summary.csv`；2,000 次任务级 bootstrap 的同时校正结果见 `results/bootstrap_summary.csv`。",
        "",
        "## 复现与边界",
        "",
        "冻结账本 `frozen/freeze_manifest.json` 保存输入哈希、特征 schema、阈值和模型哈希；`verify_gate2.py` 负责独立复算。技术失败没有从分母删除，诊断失败强制回退 CDFM。",
        "",
        "本实验是严格合成筛选。Gate-1 的真实数据结果只作问题诊断，本轮不把目标真实数据加入训练、校准或测试。",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.fit == args.evaluate:
        raise SystemExit("choose exactly one of --fit or --evaluate")
    _fit() if args.fit else _evaluate()


if __name__ == "__main__":
    main()
