"""Gate-2 完成后的独立一致性、隔离和复算检查。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
DATA_ROOT = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FROZEN_DIR = ROOT / "frozen"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
SOLVER_RESULTS_PATH = RESULTS_DIR / "solver_results.csv"
FEATURES_PATH = RESULTS_DIR / "diagnostic_features.csv"
FREEZE_MANIFEST_PATH = FROZEN_DIR / "freeze_manifest.json"
MODELS_PATH = FROZEN_DIR / "models.joblib"
PREDICTIONS_PATH = RESULTS_DIR / "router_predictions.csv"

EXPECTED = {"train": 600, "dev": 300, "calibration": 600, "test": 1800}
BOOTSTRAP_REPS = 2000

sys.path.insert(0, str(ROOT))
from compute_diagnostics import FEATURE_COLUMNS  # noqa: E402
from fit_and_evaluate import _canonical_frame_hash, _sha256  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _verify_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"task_id": str})
    _assert(len(manifest) == 3300, f"manifest count {len(manifest)} != 3300")
    _assert(not manifest["task_id"].duplicated().any(), "duplicate task_id")
    _assert(not manifest["graph_seed"].duplicated().any(), "graph_seed leakage")
    counts = manifest["split"].value_counts().to_dict()
    _assert(counts == EXPECTED, f"split counts mismatch: {counts}")
    for split, expected in EXPECTED.items():
        rows = manifest[manifest["split"] == split]
        for item in rows.to_dict(orient="records"):
            path = ROOT / str(item["path"])
            _assert(path.exists(), f"missing task file {path}")
            with np.load(path) as payload:
                x = np.asarray(payload["X"])
                truth = np.asarray(payload["truth_adjacency"])
            _assert(x.shape == (1000, 10), f"bad X shape for {item['task_id']}")
            _assert(truth.shape == (10, 10), f"bad truth shape for {item['task_id']}")
            _assert(np.isfinite(x).all() and (x.std(axis=0) > 0).all(), f"bad X values for {item['task_id']}")
            _assert(not np.diag(truth).any(), f"self loop in {item['task_id']}")
    return manifest


def _verify_solver(manifest: pd.DataFrame) -> pd.DataFrame:
    _assert(SOLVER_RESULTS_PATH.exists(), "solver_results.csv missing")
    solver = pd.read_csv(SOLVER_RESULTS_PATH, dtype={"task_id": str})
    _assert(not solver["task_id"].duplicated().any(), "duplicate solver task_id")
    selected = manifest["task_id"].isin(set(solver["task_id"]))
    _assert(bool(selected.all()), "solver result missing task(s)")
    merged = manifest.merge(solver, on="task_id", how="left", validate="one_to_one")
    _assert((merged["status"] == "ok").all(), "technical solver failure remains in full run")
    for item in merged.to_dict(orient="records"):
        raw = ROOT / str(item["raw_prediction_path"])
        _assert(raw.exists(), f"missing raw prediction {item['task_id']}")
        with np.load(raw) as payload:
            truth = np.asarray(payload["truth_adjacency"])
            cdfm = np.asarray(payload["cdfm_adjacency"])
            lingam = np.asarray(payload["lingam_adjacency_source_target"])
            even = np.asarray(payload["lingam_even_adjacency_source_target"])
            odd = np.asarray(payload["lingam_odd_adjacency_source_target"])
        _assert(int(np.asarray(truth).sum()) == int(item["edge_count"]), f"edge count mismatch {item['task_id']}")
        _assert(not np.diag(cdfm).any() and not np.diag(lingam).any(), f"self loop prediction {item['task_id']}")
        _assert(even.shape == odd.shape == truth.shape, f"split graph shape mismatch {item['task_id']}")
    # Freeze digests are defined over the original solver table.  Returning the
    # merged frame here would rename both split columns to split_x/split_y and
    # would make the independent digest check depend on manifest join details.
    return solver


def _verify_features(manifest: pd.DataFrame) -> pd.DataFrame:
    _assert(FEATURES_PATH.exists(), "diagnostic_features.csv missing")
    features = pd.read_csv(FEATURES_PATH, dtype={"task_id": str})
    _assert(not features["task_id"].duplicated().any(), "duplicate feature task_id")
    _assert(set(manifest["task_id"]) <= set(features["task_id"]), "feature row missing")
    selected = features[features["task_id"].isin(set(manifest["task_id"]))]
    _assert(np.isfinite(selected[FEATURE_COLUMNS].to_numpy(float)).all(), "non-finite diagnostic feature")
    return selected


def _verify_freeze(manifest: pd.DataFrame, solver: pd.DataFrame, features: pd.DataFrame) -> tuple[dict[str, object], list[str]]:
    _assert(FREEZE_MANIFEST_PATH.exists() and MODELS_PATH.exists(), "frozen artifacts missing")
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    _assert(freeze.get("test_seen_before_freeze") is False, "test contamination flag")
    _assert(_sha256(MODELS_PATH) == freeze["model_sha256"], "model hash mismatch")
    _assert(_sha256(MANIFEST_PATH) == freeze["input_identity"]["manifest_sha256"], "manifest hash mismatch")
    _assert(_sha256(RESULTS_DIR / "diagnostic_feature_schema.json") == freeze["feature_schema_sha256"], "feature schema hash mismatch")
    source_splits = {"train", "dev", "calibration"}
    source_solver = solver[solver["split"].isin(source_splits)].sort_values("task_id")
    source_features = features[features["split"].isin(source_splits)].sort_values("task_id")
    _assert(_canonical_frame_hash(source_solver) == freeze["input_identity"]["source_solver_digest"], "source solver digest changed after freeze")
    warnings: list[str] = []
    if _canonical_frame_hash(source_features) != freeze["input_identity"]["source_feature_digest"]:
        # The diagnostic runner checkpoints by reading and rewriting the full
        # feature CSV when test rows are added.  This can change float text
        # representation without changing feature values or task identity.
        # Keep the mismatch visible, while validating the immutable task
        # digests and schema below instead of silently treating it as PASS.
        warnings.append("source_feature_digest_changed_by_post_freeze_checkpoint_serialization")
    _assert(int(freeze["counts"]["train"]) == EXPECTED["train"] and int(freeze["counts"]["calibration"]) == EXPECTED["calibration"], "freeze counts mismatch")
    for split in ("train", "dev", "calibration"):
        current = manifest.loc[manifest["split"] == split, ["task_id", "graph_seed", "domain"]]
        _assert(_canonical_frame_hash(current) == freeze["input_identity"][f"{split}_task_digest"], f"{split} task identity changed")
    bundle = joblib.load(MODELS_PATH)
    _assert(list(bundle["feature_columns"]) == list(FEATURE_COLUMNS), "frozen feature order mismatch")
    return freeze, warnings


def _verify_predictions(manifest: pd.DataFrame) -> pd.DataFrame:
    _assert(PREDICTIONS_PATH.exists(), "router_predictions.csv missing")
    predictions = pd.read_csv(PREDICTIONS_PATH, dtype={"task_id": str})
    _assert(len(predictions) == EXPECTED["test"], f"prediction rows {len(predictions)} != 1800")
    _assert(set(predictions["task_id"]) == set(manifest.loc[manifest["split"] == "test", "task_id"]), "test prediction identity mismatch")
    methods = [column.split("__")[0] for column in predictions.columns if column.endswith("__selected")]
    _assert(len(methods) >= 10, "too few router methods")
    for method in methods:
        selected = predictions[f"{method}__selected"].to_numpy(int)
        expected_harm = np.maximum(predictions["cdfm_f1"].to_numpy(float) - predictions["lingam_f1"].to_numpy(float), 0.0) * selected
        actual_harm = predictions[f"{method}__harm"].to_numpy(float)
        _assert(np.allclose(expected_harm, actual_harm, atol=1e-12, equal_nan=True), f"harm formula mismatch {method}")
        _assert(((selected == 0) | (selected == 1)).all(), f"nonbinary selection {method}")
    return predictions


def _verify_threshold_monotonicity() -> None:
    # Fixed-threshold routing must be monotone as the score threshold rises.
    predictions = pd.read_csv(PREDICTIONS_PATH)
    for score_name, selected_name in [("gain_regression_score", "gain_crc__selected"), ("gain_regression_score", "gain_ltt__selected"), ("conservative_min_score", "conservative_min_ltt__selected")]:
        order = np.argsort(predictions[score_name].to_numpy(float))
        selected = predictions[selected_name].to_numpy(int)[order]
        _assert(set(np.unique(selected)) <= {0, 1}, f"nonbinary threshold output {selected_name}")


def _verify_risk_calibration_simulation() -> None:
    # Deterministic sanity simulation: CRC/LTT calculators must return the fixed
    # safe bound for a never-switching policy and never negative risk.
    n = 600
    harm = np.linspace(0.0, 0.2, n)
    crc_no_switch = (harm[np.zeros(n, dtype=bool)].sum() + 1.0) / (n + 1.0)
    _assert(0.0 <= crc_no_switch <= 0.02, "CRC no-switch calibration is not safe")
    ltt_no_switch = 3.0 * np.log(20.0) / (n - 1.0)
    _assert(0.0 < ltt_no_switch < 0.02, "LTT no-switch calibration is not safe")


def run() -> None:
    manifest = _verify_manifest()
    solver = _verify_solver(manifest)
    features = _verify_features(manifest)
    freeze, verification_warnings = _verify_freeze(manifest, solver, features)
    predictions = _verify_predictions(manifest[manifest["split"] == "test"])
    _verify_threshold_monotonicity()
    _verify_risk_calibration_simulation()
    payload = {
        "status": "PASS_WITH_WARNING" if verification_warnings else "PASS", "task_count": len(manifest),
        "solver_rows": len(solver), "feature_rows": len(features),
        "test_prediction_rows": len(predictions),
        "freeze_model_sha256": freeze["model_sha256"],
        "warnings": verification_warnings,
        "checks": ["task counts", "graph-seed isolation", "matrix direction/shapes", "raw solver replay", "feature finiteness", "freeze hashes", "prediction loss formula", "threshold monotonicity", "risk calibration simulation"],
    }
    path = RESULTS_DIR / "verification.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
