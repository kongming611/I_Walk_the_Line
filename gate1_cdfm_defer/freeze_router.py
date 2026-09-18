"""在接触 Gate-1 标签前，用 Gate-0 全量数据训练并冻结最终 Router。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
FROZEN_DIR = ROOT / "frozen"
MODEL_PATH = FROZEN_DIR / "router.joblib"
MANIFEST_PATH = FROZEN_DIR / "freeze_manifest.json"
DEFER_MARGIN = 0.03
ROUTER_THRESHOLD = 0.5
ROUTER_RANDOM_SEED = 20260916

sys.path.insert(0, str(GATE0_ROOT))
from meta_features import (  # noqa: E402
    CV_SEED,
    CV_SPLITS,
    FEATURE_COLUMNS,
    GAIN_EPSILON,
    TOP_PAIR_COUNT,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_training_frame() -> pd.DataFrame:
    features_path = GATE0_ROOT / "results" / "meta_features.csv"
    results_path = GATE0_ROOT / "results" / "per_dataset_results.csv"
    features = pd.read_csv(features_path)
    results = pd.read_csv(results_path)
    frame = results[["dataset_id", "base_seed", "lambda", "cdfm_f1", "lingam_f1"]].merge(
        features,
        on=["dataset_id", "base_seed", "lambda"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != 150 or frame["dataset_id"].nunique() != 150:
        raise RuntimeError("Gate-0 frozen training set must contain exactly 150 unique datasets")
    if frame["base_seed"].nunique() != 30:
        raise RuntimeError("Gate-0 frozen training set must contain exactly 30 base seeds")
    if not np.isfinite(frame[FEATURE_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("Gate-0 frozen features contain non-finite values")
    return frame


def _expected_input_hashes() -> dict[str, str]:
    paths = {
        "gate0_per_dataset_results": GATE0_ROOT / "results" / "per_dataset_results.csv",
        "gate0_meta_features": GATE0_ROOT / "results" / "meta_features.csv",
        "gate0_summary": GATE0_ROOT / "results" / "summary.json",
        "gate0_meta_feature_source": GATE0_ROOT / "meta_features.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def freeze() -> None:
    if MODEL_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "Frozen Router already exists. Refusing to overwrite; use --verify instead."
        )
    frame = _load_training_frame()
    x = frame[FEATURE_COLUMNS].to_numpy(float)
    y = (frame["lingam_f1"] >= frame["cdfm_f1"] + DEFER_MARGIN).astype(int).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError("Gate-0 Router target must contain both classes")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=ROUTER_RANDOM_SEED,
                ),
            ),
        ]
    )
    model.fit(x, y)

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    temp_model = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(model, temp_model)
    os.replace(temp_model, MODEL_PATH)

    training_rows = frame[["dataset_id", "base_seed", "lambda"]].sort_values("dataset_id")
    training_row_digest = hashlib.sha256(
        training_rows.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "gate1_frozen_router_v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate1_label_or_performance_seen_before_freeze": False,
        "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_sha256": sha256(MODEL_PATH),
        "training": {
            "source": "Gate-0 frozen 150-dataset results and meta-features",
            "row_count": int(len(frame)),
            "base_seed_count": int(frame["base_seed"].nunique()),
            "row_identity_sha256": training_row_digest,
            "input_sha256": _expected_input_hashes(),
            "target_definition": f"lingam_f1 >= cdfm_f1 + {DEFER_MARGIN}",
            "class_counts": {
                "use_cdfm": int(np.sum(y == 0)),
                "defer_to_lingam": int(np.sum(y == 1)),
            },
        },
        "feature_pipeline": {
            "feature_columns": list(FEATURE_COLUMNS),
            "top_pair_count": TOP_PAIR_COUNT,
            "pair_cv_splits": CV_SPLITS,
            "pair_cv_seed": CV_SEED,
            "gain_epsilon": GAIN_EPSILON,
            "nonlinear_regressor": "SplineTransformer(n_knots=7, degree=3, include_bias=False) + Ridge(alpha=1e-3)",
            "source_file": "../gate0_cdfm_defer/meta_features.py",
            "forbidden_features": [
                "mechanism_label",
                "lambda",
                "dataset_id",
                "base_seed",
                "ground_truth",
                "algorithm_output",
                "algorithm_metric",
            ],
        },
        "router": {
            "type": "StandardScaler + LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": ROUTER_RANDOM_SEED,
            "decision_threshold": ROUTER_THRESHOLD,
            "gate1_refit_allowed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "joblib": importlib.metadata.version("joblib"),
        },
    }
    temp_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_manifest, MANIFEST_PATH)
    print(
        "Frozen Gate-1 Router: "
        f"rows={len(frame)}, classes={manifest['training']['class_counts']}, "
        f"model_sha256={manifest['model_sha256']}"
    )


def verify() -> None:
    if not MODEL_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Frozen Router artifacts are incomplete")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if sha256(MODEL_PATH) != manifest["model_sha256"]:
        raise AssertionError("Frozen Router model hash mismatch")
    if _expected_input_hashes() != manifest["training"]["input_sha256"]:
        raise AssertionError("Gate-0 inputs or feature source changed after Router freeze")
    model = joblib.load(MODEL_PATH)
    frame = _load_training_frame()
    probabilities = model.predict_proba(frame[FEATURE_COLUMNS].to_numpy(float))[:, 1]
    if probabilities.shape != (150,) or not np.isfinite(probabilities).all():
        raise AssertionError("Frozen Router prediction smoke failed")
    print(
        "Frozen Router verified: hashes match, Gate-0 inputs unchanged, "
        f"probability_range=[{probabilities.min():.6f}, {probabilities.max():.6f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        freeze()


if __name__ == "__main__":
    main()
