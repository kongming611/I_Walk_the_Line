"""复核 Gate-0 全部数据、原始预测、指标与 group fold 不变量。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from generate_data import BASE_SPEC_DIR, DATASET_DIR, N_SAMPLES, N_VARIABLES, ROOT
from meta_features import FEATURES_PATH
from metrics import directed_graph_metrics, lingam_target_source_to_source_target
from run_benchmark import RAW_PREDICTIONS_DIR, RESULTS_PATH
from run_router import ROUTER_PATH, SUMMARY_PATH


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-10, atol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def validate() -> None:
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    benchmark = pd.read_csv(RESULTS_PATH)
    features = pd.read_csv(FEATURES_PATH)
    router = pd.read_csv(ROUTER_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    expected_ids = set(manifest["dataset_id"])
    for name, frame in [("benchmark", benchmark), ("features", features), ("router", router)]:
        if len(frame) != 150 or frame["dataset_id"].nunique() != 150:
            raise AssertionError(f"{name} does not contain 150 unique datasets")
        if set(frame["dataset_id"]) != expected_ids:
            raise AssertionError(f"{name} dataset ids differ from manifest")
    if len(list(DATASET_DIR.glob("*.npz"))) != 150:
        raise AssertionError("dataset artifact count is not 150")
    if len(list(BASE_SPEC_DIR.glob("*.npz"))) != 30:
        raise AssertionError("base spec artifact count is not 30")
    if len(list(RAW_PREDICTIONS_DIR.glob("*.npz"))) != 150:
        raise AssertionError("raw prediction artifact count is not 150")

    result_by_id = benchmark.set_index("dataset_id")
    x_by_seed: dict[int, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    for row in manifest.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        with np.load(ROOT / str(row["dataset_path"])) as dataset:
            x = np.asarray(dataset["X"], dtype=float)
            truth = np.asarray(dataset["adjacency"], dtype=np.int8)
        if x.shape != (N_SAMPLES, N_VARIABLES) or not np.isfinite(x).all():
            raise AssertionError(f"invalid X for {dataset_id}")
        if np.any(x.std(axis=0) <= 0) or not np.allclose(x.mean(axis=0), 0.0, atol=2e-6):
            raise AssertionError(f"non-standardized or constant X for {dataset_id}")
        if truth.shape != (N_VARIABLES, N_VARIABLES) or np.any(np.diag(truth)):
            raise AssertionError(f"invalid truth graph for {dataset_id}")
        x_by_seed.setdefault(int(row["base_seed"]), []).append((float(row["lambda"]), x, truth))

        result_row = result_by_id.loc[dataset_id]
        with np.load(ROOT / result_row.raw_prediction_path) as raw:
            if not np.array_equal(raw["truth_adjacency"], truth):
                raise AssertionError(f"raw truth mismatch for {dataset_id}")
            cdfm_adjacency = np.asarray(raw["cdfm_adjacency"], dtype=np.int8)
            lingam_adjacency = np.asarray(raw["lingam_adjacency_source_target"], dtype=np.int8)
            converted = lingam_target_source_to_source_target(raw["lingam_coefficients_target_source"])
            if not np.array_equal(converted, lingam_adjacency):
                raise AssertionError(f"LiNGAM conversion mismatch for {dataset_id}")
            if raw["cdfm_probabilities"].shape != truth.shape or not np.isfinite(raw["cdfm_probabilities"]).all():
                raise AssertionError(f"invalid CDFM probabilities for {dataset_id}")
        for prefix, adjacency in [("cdfm", cdfm_adjacency), ("lingam", lingam_adjacency)]:
            metrics = directed_graph_metrics(adjacency, truth)
            for metric in ["f1", "precision", "recall", "shd"]:
                _assert_close(
                    float(metrics[metric]),
                    float(result_row[f"{prefix}_{metric}"]),
                    label=f"{dataset_id} {prefix}_{metric}",
                )

    for base_seed, items in x_by_seed.items():
        items.sort(key=lambda item: item[0])
        reference_truth = items[0][2]
        roots = np.flatnonzero(reference_truth.sum(axis=0) == 0)
        for _, x, truth in items[1:]:
            if not np.array_equal(truth, reference_truth):
                raise AssertionError(f"truth changed across lambda for seed {base_seed}")
            if not np.array_equal(x[:, roots], items[0][1][:, roots]):
                raise AssertionError(f"shared root noise changed across lambda for seed {base_seed}")
        with np.load(BASE_SPEC_DIR / f"seed_{base_seed:02d}.npz") as spec:
            weights = np.asarray(spec["weights"], dtype=float)
            adjacency = np.asarray(spec["adjacency"], dtype=np.int8)
            edge_weights = np.abs(weights[adjacency.astype(bool)])
            if np.any(edge_weights < 0.5) or np.any(edge_weights > 1.5):
                raise AssertionError(f"edge weight range violated for seed {base_seed}")

    fold_counts = router.groupby("base_seed")["router_fold"].nunique()
    if not (fold_counts == 1).all():
        raise AssertionError("a base seed leaked across Router folds")
    if summary["decision"]["label"] not in {"GO", "BORDERLINE", "STOP"}:
        raise AssertionError("invalid Gate-0 decision label")

    cdfm = benchmark["cdfm_f1"].to_numpy(float)
    lingam = benchmark["lingam_f1"].to_numpy(float)
    hybrid = router["hybrid_selected_f1"].to_numpy(float)
    _assert_close(float(cdfm.mean()), summary["methods"]["always_cdfm"]["mean"], label="mean CDFM")
    _assert_close(float(lingam.mean()), summary["methods"]["always_lingam"]["mean"], label="mean LiNGAM")
    _assert_close(float(hybrid.mean()), summary["methods"]["hybrid_router"]["mean"], label="mean Hybrid")
    _assert_close(float(np.maximum(cdfm, lingam).mean()), summary["methods"]["oracle"]["mean"], label="mean Oracle")

    print(
        "Gate-0 validation passed: 150 datasets, 30 base specs, 150 raw predictions, "
        "metrics reproduced, no group leakage."
    )


if __name__ == "__main__":
    validate()
