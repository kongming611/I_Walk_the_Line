"""对 Gate-1 已保存 raw predictions 与汇总指标做独立只读复算。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from freeze_router import verify as verify_frozen_router
from prepare_real_data import ROOT


REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))
from metrics import directed_graph_metrics  # noqa: E402


def _close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-12, rtol=1e-10):
        raise AssertionError(f"{label}: {actual} != {expected}")


def verify_synthetic() -> None:
    frame = pd.read_csv(ROOT / "results" / "synthetic" / "per_dataset_results.csv")
    if len(frame) != 190 or frame["dataset_id"].nunique() != 190:
        raise AssertionError("synthetic count/uniqueness mismatch")
    for row in frame.itertuples(index=False):
        with np.load(ROOT / row.raw_prediction_path) as raw:
            cdfm = directed_graph_metrics(raw["cdfm_adjacency"], raw["truth_adjacency"])["f1"]
            lingam = directed_graph_metrics(
                raw["lingam_adjacency_source_target"], raw["truth_adjacency"]
            )["f1"]
        _close(float(cdfm), float(row.cdfm_f1), f"{row.dataset_id} CDFM")
        _close(float(lingam), float(row.lingam_f1), f"{row.dataset_id} LiNGAM")
        expected_hybrid = lingam if int(row.router_prediction) else cdfm
        _close(float(expected_hybrid), float(row.hybrid_f1), f"{row.dataset_id} hybrid")
        _close(float(max(cdfm, lingam)), float(row.oracle_f1), f"{row.dataset_id} oracle")


def verify_chamber() -> None:
    a1 = json.loads((ROOT / "results" / "causal_chamber" / "a1_result.json").read_text(encoding="utf-8"))
    with np.load(ROOT / "results" / "causal_chamber" / "raw_predictions" / "a1_uniform_reference.npz") as raw:
        _close(
            float(directed_graph_metrics(raw["cdfm_adjacency"], raw["truth_adjacency"])["f1"]),
            float(a1["cdfm_f1"]),
            "Causal Chamber A1 CDFM",
        )
    frame = pd.read_csv(ROOT / "results" / "causal_chamber" / "a2_per_environment.csv")
    if len(frame) != 20 or frame["dataset_id"].nunique() != 20:
        raise AssertionError("Causal Chamber A2 count/uniqueness mismatch")
    for row in frame.itertuples(index=False):
        raw_path = ROOT / "results" / "causal_chamber" / "raw_predictions" / f"a2_{row.dataset_id}.npz"
        with np.load(raw_path) as raw:
            cdfm = directed_graph_metrics(raw["cdfm_adjacency"], raw["truth_adjacency"])["f1"]
            lingam = directed_graph_metrics(
                raw["lingam_adjacency_source_target"], raw["truth_adjacency"]
            )["f1"]
        _close(float(cdfm), float(row.cdfm_f1), f"{row.dataset_id} CDFM")
        _close(float(lingam), float(row.lingam_f1), f"{row.dataset_id} LiNGAM")
        expected_hybrid = cdfm if str(row.selected_solver).startswith("CDFM") else lingam
        _close(float(expected_hybrid), float(row.hybrid_f1), f"{row.dataset_id} hybrid")


def verify_tuebingen() -> None:
    frame = pd.read_csv(ROOT / "results" / "tuebingen" / "per_pair_results.csv")
    if len(frame) != 95 or frame["pair_id"].nunique() != 95:
        raise AssertionError("Tübingen count/uniqueness mismatch")
    for row in frame.itertuples(index=False):
        with np.load(ROOT / row.raw_prediction_path) as raw:
            true_direction = str(raw["true_direction"].item())
            probabilities = raw["cdfm_probabilities"]
            if probabilities[0, 1] == probabilities[1, 0]:
                cdfm_direction = "failure"
            else:
                cdfm_direction = "0->1" if probabilities[0, 1] > probabilities[1, 0] else "1->0"
            order = list(raw["lingam_causal_order"].astype(int))
            lingam_direction = f"{order[0]}->{order[1]}" if sorted(order) == [0, 1] else "failure"
        if cdfm_direction != row.cdfm_direction or lingam_direction != row.lingam_direction:
            raise AssertionError(f"{row.pair_id} direction mismatch")
        if int(cdfm_direction == true_direction) != int(row.cdfm_correct):
            raise AssertionError(f"{row.pair_id} CDFM correctness mismatch")
        if int(lingam_direction == true_direction) != int(row.lingam_correct):
            raise AssertionError(f"{row.pair_id} LiNGAM correctness mismatch")


def main() -> None:
    verify_frozen_router()
    verify_synthetic()
    verify_chamber()
    verify_tuebingen()
    print("Gate-1 verification PASS: frozen hashes, counts, raw metrics, hybrid choices, directions")


if __name__ == "__main__":
    main()
