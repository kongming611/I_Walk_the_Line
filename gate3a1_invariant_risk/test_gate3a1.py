"""Fast Gate-3A.1 invariant checks; does not load a CDFM checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_experiment import (  # noqa: E402
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    TEST_SPECS,
    _gate_decision,
    threshold_features,
)


def main() -> None:
    protocol = json.loads((ROOT / "frozen_protocol.json").read_text(encoding="utf-8"))
    assert protocol["primary"]["features"] == PRIMARY_FEATURES
    assert protocol["secondary_threshold_aware"]["features"] == SECONDARY_FEATURES
    assert len(set(TEST_SPECS["softsign"])) == 25
    assert len(set(TEST_SPECS["sine"])) == 25
    assert set(TEST_SPECS["softsign"]).isdisjoint(TEST_SPECS["sine"])
    assert set(TEST_SPECS["softsign"]).isdisjoint(range(310000, 310015))
    assert set(TEST_SPECS["sine"]).isdisjoint(range(310000, 310015))

    probabilities = np.array([
        [0.0, 0.48, 0.50],
        [0.55, 0.0, 0.60],
        [0.40, 0.49, 0.0],
    ])
    features = threshold_features(probabilities, 0.50)
    off_diagonal = probabilities[~np.eye(3, dtype=bool)]
    margins = np.abs(off_diagonal - 0.50)
    assert np.isclose(features["cdfm_threshold_margin_mean"], margins.mean())
    assert np.isclose(features["cdfm_threshold_near_frac_002"], np.mean(margins <= 0.02))
    assert np.isclose(features["cdfm_threshold_near_frac_005"], np.mean(margins <= 0.05))

    import pandas as pd

    go = pd.DataFrame([
        {"family": "softsign", "model": "stable_ridge_v1", "spearman": 0.50, "risk_at_50": 0.15, "risk_at_100": 0.20, "relative_risk_reduction_at_50": 0.25},
        {"family": "sine", "model": "stable_ridge_v1", "spearman": 0.40, "risk_at_50": 0.17, "risk_at_100": 0.20, "relative_risk_reduction_at_50": 0.15},
    ])
    assert _gate_decision(go)[0] == "GO"
    stop = go.copy()
    stop["spearman"] = [0.10, 0.20]
    assert _gate_decision(stop)[0] == "STOP"
    print("Gate-3A.1 invariant tests passed.")


if __name__ == "__main__":
    main()
