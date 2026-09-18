"""Fast contract tests for the Gate-3C local generator and feature boundary."""

from __future__ import annotations

import inspect

import numpy as np

from run_experiment import (
    ALL_FEATURES,
    FINAL_SEED_START,
    PROPOSED_FEATURES,
    _graph,
    _is_dag,
    generate_x_gate3c,
    ground_truth_free_features,
)


def test_new_mechanisms_are_finite_nonconstant_and_distinct() -> None:
    spec = _graph(FINAL_SEED_START)
    assert _is_dag(spec.adjacency)
    arrays = {
        mechanism: generate_x_gate3c(spec, mechanism, "gaussian", 1.0, FINAL_SEED_START)
        for mechanism in ("tanh", "rff", "quadratic", "piecewise")
    }
    for array in arrays.values():
        assert np.isfinite(array).all()
        assert np.all(np.var(array, axis=0) > 1e-12)
    for new_mechanism in ("quadratic", "piecewise"):
        assert not np.array_equal(arrays[new_mechanism], arrays["tanh"])
        assert not np.array_equal(arrays[new_mechanism], arrays["rff"])


def test_feature_contract_has_no_truth_graph_or_label_feature() -> None:
    assert "truth_adjacency" not in inspect.signature(ground_truth_free_features).parameters
    assert "truth_adjacency" not in inspect.signature(generate_x_gate3c).parameters
    forbidden = ("truth", "f1", "shd", "mechanism", "noise", "seed", "oracle", "label")
    assert not any(any(token in name.lower() for token in forbidden) for name in PROPOSED_FEATURES)
    assert not any(any(token in name.lower() for token in forbidden) for name in ALL_FEATURES)
