"""Fast, truth-free contract tests for the Gate-4 implementation."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import run_experiment as gate4


def test_protocol_has_frozen_certificate_contract() -> None:
    payload = gate4.protocol_payload()
    assert payload["scientific_question"].startswith("Ground-truth-free")
    assert payload["certificate"]["primary_alpha"] == 0.10
    assert payload["certificate"]["bad_risk_threshold"] == 0.30
    assert payload["data"]["final"]["task_count"] == 180
    assert payload["truth_isolation"]["final_truth_used_for_feature_construction"] is False


def test_new_generator_is_finite_distinct_and_dag_compatible() -> None:
    spec = gate4._graph(700000)
    arrays = [gate4.generate_x_gate4(spec, mechanism, "centered_lognormal", 700000)
              for mechanism in ("arctan", "cubic")]
    assert all(np.isfinite(item).all() for item in arrays)
    assert all(np.std(item, axis=0).min() > 1e-8 for item in arrays)
    assert not np.array_equal(arrays[0], arrays[1])


def test_feature_builder_does_not_accept_truth() -> None:
    assert "truth_adjacency" not in inspect.signature(gate4.build_task_features).parameters
    assert not any(token in name.lower() for name in gate4.FEATURE_COLUMNS
                   for token in ("truth", "f1", "shd", "oracle", "label"))


def test_latent_projection_handles_chain_fork_and_collider() -> None:
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = chain[1, 2] = 1
    directed, bidirected = gate4.latent_projection(chain, 1)
    assert directed[0, 2] == 1 and bidirected.sum() == 0

    fork = np.zeros((3, 3), dtype=np.int8)
    fork[0, 1] = fork[0, 2] = 1
    directed, bidirected = gate4.latent_projection(fork, 0)
    assert bidirected[1, 2] == bidirected[2, 1] == 1
    assert directed.sum() == 0

    collider = np.zeros((3, 3), dtype=np.int8)
    collider[1, 0] = collider[2, 0] = 1
    directed, bidirected = gate4.latent_projection(collider, 0)
    assert bidirected.sum() == 0 and directed.sum() == 0


def test_conformal_quantile_is_finite_and_conservative() -> None:
    values = np.linspace(-1.0, 1.0, 10)
    q = gate4.finite_sample_quantile(values, 0.10)
    assert np.isfinite(q)
    assert q >= np.quantile(values, 0.90)

