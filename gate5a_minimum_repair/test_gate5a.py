"""Fast contract tests for the Gate-5A implementation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment as gate5a  # noqa: E402


def test_protocol_freezes_split_projection_and_gate() -> None:
    payload = gate5a.protocol_payload()
    assert payload["scientific_question"].startswith("Can held-out causal contradictions")
    assert payload["data"]["domain_count"] == 6
    assert payload["data"]["tasks_per_domain"] == 20
    assert payload["data"]["D"] == 10
    assert payload["data"]["N_discovery"] == 500
    assert payload["data"]["N_validation"] == 500
    assert payload["candidate_graph"]["candidate_is_dag"] is True
    assert payload["witnesses"]["alpha"] == 0.05
    assert payload["repair"]["search"].endswith("k=0,1,2,3")


def test_projection_breaks_reciprocal_edges_and_cycles_deterministically() -> None:
    raw = np.zeros((gate5a.N_VARIABLES, gate5a.N_VARIABLES), dtype=np.int8)
    probabilities = np.zeros_like(raw, dtype=float)
    raw[0, 1] = raw[1, 0] = 1
    probabilities[0, 1] = 0.8
    probabilities[1, 0] = 0.2
    raw[1, 2] = raw[2, 1] = 1
    probabilities[1, 2] = 0.7
    probabilities[2, 1] = 0.3
    raw[2, 0] = raw[0, 2] = 1
    probabilities[2, 0] = 0.6
    probabilities[0, 2] = 0.4
    projected = gate5a.project_to_dag(raw, probabilities)
    assert gate5a._is_dag(projected)
    assert projected[0, 1] == 1
    assert projected[1, 2] == 1
    assert projected[2, 0] == 0


def test_local_markov_and_d_separation() -> None:
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = 1
    chain[1, 2] = 1
    assert gate5a.d_separated(chain, 0, 2, [1])
    assert not gate5a.d_separated(chain, 0, 2, [])
    statements = gate5a.local_markov_statements(chain)
    assert {tuple(item["conditioning_set"]) for item in statements if item["x"] == 0 and item["y"] == 2} == {(1,)}


def test_holm_step_down() -> None:
    rejected, adjusted = gate5a.holm_adjust([0.001, 0.02, 0.2], 0.05)
    assert rejected.tolist() == [True, True, False]
    assert np.allclose(adjusted, [0.003, 0.04, 0.2])


def test_minimum_repair_radius_is_one_for_a_chain_witness() -> None:
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = 1
    chain[1, 2] = 1
    witness = [{"x": 0, "y": 2, "conditioning_set": [1]}]
    result = gate5a.minimum_repair_radius(chain, witness, max_k=2)
    assert result["rr_lower_bound"] == 1
    assert result["rr_exact"] is True


def test_truth_is_not_a_witness_or_certificate_builder_argument() -> None:
    assert "truth" not in inspect.signature(gate5a.build_task_witnesses).parameters
    assert "truth" not in inspect.signature(gate5a.minimum_repair_radius).parameters
    assert "truth_adjacency" not in inspect.getsource(gate5a.build_task_witnesses)
    assert "truth_adjacency" not in inspect.getsource(gate5a.build_certificates)
