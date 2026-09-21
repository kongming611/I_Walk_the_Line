"""Fast contract tests for Gate-5B without running CDFM or RCIT."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gate5b_under_test", ROOT / "run_experiment.py")
if _spec is None or _spec.loader is None:
    raise ImportError("cannot load Gate-5B runner")
gate5b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate5b)


def test_protocol_freezes_two_separate_certificates() -> None:
    payload = gate5b.protocol_payload()
    assert payload["data"]["domain_count"] == 6
    assert payload["data"]["tasks_per_domain"] == 20
    assert payload["data"]["D"] == 10
    assert payload["data"]["N_discovery"] == 500
    assert payload["data"]["N_validation"] == 500
    assert payload["certificates"]["rr_ci"]["alpha"] == 0.05
    assert payload["certificates"]["rr_anm"]["regressor"]["n_estimators"] == 300
    assert payload["repair"]["certificates_separate"] is True


def test_two_parent_set_witnesses_require_two_edits() -> None:
    candidate = np.zeros((4, 4), dtype=np.int8)
    candidate[0, 1] = 1
    candidate[2, 3] = 1
    witnesses = [
        {"kind": "anm", "y": 1, "z": 0, "candidate_parents": [0]},
        {"kind": "anm", "y": 3, "z": 2, "candidate_parents": [2]},
    ]
    result = gate5b.minimum_repair_radius(candidate, witnesses, max_k=2)
    assert result["rr_lower_bound"] == 2
    assert result["rr_exact"] is True


def test_one_reversal_can_resolve_two_witnesses() -> None:
    candidate = np.zeros((3, 3), dtype=np.int8)
    candidate[0, 1] = 1
    witnesses = [
        {"kind": "anm", "y": 0, "z": 2, "candidate_parents": []},
        {"kind": "anm", "y": 1, "z": 0, "candidate_parents": [0]},
    ]
    result = gate5b.minimum_repair_radius(candidate, witnesses, max_k=1)
    assert result["rr_lower_bound"] == 1
    assert result["rr_exact"] is True


def test_ci_witness_resolution_uses_d_separation() -> None:
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = 1
    chain[1, 2] = 1
    witness = {"kind": "ci", "x": 0, "y": 2, "conditioning_set": [1]}
    assert gate5b.witness_persists(chain, witness) is True
    repaired = chain.copy()
    repaired[0, 2] = 1
    assert gate5b.witness_persists(repaired, witness) is False


def test_truth_is_not_a_witness_builder_argument() -> None:
    assert "truth" not in inspect.signature(gate5b._anm_witnesses_for_task).parameters
    assert "truth" not in inspect.signature(gate5b.minimum_repair_radius).parameters
    assert "truth_adjacency" not in inspect.getsource(gate5b._anm_witnesses_for_task)
    assert "truth_adjacency" not in inspect.getsource(gate5b.build_anm)
