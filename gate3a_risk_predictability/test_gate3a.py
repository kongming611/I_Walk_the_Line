"""Fast invariant tests for Gate-3A; does not load CDFM."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_experiment import (  # noqa: E402
    MECHANISMS,
    _graph,
    _mechanism_config,
    assert_no_label_leakage,
    diagnostic_features,
    generate_x,
)


def main() -> None:
    graph_seed = 399999
    spec = _graph(graph_seed)
    generated = {}
    for mechanism in MECHANISMS:
        generator_mechanism, lambda_value = _mechanism_config(mechanism)
        generated[mechanism] = generate_x(spec, generator_mechanism, "laplace", lambda_value, graph_seed)
        assert generated[mechanism].shape == (1000, 10)
        assert np.isfinite(generated[mechanism]).all()
    assert not np.allclose(generated["linear"], generated["rff"])
    assert_no_label_leakage(["mean_abs_skewness", "cdfm_edge_density", "ood_score"])
    try:
        assert_no_label_leakage(["cdfm_f1"])
    except AssertionError:
        pass
    else:
        raise AssertionError("leakage guard did not reject cdfm_f1")

    identity = np.eye(4, k=1, dtype=np.int8)
    probabilities = identity * 0.8 + (1 - identity) * 0.2
    x = np.random.default_rng(7).normal(size=(100, 4))
    features = diagnostic_features(x, identity, probabilities, identity.copy(), np.stack([identity, identity]))
    assert features["cdfm_lingam_adj_disagreement"] == 0.0
    assert features["cdfm_bootstrap_disagreement"] == 0.0
    assert features["cdfm_bootstrap_edge_jaccard"] == 1.0
    print("Gate-3A invariant tests passed.")


if __name__ == "__main__":
    main()

