"""Gate-0 核心约定的轻量单元与真实算法 smoke 检查。"""

from __future__ import annotations

import numpy as np
from causallearn.search.FCMBased import lingam

from generate_data import LAMBDAS, generate_base_spec, generate_dataset
from metrics import directed_graph_metrics, lingam_target_source_to_source_target


def test_manual_direction_conversion() -> None:
    coefficients = np.zeros((3, 3), dtype=float)
    coefficients[2, 0] = 1.25  # B[target=2, source=0]
    converted = lingam_target_source_to_source_target(coefficients)
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[0, 2] = 1
    assert np.array_equal(converted, expected)


def test_directed_metric_penalizes_reversal() -> None:
    truth = np.array([[0, 1], [0, 0]], dtype=np.int8)
    reversed_edge = np.array([[0, 0], [1, 0]], dtype=np.int8)
    metrics = directed_graph_metrics(reversed_edge, truth)
    assert metrics["f1"] == 0.0
    assert metrics["shd"] == 1


def test_generator_invariants_and_shared_roots() -> None:
    spec = generate_base_spec(0)
    datasets = [generate_dataset(spec, value) for value in LAMBDAS]
    roots = np.flatnonzero(spec.adjacency.sum(axis=0) == 0)
    assert roots.size > 0
    for x in datasets:
        assert x.shape == (1000, 10)
        assert np.isfinite(x).all()
        assert np.all(x.std(axis=0) > 0)
    for root in roots:
        for x in datasets[1:]:
            assert np.allclose(datasets[0][:, root], x[:, root])


def test_real_lingam_direction_smoke() -> None:
    rng = np.random.default_rng(91)
    source = rng.laplace(size=4000)
    target = 1.2 * source + rng.laplace(scale=0.2, size=4000)
    x = np.column_stack([source, target])
    model = lingam.DirectLiNGAM(measure="pwling")
    model.fit(x)
    converted = lingam_target_source_to_source_target(model.adjacency_matrix_)
    assert converted[0, 1] == 1
    assert converted[1, 0] == 0


def main() -> None:
    tests = [
        test_manual_direction_conversion,
        test_directed_metric_penalizes_reversal,
        test_generator_invariants_and_shared_roots,
        test_real_lingam_direction_smoke,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
