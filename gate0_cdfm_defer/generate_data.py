"""生成 Gate-0 冻结的线性到 RFF 非线性连续谱数据。"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BASE_SPEC_DIR = DATA_DIR / "base_specs"
DATASET_DIR = DATA_DIR / "datasets"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

MASTER_SEED = 20260916
N_SAMPLES = 1000
N_VARIABLES = 10
EXPECTED_TOTAL_DEGREE = 2.0
EDGE_PROBABILITY = EXPECTED_TOTAL_DEGREE / (N_VARIABLES - 1)
RFF_FEATURES = 30
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASE_SEEDS = tuple(range(30))
NOISE_SCALE = 1.0 / np.sqrt(2.0)
SCALE_EPS = 1e-12


@dataclass(frozen=True)
class BaseSpec:
    adjacency: np.ndarray
    weights: np.ndarray
    topological_order: np.ndarray
    noise: np.ndarray
    rff_omega: np.ndarray
    rff_phase: np.ndarray
    rff_coeff: np.ndarray


def _rng_for(base_seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, base_seed]))


def generate_base_spec(base_seed: int) -> BaseSpec:
    """生成一个可审计的 DAG、参数与跨 lambda 共享的噪声。"""
    rng = _rng_for(base_seed)
    order = rng.permutation(N_VARIABLES)
    adjacency = np.zeros((N_VARIABLES, N_VARIABLES), dtype=np.int8)

    for earlier in range(N_VARIABLES):
        for later in range(earlier + 1, N_VARIABLES):
            if rng.random() < EDGE_PROBABILITY:
                adjacency[order[earlier], order[later]] = 1

    # 极小概率的空图不适合比较；该保底规则只依赖生成随机数，不依赖算法结果。
    if not adjacency.any():
        adjacency[order[0], order[1]] = 1

    magnitudes = rng.uniform(0.5, 1.5, size=(N_VARIABLES, N_VARIABLES))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_VARIABLES, N_VARIABLES))
    weights = adjacency * magnitudes * signs

    shape = (N_VARIABLES, N_VARIABLES, RFF_FEATURES)
    rff_omega = rng.normal(loc=0.0, scale=1.0, size=shape)
    rff_phase = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    rff_coeff = rng.normal(loc=0.0, scale=1.0, size=shape)
    noise = rng.laplace(
        loc=0.0,
        scale=NOISE_SCALE,
        size=(N_SAMPLES, N_VARIABLES),
    )
    return BaseSpec(
        adjacency=adjacency,
        weights=weights,
        topological_order=order.astype(np.int64),
        noise=noise,
        rff_omega=rff_omega,
        rff_phase=rff_phase,
        rff_coeff=rff_coeff,
    )


def _standardize_component(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = centered.std(ddof=0)
    if not np.isfinite(scale) or scale <= SCALE_EPS:
        raise ValueError("SEM component has zero or non-finite scale")
    return centered / scale


def _rff_edge(x: np.ndarray, omega: np.ndarray, phase: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    features = np.cos(x[:, None] * omega[None, :] + phase[None, :])
    return np.sqrt(2.0 / RFF_FEATURES) * (features @ coeff)


def generate_dataset(spec: BaseSpec, lambda_value: float) -> np.ndarray:
    """以相同外生噪声按拓扑序生成一个 lambda 数据集。"""
    if lambda_value not in LAMBDAS:
        raise ValueError(f"lambda must be one of {LAMBDAS}")
    x = np.zeros((N_SAMPLES, N_VARIABLES), dtype=np.float64)

    for child in spec.topological_order:
        parents = np.flatnonzero(spec.adjacency[:, child])
        if parents.size == 0:
            x[:, child] = spec.noise[:, child]
            continue

        linear = x[:, parents] @ spec.weights[parents, child]
        nonlinear = np.zeros(N_SAMPLES, dtype=np.float64)
        for parent in parents:
            nonlinear += spec.weights[parent, child] * _rff_edge(
                x[:, parent],
                spec.rff_omega[parent, child],
                spec.rff_phase[parent, child],
                spec.rff_coeff[parent, child],
            )

        linear_normalized = _standardize_component(linear)
        nonlinear_normalized = _standardize_component(nonlinear)
        signal = (
            (1.0 - lambda_value) * linear_normalized
            + lambda_value * nonlinear_normalized
        )
        x[:, child] = signal + spec.noise[:, child]

    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=0)
    if not np.isfinite(x).all():
        raise ValueError("generated data contains NaN or inf")
    if np.any(scales <= SCALE_EPS):
        raise ValueError("generated data contains a constant column")
    x = (x - means) / scales
    if not np.isfinite(x).all():
        raise ValueError("standardized data contains NaN or inf")
    return x.astype(np.float32)


def _assert_dag(adjacency: np.ndarray, order: np.ndarray) -> None:
    if adjacency.shape != (N_VARIABLES, N_VARIABLES):
        raise AssertionError("ground-truth adjacency has wrong shape")
    if np.any(np.diag(adjacency)) or np.any((adjacency != 0) & (adjacency != 1)):
        raise AssertionError("ground-truth adjacency is not a binary loop-free graph")
    positions = np.empty(N_VARIABLES, dtype=int)
    positions[order] = np.arange(N_VARIABLES)
    sources, targets = np.nonzero(adjacency)
    if np.any(positions[sources] >= positions[targets]):
        raise AssertionError("ground-truth graph violates its topological order")


def _atomic_write_manifest(rows: list[dict[str, object]]) -> None:
    temp_path = MANIFEST_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, MANIFEST_PATH)


def generate_all(overwrite: bool = False) -> None:
    BASE_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for base_seed in BASE_SEEDS:
        spec = generate_base_spec(base_seed)
        _assert_dag(spec.adjacency, spec.topological_order)
        spec_path = BASE_SPEC_DIR / f"seed_{base_seed:02d}.npz"
        if overwrite or not spec_path.exists():
            np.savez_compressed(
                spec_path,
                adjacency=spec.adjacency,
                weights=spec.weights,
                topological_order=spec.topological_order,
                noise=spec.noise,
                rff_omega=spec.rff_omega,
                rff_phase=spec.rff_phase,
                rff_coeff=spec.rff_coeff,
            )

        for lambda_value in LAMBDAS:
            x = generate_dataset(spec, lambda_value)
            lambda_tag = f"{lambda_value:.2f}".replace(".", "p")
            dataset_path = DATASET_DIR / f"seed_{base_seed:02d}_lambda_{lambda_tag}.npz"
            if overwrite or not dataset_path.exists():
                np.savez_compressed(
                    dataset_path,
                    X=x,
                    adjacency=spec.adjacency,
                    base_seed=np.int64(base_seed),
                    lambda_value=np.float64(lambda_value),
                )
            rows.append(
                {
                    "dataset_id": f"seed_{base_seed:02d}_lambda_{lambda_tag}",
                    "base_seed": base_seed,
                    "lambda": lambda_value,
                    "n_samples": N_SAMPLES,
                    "n_variables": N_VARIABLES,
                    "edge_count": int(spec.adjacency.sum()),
                    "dataset_path": dataset_path.relative_to(ROOT).as_posix(),
                    "base_spec_path": spec_path.relative_to(ROOT).as_posix(),
                }
            )

    _atomic_write_manifest(rows)
    print(f"Generated {len(rows)} datasets at {DATA_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()

