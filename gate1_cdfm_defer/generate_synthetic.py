"""生成 Gate-1 的 router-OOD 与 CDFM-scale-OOD 合成数据。"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "synthetic"
BASE_SPEC_DIR = DATA_DIR / "base_specs"
DATASET_DIR = DATA_DIR / "datasets"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

MASTER_SEED = 2026091601
RFF_FEATURES = 30
EXPECTED_TOTAL_DEGREE = 2.0
SCALE_EPS = 1e-12
LAPLACE_SCALE = 1.0 / np.sqrt(2.0)
STUDENT_T_DF = 3
STUDENT_T_SCALE = 1.0 / np.sqrt(STUDENT_T_DF / (STUDENT_T_DF - 2.0))

STRATA = {
    "router_ood_tanh": {
        "seeds": range(1000, 1015),
        "lambdas": (0.0, 0.25, 0.5, 0.75, 1.0),
        "n_samples": 1000,
        "n_variables": 10,
        "mechanism": "tanh_additive",
        "noise": "laplace",
    },
    "router_ood_student_t": {
        "seeds": range(2000, 2015),
        "lambdas": (0.0, 0.25, 0.5, 0.75, 1.0),
        "n_samples": 1000,
        "n_variables": 10,
        "mechanism": "rff",
        "noise": "student_t_df3",
    },
    "scale_ood_n40_d10": {
        "seeds": range(3000, 3010),
        "lambdas": (0.0, 1.0),
        "n_samples": 40,
        "n_variables": 10,
        "mechanism": "rff",
        "noise": "laplace",
    },
    "scale_ood_d120_n1000": {
        "seeds": range(4000, 4010),
        "lambdas": (0.0, 1.0),
        "n_samples": 1000,
        "n_variables": 120,
        "mechanism": "rff",
        "noise": "laplace",
    },
}


@dataclass(frozen=True)
class BaseSpec:
    adjacency: np.ndarray
    edge_sources: np.ndarray
    edge_targets: np.ndarray
    edge_weights: np.ndarray
    topological_order: np.ndarray
    noise: np.ndarray
    rff_omega: np.ndarray
    rff_phase: np.ndarray
    rff_coeff: np.ndarray
    tanh_slope: np.ndarray
    tanh_bias: np.ndarray


def _rng_for(stratum: str, base_seed: int) -> np.random.Generator:
    stratum_index = list(STRATA).index(stratum)
    return np.random.default_rng(
        np.random.SeedSequence([MASTER_SEED, stratum_index, base_seed])
    )


def generate_base_spec(stratum: str, base_seed: int) -> BaseSpec:
    config = STRATA[stratum]
    n_samples = int(config["n_samples"])
    n_variables = int(config["n_variables"])
    rng = _rng_for(stratum, base_seed)
    order = rng.permutation(n_variables)
    adjacency = np.zeros((n_variables, n_variables), dtype=np.int8)
    edge_probability = EXPECTED_TOTAL_DEGREE / (n_variables - 1)
    for earlier in range(n_variables):
        for later in range(earlier + 1, n_variables):
            if rng.random() < edge_probability:
                adjacency[order[earlier], order[later]] = 1
    if not adjacency.any():
        adjacency[order[0], order[1]] = 1

    edge_sources, edge_targets = np.nonzero(adjacency)
    edge_count = len(edge_sources)
    edge_weights = (
        rng.uniform(0.5, 1.5, size=edge_count)
        * rng.choice(np.array([-1.0, 1.0]), size=edge_count)
    )
    rff_omega = rng.normal(size=(edge_count, RFF_FEATURES))
    rff_phase = rng.uniform(0.0, 2.0 * np.pi, size=(edge_count, RFF_FEATURES))
    rff_coeff = rng.normal(size=(edge_count, RFF_FEATURES))
    tanh_slope = rng.uniform(0.5, 2.0, size=edge_count)
    tanh_bias = rng.uniform(-1.0, 1.0, size=edge_count)

    if config["noise"] == "laplace":
        noise = rng.laplace(0.0, LAPLACE_SCALE, size=(n_samples, n_variables))
    elif config["noise"] == "student_t_df3":
        noise = rng.standard_t(STUDENT_T_DF, size=(n_samples, n_variables)) * STUDENT_T_SCALE
    else:
        raise ValueError(f"unsupported noise: {config['noise']}")

    return BaseSpec(
        adjacency=adjacency,
        edge_sources=edge_sources.astype(np.int64),
        edge_targets=edge_targets.astype(np.int64),
        edge_weights=edge_weights,
        topological_order=order.astype(np.int64),
        noise=noise,
        rff_omega=rff_omega,
        rff_phase=rff_phase,
        rff_coeff=rff_coeff,
        tanh_slope=tanh_slope,
        tanh_bias=tanh_bias,
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = centered.std(ddof=0)
    if not np.isfinite(scale) or scale <= SCALE_EPS:
        raise ValueError("SEM component has zero or non-finite scale")
    return centered / scale


def _edge_rff(x: np.ndarray, omega: np.ndarray, phase: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return np.sqrt(2.0 / RFF_FEATURES) * (
        np.cos(x[:, None] * omega[None, :] + phase[None, :]) @ coeff
    )


def generate_dataset(stratum: str, spec: BaseSpec, lambda_value: float) -> np.ndarray:
    config = STRATA[stratum]
    n_samples = int(config["n_samples"])
    n_variables = int(config["n_variables"])
    x = np.zeros((n_samples, n_variables), dtype=np.float64)
    incoming = {
        child: np.flatnonzero(spec.edge_targets == child)
        for child in range(n_variables)
    }
    for child in spec.topological_order:
        edge_indices = incoming[int(child)]
        if edge_indices.size == 0:
            x[:, child] = spec.noise[:, child]
            continue
        linear = np.zeros(n_samples, dtype=np.float64)
        nonlinear = np.zeros(n_samples, dtype=np.float64)
        for edge_index in edge_indices:
            source = int(spec.edge_sources[edge_index])
            weight = float(spec.edge_weights[edge_index])
            linear += weight * x[:, source]
            if config["mechanism"] == "rff":
                nonlinear += weight * _edge_rff(
                    x[:, source],
                    spec.rff_omega[edge_index],
                    spec.rff_phase[edge_index],
                    spec.rff_coeff[edge_index],
                )
            elif config["mechanism"] == "tanh_additive":
                nonlinear += weight * np.tanh(
                    spec.tanh_slope[edge_index] * x[:, source]
                    + spec.tanh_bias[edge_index]
                )
            else:
                raise ValueError(f"unsupported mechanism: {config['mechanism']}")
        signal = (1.0 - lambda_value) * _normalize(linear) + lambda_value * _normalize(nonlinear)
        x[:, child] = signal + spec.noise[:, child]

    if not np.isfinite(x).all():
        raise ValueError("generated data contains NaN or inf")
    scales = x.std(axis=0, ddof=0)
    if np.any(scales <= SCALE_EPS):
        raise ValueError("generated data contains a constant column")
    x = (x - x.mean(axis=0)) / scales
    return x.astype(np.float32)


def _assert_dag(spec: BaseSpec) -> None:
    positions = np.empty(len(spec.topological_order), dtype=int)
    positions[spec.topological_order] = np.arange(len(spec.topological_order))
    if np.any(positions[spec.edge_sources] >= positions[spec.edge_targets]):
        raise AssertionError("ground-truth graph violates topological order")
    if np.any(np.diag(spec.adjacency)):
        raise AssertionError("ground-truth graph contains a self loop")
    if not np.all((np.abs(spec.edge_weights) >= 0.5) & (np.abs(spec.edge_weights) <= 1.5)):
        raise AssertionError("edge weights violate the frozen range")


def _write_manifest(rows: list[dict[str, object]]) -> None:
    temp_path = MANIFEST_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, MANIFEST_PATH)


def generate_all(*, overwrite: bool = False) -> None:
    BASE_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for stratum, config in STRATA.items():
        for base_seed in config["seeds"]:
            spec = generate_base_spec(stratum, int(base_seed))
            _assert_dag(spec)
            spec_path = BASE_SPEC_DIR / f"{stratum}_seed_{base_seed}.npz"
            if overwrite or not spec_path.exists():
                np.savez_compressed(
                    spec_path,
                    adjacency=spec.adjacency,
                    edge_sources=spec.edge_sources,
                    edge_targets=spec.edge_targets,
                    edge_weights=spec.edge_weights,
                    topological_order=spec.topological_order,
                    noise=spec.noise,
                    rff_omega=spec.rff_omega,
                    rff_phase=spec.rff_phase,
                    rff_coeff=spec.rff_coeff,
                    tanh_slope=spec.tanh_slope,
                    tanh_bias=spec.tanh_bias,
                )
            for lambda_value in config["lambdas"]:
                x = generate_dataset(stratum, spec, float(lambda_value))
                lambda_tag = f"{float(lambda_value):.2f}".replace(".", "p")
                dataset_id = f"{stratum}_seed_{base_seed}_lambda_{lambda_tag}"
                dataset_path = DATASET_DIR / f"{dataset_id}.npz"
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
                        "dataset_id": dataset_id,
                        "stratum": stratum,
                        "base_seed": int(base_seed),
                        "lambda": float(lambda_value),
                        "n_samples": int(config["n_samples"]),
                        "n_variables": int(config["n_variables"]),
                        "mechanism": str(config["mechanism"]),
                        "noise": str(config["noise"]),
                        "edge_count": int(spec.adjacency.sum()),
                        "dataset_path": dataset_path.relative_to(ROOT).as_posix(),
                        "base_spec_path": spec_path.relative_to(ROOT).as_posix(),
                    }
                )
    _write_manifest(rows)
    print(f"Generated {len(rows)} Gate-1 synthetic datasets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
