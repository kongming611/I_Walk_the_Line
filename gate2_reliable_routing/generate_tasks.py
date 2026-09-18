"""生成 Gate-2 的独立源域、校准域与未见机制迁移任务。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TASK_DIR = DATA_DIR / "tasks"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SMOKE_MANIFEST_PATH = DATA_DIR / "smoke_manifest.csv"
MASTER_SEED = 2026091701
N_VARIABLES = 10
N_SAMPLES = 1000
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_TOTAL_DEGREE = 2.0
RFF_FEATURES = 30
NOISE_SCALE = 1.0 / np.sqrt(2.0)

SOURCE_DOMAINS = (
    {"domain": "src_rff_laplace", "mechanism": "rff", "noise": "laplace"},
    {"domain": "src_rff_gaussian", "mechanism": "rff", "noise": "gaussian"},
    {"domain": "src_tanh_laplace", "mechanism": "tanh", "noise": "laplace"},
    {"domain": "src_tanh_gaussian", "mechanism": "tanh", "noise": "gaussian"},
    {"domain": "src_softsign_laplace", "mechanism": "softsign", "noise": "laplace"},
    {"domain": "src_interaction_studentt8", "mechanism": "interaction", "noise": "student_t8"},
)
TARGET_DOMAINS = (
    {"domain": "tgt_rff_studentt3", "mechanism": "rff", "noise": "student_t3"},
    {"domain": "tgt_tanh_studentt3", "mechanism": "tanh", "noise": "student_t3"},
    {"domain": "tgt_softsign_gaussian", "mechanism": "softsign", "noise": "gaussian"},
    {"domain": "tgt_sine_laplace", "mechanism": "sine", "noise": "laplace"},
    {"domain": "tgt_rff_exponential", "mechanism": "rff", "noise": "exponential"},
    {"domain": "tgt_interaction_exponential", "mechanism": "interaction", "noise": "exponential"},
)
SPLIT_COUNTS = {"train": 600, "dev": 300, "calibration": 600}
TARGET_TASKS_PER_DOMAIN = 300


@dataclass(frozen=True)
class GraphSpec:
    adjacency: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    order: np.ndarray
    rff_omega: np.ndarray
    rff_phase: np.ndarray
    rff_coeff: np.ndarray
    slopes: np.ndarray
    biases: np.ndarray


def _rng(seed: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, seed, *parts]))


def _normalize(value: np.ndarray) -> np.ndarray:
    centered = value - np.mean(value)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("zero or non-finite component scale")
    return centered / scale


def _graph(seed: int) -> GraphSpec:
    rng = _rng(seed, 1)
    order = rng.permutation(N_VARIABLES)
    adjacency = np.zeros((N_VARIABLES, N_VARIABLES), dtype=np.int8)
    probability = EXPECTED_TOTAL_DEGREE / (N_VARIABLES - 1)
    for left in range(N_VARIABLES):
        for right in range(left + 1, N_VARIABLES):
            if rng.random() < probability:
                adjacency[order[left], order[right]] = 1
    if not adjacency.any():
        adjacency[order[0], order[1]] = 1
    sources, targets = np.nonzero(adjacency)
    count = len(sources)
    return GraphSpec(
        adjacency=adjacency,
        sources=sources.astype(np.int64),
        targets=targets.astype(np.int64),
        weights=rng.choice(np.array([-1.0, 1.0]), count) * rng.uniform(0.5, 1.5, count),
        order=order.astype(np.int64),
        rff_omega=rng.normal(size=(count, RFF_FEATURES)),
        rff_phase=rng.uniform(0, 2 * np.pi, size=(count, RFF_FEATURES)),
        rff_coeff=rng.normal(size=(count, RFF_FEATURES)),
        slopes=rng.uniform(0.5, 2.0, count),
        biases=rng.uniform(-1.0, 1.0, count),
    )


def _noise(kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "laplace":
        return rng.laplace(0, NOISE_SCALE, size=(N_SAMPLES, N_VARIABLES))
    if kind == "gaussian":
        return rng.normal(0, 1, size=(N_SAMPLES, N_VARIABLES))
    if kind == "student_t8":
        return rng.standard_t(8, size=(N_SAMPLES, N_VARIABLES)) * np.sqrt(6.0 / 8.0)
    if kind == "student_t3":
        return rng.standard_t(3, size=(N_SAMPLES, N_VARIABLES)) / np.sqrt(3.0)
    if kind == "exponential":
        return rng.exponential(1.0, size=(N_SAMPLES, N_VARIABLES)) - 1.0
    raise ValueError(f"unknown noise: {kind}")


def _edge_transform(kind: str, x: np.ndarray, omega: np.ndarray, phase: np.ndarray, coeff: np.ndarray, slope: float, bias: float) -> np.ndarray:
    if kind == "rff":
        return np.sqrt(2.0 / RFF_FEATURES) * (np.cos(x[:, None] * omega[None, :] + phase[None, :]) @ coeff)
    if kind == "tanh":
        return np.tanh(slope * x + bias)
    if kind == "softsign":
        return (slope * x + bias) / (1.0 + np.abs(slope * x + bias))
    if kind == "sine":
        return np.sin(1.3 * x + bias)
    if kind == "interaction":
        return np.tanh(slope * x + bias)
    raise ValueError(f"unknown mechanism: {kind}")


def generate_x(spec: GraphSpec, mechanism: str, noise_kind: str, lambda_value: float, seed: int) -> np.ndarray:
    rng = _rng(seed, 2)
    exogenous = _noise(noise_kind, rng)
    x = np.zeros((N_SAMPLES, N_VARIABLES), dtype=np.float64)
    incoming = {node: np.flatnonzero(spec.targets == node) for node in range(N_VARIABLES)}
    for child in spec.order:
        child = int(child)
        indices = incoming[child]
        if len(indices) == 0:
            x[:, child] = exogenous[:, child]
            continue
        linear = np.zeros(N_SAMPLES)
        nonlinear = np.zeros(N_SAMPLES)
        parents: list[np.ndarray] = []
        for edge_index in indices:
            source = int(spec.sources[edge_index])
            parents.append(x[:, source])
            weight = float(spec.weights[edge_index])
            linear += weight * x[:, source]
            nonlinear += weight * _edge_transform(
                mechanism,
                x[:, source],
                spec.rff_omega[edge_index],
                spec.rff_phase[edge_index],
                spec.rff_coeff[edge_index],
                float(spec.slopes[edge_index]),
                float(spec.biases[edge_index]),
            )
        if mechanism == "interaction" and len(parents) >= 2:
            for left in range(len(parents)):
                for right in range(left + 1, len(parents)):
                    nonlinear += 0.35 * np.tanh(parents[left]) * np.tanh(parents[right])
        x[:, child] = (1.0 - lambda_value) * _normalize(linear) + lambda_value * _normalize(nonlinear) + exogenous[:, child]
    if not np.isfinite(x).all() or np.any(np.std(x, axis=0) <= 1e-12):
        raise ValueError("generated X is invalid")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x.astype(np.float32)


def _task_row(task_id: str, split: str, domain: dict[str, str], index: int, graph_seed: int, lambda_value: float) -> dict[str, object]:
    return {
        "task_id": task_id,
        "split": split,
        "domain": domain["domain"],
        "mechanism": domain["mechanism"],
        "noise": domain["noise"],
        "task_index": index,
        "graph_seed": graph_seed,
        "lambda": lambda_value,
        "n_samples": N_SAMPLES,
        "n_variables": N_VARIABLES,
        "path": f"data/tasks/{task_id}.npz",
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def generate_all(*, overwrite: bool = False) -> None:
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    domain_by_name = {item["domain"]: item for item in (*SOURCE_DOMAINS, *TARGET_DOMAINS)}
    source_index = 0
    for split, count in SPLIT_COUNTS.items():
        for local_index in range(count):
            domain = SOURCE_DOMAINS[source_index % len(SOURCE_DOMAINS)]
            source_index += 1
            graph_seed = 100000 + source_index
            lambda_value = LAMBDA_GRID[local_index % len(LAMBDA_GRID)]
            task_id = f"{split}_{domain['domain']}_{local_index:04d}"
            row = _task_row(task_id, split, domain, local_index, graph_seed, lambda_value)
            _write_task(row, domain_by_name[domain["domain"]], overwrite)
            rows.append(row)
    target_index = 0
    for target in TARGET_DOMAINS:
        for local_index in range(TARGET_TASKS_PER_DOMAIN):
            graph_seed = 200000 + target_index
            lambda_value = LAMBDA_GRID[local_index % len(LAMBDA_GRID)]
            task_id = f"test_{target['domain']}_{local_index:04d}"
            row = _task_row(task_id, "test", target, local_index, graph_seed, lambda_value)
            _write_task(row, target, overwrite)
            rows.append(row)
            target_index += 1
    _write_manifest(MANIFEST_PATH, rows)

    smoke_rows = []
    for domain in SOURCE_DOMAINS:
        smoke_rows.extend([row for row in rows if row["split"] == "train" and row["domain"] == domain["domain"]][:4])
    _write_manifest(SMOKE_MANIFEST_PATH, smoke_rows)
    manifest_hash = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    (DATA_DIR / "generation_manifest.json").write_text(
        json.dumps({"schema_version": "gate2_tasks_v1", "task_count": len(rows), "smoke_count": len(smoke_rows), "manifest_sha256": manifest_hash, "source_domains": SOURCE_DOMAINS, "target_domains": TARGET_DOMAINS, "split_counts": SPLIT_COUNTS, "target_tasks_per_domain": TARGET_TASKS_PER_DOMAIN, "seed": MASTER_SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Gate-2 tasks generated: {len(rows)} full / {len(smoke_rows)} smoke")


def _write_task(row: dict[str, object], domain: dict[str, str], overwrite: bool) -> None:
    path = ROOT / str(row["path"])
    if path.exists() and not overwrite:
        return
    spec = _graph(int(row["graph_seed"]))
    x = generate_x(spec, domain["mechanism"], domain["noise"], float(row["lambda"]), int(row["graph_seed"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    # Passing an open handle prevents NumPy from appending a second ``.npz``
    # suffix; the rename is therefore genuinely atomic on the target volume.
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, X=x, truth_adjacency=spec.adjacency, graph_seed=np.int64(row["graph_seed"]), lambda_value=np.float64(row["lambda"]))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate_all(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
