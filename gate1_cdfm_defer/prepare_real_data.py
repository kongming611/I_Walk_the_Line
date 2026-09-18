"""下载并验证 Gate-1 的 Tübingen 与 CDFM Causal Chamber 官方输入。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REAL_DIR = ROOT / "data" / "real"
TUEBINGEN_DIR = REAL_DIR / "tuebingen"
CHAMBER_DIR = REAL_DIR / "causal_chamber"
SOURCE_MANIFEST_PATH = REAL_DIR / "source_manifest.json"

TUEBINGEN_URL = "https://webdav.tuebingen.mpg.de/cause-effect/pairs_1.0.zip"
CDFM_COMMIT = "996816049ac5d36fc85e3c2b501509cbd8bc3d7f"
CDFM_RAW_BASE = (
    "https://raw.githubusercontent.com/DMIRLAB-Group/CDFM/"
    f"{CDFM_COMMIT}/tests/data/causal_chamber"
)

DAG_VARS = [
    "red",
    "green",
    "blue",
    "current",
    "pol_1",
    "pol_2",
    "ir_1",
    "vis_1",
    "ir_2",
    "vis_2",
    "ir_3",
    "vis_3",
    "angle_1",
    "angle_2",
    "l_11",
    "l_12",
    "l_21",
    "l_22",
    "l_31",
    "l_32",
]

A2_ENVIRONMENTS = [
    "uniform_red_strong",
    "uniform_green_strong",
    "uniform_blue_strong",
    "uniform_v_c_strong",
    "uniform_t_ir_1_strong",
    "uniform_t_ir_2_strong",
    "uniform_t_ir_3_strong",
    "uniform_t_vis_1_strong",
    "uniform_t_vis_2_strong",
    "uniform_t_vis_3_strong",
    "uniform_pol_1_strong",
    "uniform_pol_2_strong",
    "uniform_v_angle_1_strong",
    "uniform_v_angle_2_strong",
    "uniform_l_11_mid",
    "uniform_l_12_mid",
    "uniform_l_21_mid",
    "uniform_l_22_mid",
    "uniform_l_31_mid",
    "uniform_l_32_mid",
]

STRICT_95_IDS = [
    pair_id
    for pair_id in range(1, 101)
    if pair_id not in {52, 53, 54, 55, 71}
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "CausalAgent-Gate1/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(temp_path, destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        return
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"unsafe ZIP member: {member.filename}")
        zipped.extractall(destination)


def _find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {filename}, found {len(matches)}")
    return matches[0]


def _parse_metadata(path: Path) -> dict[int, dict[str, float | int]]:
    metadata: dict[int, dict[str, float | int]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw_line.split()
        if len(fields) != 6 or not fields[0].isdigit():
            continue
        pair_id, cause_first, cause_last, effect_first, effect_last = map(int, fields[:5])
        metadata[pair_id] = {
            "cause_first": cause_first,
            "cause_last": cause_last,
            "effect_first": effect_first,
            "effect_last": effect_last,
            "weight": float(fields[5]),
        }
    return metadata


def prepare_tuebingen() -> dict[str, object]:
    archive = TUEBINGEN_DIR / "pairs_1.0.zip"
    extract_dir = TUEBINGEN_DIR / "pairs_1_0"
    _download(TUEBINGEN_URL, archive)
    _safe_extract(archive, extract_dir)
    metadata_path = _find_one(extract_dir, "pairmeta.txt")
    metadata = _parse_metadata(metadata_path)
    if len(STRICT_95_IDS) != 95:
        raise AssertionError("strict pair list must contain 95 ids")
    rows: list[dict[str, object]] = []
    for pair_id in STRICT_95_IDS:
        if pair_id not in metadata:
            raise RuntimeError(f"missing metadata for pair {pair_id:04d}")
        meta = metadata[pair_id]
        cause_width = int(meta["cause_last"]) - int(meta["cause_first"]) + 1
        effect_width = int(meta["effect_last"]) - int(meta["effect_first"]) + 1
        if cause_width != 1 or effect_width != 1:
            raise RuntimeError(f"pair {pair_id:04d} is not strict bivariate")
        pair_path = _find_one(extract_dir, f"pair{pair_id:04d}.txt")
        data = np.loadtxt(pair_path)
        if data.ndim != 2:
            raise RuntimeError(f"pair {pair_id:04d} has invalid data")
        cause_column = int(meta["cause_first"]) - 1
        effect_column = int(meta["effect_first"]) - 1
        if cause_column == effect_column or max(cause_column, effect_column) >= data.shape[1]:
            raise RuntimeError(f"pair {pair_id:04d} has invalid direction columns")
        selected = data[:, [cause_column, effect_column]]
        if not np.isfinite(selected).all():
            raise RuntimeError(f"pair {pair_id:04d} has non-finite cause/effect values")
        rows.append(
            {
                "pair_id": f"pair{pair_id:04d}",
                "path": pair_path.relative_to(ROOT).as_posix(),
                "n_samples": int(data.shape[0]),
                "raw_column_count": int(data.shape[1]),
                "cause_column": cause_column,
                "effect_column": effect_column,
                "weight": float(meta["weight"]),
                "sha256": sha256(pair_path),
            }
        )
    manifest_path = TUEBINGEN_DIR / "strict95_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return {
        "url": TUEBINGEN_URL,
        "archive_path": archive.relative_to(ROOT).as_posix(),
        "archive_sha256": sha256(archive),
        "metadata_sha256": sha256(metadata_path),
        "strict_pair_count": len(rows),
        "total_weight": float(sum(float(row["weight"]) for row in rows)),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
    }


def prepare_causal_chamber() -> dict[str, object]:
    files = ["dag.csv", "task_a1/uniform_reference.csv"] + [
        f"task_a2/{name}.csv" for name in A2_ENVIRONMENTS
    ]
    file_records: list[dict[str, object]] = []
    for relative in files:
        destination = CHAMBER_DIR / relative
        url = f"{CDFM_RAW_BASE}/{relative}"
        _download(url, destination)
        file_records.append(
            {
                "relative_path": destination.relative_to(ROOT).as_posix(),
                "url": url,
                "sha256": sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
        )

    dag = pd.read_csv(CHAMBER_DIR / "dag.csv")
    if list(dag.columns) != ["source", "target"]:
        raise RuntimeError("CDFM Causal Chamber ground truth must be a source-target edge list")
    # 官方文件是比 notebook 的 20 变量更大的图；Gate-1 按变量名取诱导子图。
    induced = dag[dag["source"].isin(DAG_VARS) & dag["target"].isin(DAG_VARS)]
    if len(induced) != 39:
        raise RuntimeError("CDFM 20-variable induced ground truth must contain 39 edges")

    a1 = pd.read_csv(CHAMBER_DIR / "task_a1" / "uniform_reference.csv")
    if len(a1) != 10000 or not set(DAG_VARS).issubset(a1.columns):
        raise RuntimeError("A1 must contain 10,000 rows and all 20 variables")
    for environment in A2_ENVIRONMENTS:
        frame = pd.read_csv(CHAMBER_DIR / "task_a2" / f"{environment}.csv")
        if len(frame) != 1000 or not set(DAG_VARS).issubset(frame.columns):
            raise RuntimeError(f"A2 environment {environment} failed row/column validation")
    return {
        "source_repository": "https://github.com/DMIRLAB-Group/CDFM",
        "source_commit": CDFM_COMMIT,
        "variable_order": DAG_VARS,
        "a1_environment": "uniform_reference",
        "a2_environments": A2_ENVIRONMENTS,
        "full_ground_truth_edge_count": int(len(dag)),
        "induced_ground_truth_edge_count": int(len(induced)),
        "files": file_records,
    }


def main() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "gate1_real_sources_v1",
        "tuebingen": prepare_tuebingen(),
        "causal_chamber": prepare_causal_chamber(),
    }
    temp_path = SOURCE_MANIFEST_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, SOURCE_MANIFEST_PATH)
    print(
        "Prepared real benchmarks: Tübingen strict95 and Causal Chamber "
        f"A1 + {len(A2_ENVIRONMENTS)} A2 environments"
    )


if __name__ == "__main__":
    main()
