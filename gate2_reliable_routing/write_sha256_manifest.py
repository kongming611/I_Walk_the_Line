"""写入 Gate-2 交付产物的 SHA-256 清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "sha256_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    include_roots = [ROOT / "data", ROOT / "results", ROOT / "frozen", ROOT / "REPORT.md", ROOT / "README.md"]
    paths: set[Path] = set()
    for item in include_roots:
        if item.is_file():
            paths.add(item)
        elif item.is_dir():
            paths.update(path for path in item.rglob("*") if path.is_file() and path != OUTPUT)
    paths.update(path for path in ROOT.glob("*.py") if path.is_file())
    entries = []
    for path in sorted(paths):
        entries.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    payload = {
        "schema_version": "gate2_sha256_manifest_v1",
        "file_count": len(entries),
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": len(entries), "manifest": OUTPUT.relative_to(ROOT).as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
