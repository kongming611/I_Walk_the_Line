"""从已封存的 Gate-2 测试产物生成机制和策略消融汇总。

此脚本只读取最终测试预测与评估表，不重新拟合模型、不改变冻结阈值，
因此可在主评估完成后安全重放。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

METHODS = [
    "always_cdfm", "always_lingam", "old_gate0_router", "logistic_router",
    "gain_regression", "l2d_weighted", "gain_crc", "gain_ltt",
    "conformal_uncertain", "support_gate", "conservative_min_ltt", "support_gain_ltt",
]

FAMILIES = {
    "always_cdfm": "always-baseline", "always_lingam": "always-baseline",
    "old_gate0_router": "legacy-router", "logistic_router": "same-feature-classifier",
    "gain_regression": "gain-prediction", "l2d_weighted": "gain-prediction",
    "gain_crc": "risk-control", "gain_ltt": "risk-control",
    "conformal_uncertain": "uncertainty-fallback",
    "support_gate": "support-ablation", "conservative_min_ltt": "cross-source-conservative",
    "support_gain_ltt": "support-ablation",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _domain_parts(domain: str) -> tuple[str, str]:
    name = str(domain)
    known = ["interaction", "softsign", "studentt3", "exponential", "laplace", "gaussian", "rff", "tanh", "sine"]
    body = name.removeprefix("tgt_")
    for mechanism in ("interaction", "softsign", "rff", "tanh", "sine"):
        prefix = mechanism + "_"
        if body.startswith(prefix):
            return mechanism, body[len(prefix):]
    # Keep unexpected future domain names visible instead of silently dropping them.
    for token in known:
        if body.startswith(token + "_"):
            return token, body[len(token) + 1:]
    return body, "unknown"


def build_mechanism_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, local in predictions.groupby("domain", sort=True):
        mechanism, noise = _domain_parts(domain)
        cdfm = local["cdfm_f1"].to_numpy(float)
        lingam = local["lingam_f1"].to_numpy(float)
        row: dict[str, object] = {
            "domain": domain, "target_mechanism": mechanism, "target_noise": noise,
            "n": len(local), "cdfm_mean_f1": float(cdfm.mean()),
            "lingam_mean_f1": float(lingam.mean()),
            "oracle_gain_vs_cdfm": float((pd.Series(lingam).combine(pd.Series(cdfm), max) - cdfm).mean()),
            "lingam_win_rate": float((lingam > cdfm).mean()),
            "cdfm_win_rate": float((cdfm > lingam).mean()),
            "tie_rate": float((cdfm == lingam).mean()),
        }
        for method in METHODS:
            row[f"{method}_H"] = float(local[f"{method}__harm"].mean())
            row[f"{method}_T"] = float(local[f"{method}__severe_harm"].mean())
            row[f"{method}_net"] = float((local[f"{method}__hybrid_f1"] - local["cdfm_f1"]).mean())
            row[f"{method}_switch"] = float(local[f"{method}__selected"].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("domain").reset_index(drop=True)


def build_ablation_summary(predictions: pd.DataFrame, evaluation: dict[str, object], domains: pd.DataFrame) -> pd.DataFrame:
    method_values = evaluation["methods"]
    rows: list[dict[str, object]] = []
    for method in METHODS:
        local = domains[domains["method"] == method]
        values = method_values[method]
        rows.append({
            "method": method, "family": FAMILIES[method],
            "mean_harm_H": values["mean_harm_H"], "severe_harm_rate_T": values["severe_harm_rate_T"],
            "net_gain_vs_cdfm": values["net_gain_vs_cdfm"], "switch_rate": values["switch_rate"],
            "max_domain_H": float(local["mean_harm_H"].max()),
            "max_domain_T": float(local["severe_harm_rate_T"].max()),
            "min_domain_net_gain": float(local["net_gain_vs_cdfm"].min()),
            "simultaneous_max_H": values["simultaneous_max_harm_upper"],
            "simultaneous_max_T": values["simultaneous_max_severe_upper"],
            "simultaneous_safe": bool(values["simultaneous_max_harm_upper"] <= 0.02 and values["simultaneous_max_severe_upper"] <= 0.05),
        })
    return pd.DataFrame(rows)


def main() -> None:
    predictions = pd.read_csv(RESULTS / "router_predictions.csv")
    evaluation = json.loads((RESULTS / "evaluation_summary.json").read_text(encoding="utf-8"))
    domains = pd.read_csv(RESULTS / "domain_summary.csv")
    mechanism = build_mechanism_summary(predictions)
    ablation = build_ablation_summary(predictions, evaluation, domains)
    mechanism.to_csv(RESULTS / "mechanism_summary.csv", index=False, lineterminator="\n")
    ablation.to_csv(RESULTS / "ablation_summary.csv", index=False, lineterminator="\n")
    manifest = {
        "schema_version": "gate2_analysis_v1",
        "source_sha256": {
            "router_predictions.csv": _sha256(RESULTS / "router_predictions.csv"),
            "evaluation_summary.json": _sha256(RESULTS / "evaluation_summary.json"),
            "domain_summary.csv": _sha256(RESULTS / "domain_summary.csv"),
            "bootstrap_summary.csv": _sha256(RESULTS / "bootstrap_summary.csv"),
        },
        "notes": [
            "mechanism_summary aggregates the single frozen test evaluation by target mechanism/noise",
            "ablation_summary compares pre-registered policy families; it does not refit on test labels",
        ],
    }
    (RESULTS / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mechanism_rows": len(mechanism), "ablation_rows": len(ablation), "decision": evaluation["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
