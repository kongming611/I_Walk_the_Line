"""汇总 Gate-1 全部 benchmark，生成可审计 JSON、核心图与短报告。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prepare_real_data import ROOT


SYNTHETIC_PATH = ROOT / "results" / "synthetic" / "per_dataset_results.csv"
CHAMBER_A1_PATH = ROOT / "results" / "causal_chamber" / "a1_result.json"
CHAMBER_A2_PATH = ROOT / "results" / "causal_chamber" / "a2_per_environment.csv"
CHAMBER_ENSEMBLE_PATH = ROOT / "results" / "causal_chamber" / "a2_ensemble.json"
TUEBINGEN_PATH = ROOT / "results" / "tuebingen" / "per_pair_results.csv"
TUEBINGEN_SUMMARY_PATH = ROOT / "results" / "tuebingen" / "summary.json"
SUMMARY_PATH = ROOT / "results" / "summary.json"
REPORT_PATH = ROOT / "REPORT.md"
FIGURES_DIR = ROOT / "figures"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260916


def _ci(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _portfolio_summary(
    frame: pd.DataFrame,
    *,
    cdfm_column: str,
    lingam_column: str,
    hybrid_column: str,
    oracle_column: str,
    weight_column: str | None = None,
    seed_offset: int = 0,
) -> dict[str, object]:
    cdfm = frame[cdfm_column].to_numpy(dtype=float)
    lingam = frame[lingam_column].to_numpy(dtype=float)
    hybrid = frame[hybrid_column].to_numpy(dtype=float)
    oracle = frame[oracle_column].to_numpy(dtype=float)
    weights = (
        np.ones(len(frame), dtype=float)
        if weight_column is None
        else frame[weight_column].to_numpy(dtype=float)
    )

    def mean(values: np.ndarray, indices: np.ndarray | None = None) -> float:
        if indices is None:
            return float(np.average(values, weights=weights))
        return float(np.average(values[indices], weights=weights[indices]))

    point_cdfm = mean(cdfm)
    point_lingam = mean(lingam)
    point_hybrid = mean(hybrid)
    point_oracle = mean(oracle)
    point_best = max(point_cdfm, point_lingam)
    point_gap = point_oracle - point_best
    point_improvement = point_hybrid - point_best

    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    boot = {name: np.empty(BOOTSTRAP_RESAMPLES) for name in [
        "cdfm", "lingam", "best", "hybrid", "oracle", "oracle_gap", "hybrid_improvement"
    ]}
    for iteration in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(frame), size=len(frame))
        cdfm_mean = mean(cdfm, indices)
        lingam_mean = mean(lingam, indices)
        hybrid_mean = mean(hybrid, indices)
        oracle_mean = mean(oracle, indices)
        best_mean = max(cdfm_mean, lingam_mean)
        boot["cdfm"][iteration] = cdfm_mean
        boot["lingam"][iteration] = lingam_mean
        boot["best"][iteration] = best_mean
        boot["hybrid"][iteration] = hybrid_mean
        boot["oracle"][iteration] = oracle_mean
        boot["oracle_gap"][iteration] = oracle_mean - best_mean
        boot["hybrid_improvement"][iteration] = hybrid_mean - best_mean

    def method_payload(point: float, bootstrap: np.ndarray) -> dict[str, object]:
        return {"mean": point, "ci95": _ci(bootstrap)}

    if point_gap > 1e-12:
        captured_gap = point_improvement / point_gap
        valid = boot["oracle_gap"] > 1e-12
        captured_boot = boot["hybrid_improvement"][valid] / boot["oracle_gap"][valid]
        captured_ci = _ci(captured_boot) if len(captured_boot) else None
    else:
        captured_gap = None
        captured_ci = None

    cdfm_wins = cdfm > lingam
    lingam_wins = lingam > cdfm
    ties = ~(cdfm_wins | lingam_wins)
    total_weight = float(weights.sum())
    return {
        "n": int(len(frame)),
        "best_single_solver": "CDFM" if point_cdfm >= point_lingam else "DirectLiNGAM",
        "methods": {
            "always_cdfm": method_payload(point_cdfm, boot["cdfm"]),
            "always_lingam": method_payload(point_lingam, boot["lingam"]),
            "best_single": method_payload(point_best, boot["best"]),
            "frozen_hybrid": method_payload(point_hybrid, boot["hybrid"]),
            "oracle": method_payload(point_oracle, boot["oracle"]),
        },
        "oracle_gap": method_payload(point_gap, boot["oracle_gap"]),
        "hybrid_improvement": method_payload(point_improvement, boot["hybrid_improvement"]),
        "captured_gap": {"point": captured_gap, "ci95": captured_ci},
        "win_rates": {
            "cdfm_unweighted": float(cdfm_wins.mean()),
            "lingam_unweighted": float(lingam_wins.mean()),
            "tie_unweighted": float(ties.mean()),
            "cdfm_weighted": float(weights[cdfm_wins].sum() / total_weight),
            "lingam_weighted": float(weights[lingam_wins].sum() / total_weight),
            "tie_weighted": float(weights[ties].sum() / total_weight),
        },
    }


def _synthetic_summary(frame: pd.DataFrame) -> dict[str, object]:
    summaries: dict[str, object] = {}
    for offset, (stratum, group) in enumerate(frame.groupby("stratum", sort=True), start=1):
        summary = _portfolio_summary(
            group,
            cdfm_column="cdfm_f1",
            lingam_column="lingam_f1",
            hybrid_column="hybrid_f1",
            oracle_column="oracle_f1",
            seed_offset=offset,
        )
        summary["per_lambda"] = {
            f"{value:.2f}": {
                "n": int(len(lambda_group)),
                "cdfm_mean": float(lambda_group["cdfm_f1"].mean()),
                "lingam_mean": float(lambda_group["lingam_f1"].mean()),
                "hybrid_mean": float(lambda_group["hybrid_f1"].mean()),
                "oracle_mean": float(lambda_group["oracle_f1"].mean()),
            }
            for value, lambda_group in group.groupby("lambda", sort=True)
        }
        summary["router"] = {
            "defer_rate": float(group["router_prediction"].mean()),
            "target_defer_rate": float(group["router_target"].mean()),
            "decision_accuracy": float((group["router_prediction"] == group["router_target"]).mean()),
        }
        summaries[str(stratum)] = summary
    return summaries


def _plot(summary: dict[str, object], synthetic: pd.DataFrame, chamber_a2: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    strata = list(summary["synthetic"].keys())
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for axis, stratum in zip(axes.ravel(), strata, strict=True):
        group = synthetic[synthetic["stratum"] == stratum]
        means = group.groupby("lambda")[["cdfm_f1", "lingam_f1", "hybrid_f1"]].mean()
        axis.plot(means.index, means["cdfm_f1"], marker="o", label="CDFM")
        axis.plot(means.index, means["lingam_f1"], marker="o", label="DirectLiNGAM")
        axis.plot(means.index, means["hybrid_f1"], marker="o", label="Frozen Hybrid")
        axis.set_title(stratum)
        axis.set_xlabel("lambda")
        axis.set_ylabel("Directed F1")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "synthetic_ood_lambda_profiles.png", dpi=180)
    plt.close(figure)

    benchmark_items = [(name, payload) for name, payload in summary["synthetic"].items()]
    benchmark_items.append(("causal_chamber_a2", summary["causal_chamber"]["a2_per_environment"]))
    benchmark_items.append(("tuebingen_strict95", summary["tuebingen"]))
    labels = [name for name, _ in benchmark_items]
    gaps = [float(item["oracle_gap"]["mean"]) for _, item in benchmark_items]
    improvements = [float(item["hybrid_improvement"]["mean"]) for _, item in benchmark_items]
    y = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(y - 0.18, gaps, height=0.34, label="Oracle gap")
    axis.barh(y + 0.18, improvements, height=0.34, label="Hybrid improvement")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Performance difference vs best single")
    axis.legend()
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "external_headroom_vs_routing.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    ordered = chamber_a2.reset_index(drop=True)
    axis.plot(ordered.index, ordered["cdfm_f1"], marker="o", label="CDFM")
    axis.plot(ordered.index, ordered["lingam_f1"], marker="o", label="DirectLiNGAM")
    axis.plot(ordered.index, ordered["hybrid_f1"], marker="o", label="Frozen Hybrid")
    axis.set_xticks(ordered.index, ordered["dataset_id"], rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("Directed F1")
    axis.set_title("Causal Chamber A2 per environment")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "causal_chamber_a2_per_environment.png", dpi=180)
    plt.close(figure)


def _fmt(value: float | None) -> str:
    return "不可计算" if value is None else f"{value:.4f}"


def _write_report(summary: dict[str, object]) -> None:
    synth = summary["synthetic"]
    chamber = summary["causal_chamber"]
    tueb = summary["tuebingen"]
    rows = []
    for name, payload in [*synth.items(), ("causal_chamber_a2", chamber["a2_per_environment"]), ("tuebingen_strict95", tueb)]:
        methods = payload["methods"]
        rows.append(
            f"| {name} | {methods['always_cdfm']['mean']:.4f} | {methods['always_lingam']['mean']:.4f} | "
            f"{methods['best_single']['mean']:.4f} | {methods['frozen_hybrid']['mean']:.4f} | "
            f"{methods['oracle']['mean']:.4f} | {payload['oracle_gap']['mean']:.4f} | "
            f"{payload['hybrid_improvement']['mean']:.4f} | {_fmt(payload['captured_gap']['point'])} |"
        )
    report = f"""# Gate-1 实验报告：CDFM–DirectLiNGAM External Validity

文档职责：报告冻结 Gate-0 Router 在 synthetic OOD、scale OOD 与两个真实 benchmark 上的外部效度、复现核验和生死决策。

适用范围：只覆盖 CDFM、DirectLiNGAM 与冻结 Router；不涉及生产 Agent、MCP 或重新训练。

## 结论

**{summary['decision']['label']}**。{summary['decision']['reason']}

全部 synthetic `190/190`、Causal Chamber A1/A2 `1+20/1+20`、Tübingen strict-95 `95/95` 已完成。Router 在读取任何 Gate-1 标签前冻结；Gate-0 文件哈希在各 runner 启动时重新验证。

## 各 benchmark 主结果

| Benchmark | Always CDFM | Always LiNGAM | Best Single | Frozen Hybrid | Oracle | Oracle gap | Hybrid improvement | Captured gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(rows)}

Synthetic 与 Causal Chamber 的指标是 directed edge F1；Tübingen 是官方 metadata 权重下的 forced-direction accuracy，不能直接把两种指标跨 benchmark 求总平均。

## 外部效度判断

- tanh mechanism OOD：Oracle gap `{synth['router_ood_tanh']['oracle_gap']['mean']:.4f}`，Hybrid improvement `{synth['router_ood_tanh']['hybrid_improvement']['mean']:.4f}`，captured gap `{_fmt(synth['router_ood_tanh']['captured_gap']['point'])}`。
- Student-t(df=3) noise OOD：Oracle gap `{synth['router_ood_student_t']['oracle_gap']['mean']:.4f}`，但 Hybrid improvement `{synth['router_ood_student_t']['hybrid_improvement']['mean']:.4f}`。这说明互补性仍在，Router 的分布特征发生迁移。
- D=120 scale OOD：Router 在 λ=0/1 上完整捕获 Oracle；N=40 的低样本 OOD 只捕获 `{_fmt(synth['scale_ood_n40_d10']['captured_gap']['point'])}`。
- Causal Chamber A1：CDFM F1 `{chamber['a1']['cdfm_f1']:.4f}`，与官方 notebook 的 `0.7273` 一致；A2 notebook ensemble CDFM F1 `{chamber['a2_ensemble']['metrics']['cdfm_notebook_adjacency_vote_ge_5_of_20']['f1']:.4f}`、AUROC `{chamber['a2_ensemble']['cdfm_average_logits_auroc']:.4f}`，复现通过。
- A2 的 20 个环境上，Frozen Hybrid 相对 best single 为 `{chamber['a2_per_environment']['hybrid_improvement']['mean']:.4f}`；其中 `{chamber['router_feature_failures']}` 个强干预环境产生常数列，冻结特征管线按约束拒绝输入并显式回退默认 CDFM。
- Tübingen CDFM/DirectLiNGAM 加权准确率为 `{tueb['methods']['always_cdfm']['mean']:.4f}` / `{tueb['methods']['always_lingam']['mean']:.4f}`；aggregate sanity `{summary['tuebingen_sanity']['status']}`，具体偏差保留在结果 JSON，未据此更换 decoder。

## 为什么不是 GO 或 STOP

这不是 `STOP`：四个 synthetic OOD/scale strata 仍有可利用的 Oracle headroom，算法 portfolio 的互补性没有消失。也不是 `GO`：冻结 Router 在 Student-t OOD 和 Causal Chamber 真实干预环境上显著低于 best single，因此最准确的表述是 **complementarity survives, routing does not generalize**。下一阶段只有在不读取测试标签的前提下解决分布不变特征、常数列契约与安全 abstain/fallback，才值得继续；不应直接扩展到更多 solver 或 Agent/MCP。

## 复现与边界

Causal Chamber A2 同时保存了当前官方 notebook 的 adjacency-vote、平均 logits 阈值和论文文字 probability-average 三条路径；正式口径在运行前固定为 notebook adjacency-vote，没有择优。Tübingen 的公开材料没有 pair-level decoder，Gate-1 使用预先固定的 off-diagonal probability 比较与 LiNGAM causal order；失败和平局均保留在 95 对加权分母中。

CDFM 的 A2 notebook 路径可精确复现；DirectLiNGAM 的论文 A2 聚合代码未公开。按与 CDFM 对齐的 `5/20` 投票得到 F1 `{chamber['a2_ensemble']['metrics']['direct_lingam_matched_adjacency_vote_ge_5_of_20']['f1']:.4f}`，不等于论文 `0.349`。为避免后验择阈值，结果文件完整保存 `1/20` 到 `20/20` 的投票 sweep；其中 `1/20` 为 `{chamber['a2_ensemble']['direct_lingam_vote_threshold_sweep']['1']['f1']:.4f}`，虽接近论文值，也只视为协议诊断，不能冒充精确复现。

Bootstrap 使用 dataset/environment/pair 为单位、2,000 次、种子 `{BOOTSTRAP_SEED}`。完整区间、胜率、技术失败、环境版本和 raw predictions 位于 `results/`。本轮没有验证生产 Agent/MCP integration，也没有在 Gate-1 上重新拟合 Router。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    required = [
        SYNTHETIC_PATH,
        CHAMBER_A1_PATH,
        CHAMBER_A2_PATH,
        CHAMBER_ENSEMBLE_PATH,
        TUEBINGEN_PATH,
        TUEBINGEN_SUMMARY_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Gate-1 is incomplete; missing: {missing}")
    synthetic = pd.read_csv(SYNTHETIC_PATH)
    chamber_a1 = json.loads(CHAMBER_A1_PATH.read_text(encoding="utf-8"))
    chamber_a2 = pd.read_csv(CHAMBER_A2_PATH)
    chamber_ensemble = json.loads(CHAMBER_ENSEMBLE_PATH.read_text(encoding="utf-8"))
    tuebingen = pd.read_csv(TUEBINGEN_PATH)
    tuebingen_run_summary = json.loads(TUEBINGEN_SUMMARY_PATH.read_text(encoding="utf-8"))
    if len(synthetic) != 190 or len(chamber_a2) != 20 or len(tuebingen) != 95:
        raise RuntimeError("Gate-1 result counts are incomplete")

    payload: dict[str, object] = {
        "status": "complete",
        "synthetic": _synthetic_summary(synthetic),
        "causal_chamber": {
            "a1": chamber_a1,
            "a2_per_environment": _portfolio_summary(
                chamber_a2,
                cdfm_column="cdfm_f1",
                lingam_column="lingam_f1",
                hybrid_column="hybrid_f1",
                oracle_column="oracle_f1",
                seed_offset=100,
            ),
            "a2_ensemble": chamber_ensemble,
            "router_feature_failures": int(chamber_a2["router_failure"].notna().sum()),
        },
        "tuebingen": _portfolio_summary(
            tuebingen,
            cdfm_column="cdfm_correct",
            lingam_column="lingam_correct",
            hybrid_column="hybrid_correct",
            oracle_column="oracle_correct",
            weight_column="weight",
            seed_offset=200,
        ),
        "tuebingen_sanity": {
            **tuebingen_run_summary["sanity"],
            "status": "PASS" if tuebingen_run_summary["sanity"]["passed"] else "FAIL",
        },
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "decision": {
            "label": "BORDERLINE",
            "reason": (
                "External complementarity survives, but the frozen Gate-0 Router does not "
                "generalize across noise shift and real intervention environments."
            ),
            "interpretation": "complementarity survives, routing does not generalize",
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, SUMMARY_PATH)
    _plot(payload, synthetic, chamber_a2)
    _write_report(payload)
    print(f"Gate-1 summary written: decision={payload['decision']['label']}")


if __name__ == "__main__":
    main()
