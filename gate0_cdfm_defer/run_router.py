"""训练严格按 base seed 分组的简单 Router，汇总 Gate-0 并生成报告。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_data import ROOT
from meta_features import FEATURE_COLUMNS, FEATURES_PATH
from metrics import bootstrap_statistic, percentile_ci
from run_benchmark import ENVIRONMENT_PATH, RESULTS_PATH


RESULTS_DIR = ROOT / "results"
ROUTER_PATH = RESULTS_DIR / "router_predictions.csv"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
REPORT_PATH = ROOT / "REPORT.md"
DEFER_MARGIN = 0.03
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260916


def _ci_dict(estimates: np.ndarray, central: float) -> dict[str, float]:
    lower, upper = percentile_ci(estimates)
    return {"mean": float(central), "ci95_lower": lower, "ci95_upper": upper}


def _bootstrap_mean(values: np.ndarray, seed_offset: int) -> dict[str, float]:
    estimates = bootstrap_statistic(
        len(values),
        lambda indices: float(np.mean(values[indices])),
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    return _ci_dict(estimates, float(np.mean(values)))


def _method_summaries(frame: pd.DataFrame) -> tuple[dict[str, dict[str, float]], str]:
    cdfm = frame["cdfm_f1"].to_numpy(float)
    lingam = frame["lingam_f1"].to_numpy(float)
    hybrid = frame["hybrid_selected_f1"].to_numpy(float)
    oracle = np.maximum(cdfm, lingam)
    mean_cdfm = float(cdfm.mean())
    mean_lingam = float(lingam.mean())
    best_solver = "CDFM" if mean_cdfm >= mean_lingam else "DirectLiNGAM"
    best_values = cdfm if best_solver == "CDFM" else lingam

    summaries = {
        "always_cdfm": _bootstrap_mean(cdfm, 1),
        "always_lingam": _bootstrap_mean(lingam, 2),
        "best_single": _bootstrap_mean(best_values, 3),
        "hybrid_router": _bootstrap_mean(hybrid, 4),
        "oracle": _bootstrap_mean(oracle, 5),
    }
    return summaries, best_solver


def _paired_difference_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    seed_offset: int,
) -> dict[str, float]:
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return _bootstrap_mean(difference, seed_offset)


def _oracle_gap_bootstrap(cdfm: np.ndarray, lingam: np.ndarray) -> dict[str, float]:
    central = float(np.maximum(cdfm, lingam).mean() - max(cdfm.mean(), lingam.mean()))
    estimates = bootstrap_statistic(
        len(cdfm),
        lambda indices: float(
            np.maximum(cdfm[indices], lingam[indices]).mean()
            - max(cdfm[indices].mean(), lingam[indices].mean())
        ),
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + 20,
    )
    return _ci_dict(estimates, central)


def _hybrid_improvement_bootstrap(
    hybrid: np.ndarray,
    cdfm: np.ndarray,
    lingam: np.ndarray,
) -> dict[str, float]:
    central = float(hybrid.mean() - max(cdfm.mean(), lingam.mean()))
    estimates = bootstrap_statistic(
        len(cdfm),
        lambda indices: float(
            hybrid[indices].mean()
            - max(cdfm[indices].mean(), lingam[indices].mean())
        ),
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + 21,
    )
    return _ci_dict(estimates, central)


def _per_lambda_summary(frame: pd.DataFrame) -> dict[str, dict[str, float | dict[str, float]]]:
    output: dict[str, dict[str, float | dict[str, float]]] = {}
    for index, (lambda_value, group) in enumerate(frame.groupby("lambda", sort=True)):
        cdfm = group["cdfm_f1"].to_numpy(float)
        lingam = group["lingam_f1"].to_numpy(float)
        output[f"{float(lambda_value):.2f}"] = {
            "n": int(len(group)),
            "cdfm": _bootstrap_mean(cdfm, 100 + index * 2),
            "lingam": _bootstrap_mean(lingam, 101 + index * 2),
            "mean_lingam_minus_cdfm": float(np.mean(lingam - cdfm)),
        }
    return output


def _gate_decision(
    lambda0_difference: float,
    lambda1_difference: float,
    oracle_gap: float,
    hybrid_improvement: float,
    captured_gap: float | None,
    frame: pd.DataFrame,
) -> tuple[str, str]:
    router_condition = hybrid_improvement >= 0.015 or (
        captured_gap is not None and captured_gap >= 0.50
    )
    if (
        lambda0_difference >= 0.05
        and lambda1_difference >= 0.05
        and oracle_gap >= 0.03
        and router_condition
    ):
        return (
            "GO",
            "两个端点均存在至少 0.05 F1 的相反优势，oracle headroom 至少 0.03，且简单 Router 达到预设利用门槛。",
        )

    cdfm_meaningful = float(np.mean(frame["cdfm_f1"] >= frame["lingam_f1"] + DEFER_MARGIN))
    lingam_meaningful = float(np.mean(frame["lingam_f1"] >= frame["cdfm_f1"] + DEFER_MARGIN))
    minority_region = min(cdfm_meaningful, lingam_meaningful)
    if oracle_gap < 0.015 and minority_region < 0.10:
        return (
            "STOP",
            "oracle headroom 低于 0.015，且两个算法中较少出现的明确优势区域占比低于 10%；portfolio 本身缺少可利用空间。",
        )
    return (
        "BORDERLINE",
        "未同时满足 GO，也不满足严格 STOP；需要区分是算法互补性不足，还是简单 Router 未能捕获已有 headroom。",
    )


def _write_report(summary: dict[str, object]) -> None:
    methods = summary["methods"]
    bootstrap = summary["bootstrap_differences"]
    router = summary["router"]
    per_lambda = summary["per_lambda"]
    decision = summary["decision"]
    captured = router["captured_gap"]
    captured_text = "无法计算" if captured is None else f"{100.0 * captured:.1f}%"

    report = f"""# Gate-0 实验报告：CDFM vs DirectLiNGAM + 简单 Defer Router

文档职责：报告冻结 Gate-0 实验的实际结果、统计不确定性、失败记录和 GO/BORDERLINE/STOP 决策。

适用范围：30×5 个合成数据集上的 CDFM–DirectLiNGAM 必要条件筛选；不外推到真实数据、OOD、Agent 或 MCP。

## 结论

**{decision['label']}**。{decision['reason']}

完整 150/150 个数据集已运行成功。CDFM 使用官方 0.1.0 权重和逐数据集自动阈值；DirectLiNGAM 使用 causal-learn、`measure="pwling"`。所有图均转成 `A[source,target]` 后计算 directed edge F1。

## 六个问题

**Q1. DirectLiNGAM 在线性非高斯区域是否明显超过 CDFM？**  
λ=0 时 `LiNGAM - CDFM = {bootstrap['lingam_minus_cdfm_lambda0']['mean']:.4f}`，95% bootstrap CI `[{bootstrap['lingam_minus_cdfm_lambda0']['ci95_lower']:.4f}, {bootstrap['lingam_minus_cdfm_lambda0']['ci95_upper']:.4f}]`。该差值{'达到' if bootstrap['lingam_minus_cdfm_lambda0']['mean'] >= 0.05 else '未达到'}预设 0.05 门槛。

**Q2. 非线性增加后是否出现 CDFM 反超？**  
λ=1 时 `CDFM - LiNGAM = {bootstrap['cdfm_minus_lingam_lambda1']['mean']:.4f}`，95% CI `[{bootstrap['cdfm_minus_lingam_lambda1']['ci95_lower']:.4f}, {bootstrap['cdfm_minus_lingam_lambda1']['ci95_upper']:.4f}]`。端点均值为：λ=0 CDFM `{per_lambda['0.00']['cdfm']['mean']:.4f}` / LiNGAM `{per_lambda['0.00']['lingam']['mean']:.4f}`；λ=1 CDFM `{per_lambda['1.00']['cdfm']['mean']:.4f}` / LiNGAM `{per_lambda['1.00']['lingam']['mean']:.4f}`。

**Q3. Oracle 比 best single 高多少？**  
Always CDFM `{methods['always_cdfm']['mean']:.4f}`，Always LiNGAM `{methods['always_lingam']['mean']:.4f}`，best single 是 **{summary['best_single_solver']}**、F1 `{methods['best_single']['mean']:.4f}`；Oracle `{methods['oracle']['mean']:.4f}`。Oracle gap 为 `{bootstrap['oracle_minus_best_single']['mean']:.4f}`，95% CI `[{bootstrap['oracle_minus_best_single']['ci95_lower']:.4f}, {bootstrap['oracle_minus_best_single']['ci95_upper']:.4f}]`。

**Q4. 仅使用 X 的简单特征能否预测何时 defer？**  
严格按 base seed 的 5-fold GroupKFold OOF 结果：accuracy `{router['accuracy']:.4f}`，balanced accuracy `{router['balanced_accuracy']:.4f}`，ROC-AUC `{('无法计算' if router['roc_auc'] is None else f"{router['roc_auc']:.4f}")}`。Router 特征只有分布矩和相关性，以及 top-10 相关变量对双向回归的 `nonlinearity_gain`；λ、seed、真图和算法输出没有进入特征矩阵。

**Q5. Hybrid 是否提高最终 directed F1？**  
Hybrid F1 `{methods['hybrid_router']['mean']:.4f}`，相对 best single 改善 `{bootstrap['hybrid_minus_best_single']['mean']:.4f}`，95% CI `[{bootstrap['hybrid_minus_best_single']['ci95_lower']:.4f}, {bootstrap['hybrid_minus_best_single']['ci95_upper']:.4f}]`；捕获 Oracle gap 的 `{captured_text}`。

**Q6. 最终判断？**  
**{decision['label']}**。决策直接使用预注册门槛，没有依据结果修改数据生成器、CDFM threshold 或 Router 特征。

## 解释边界与实现说明

数据生成使用同一 seed 下固定 DAG、边权、RFF 参数和 Laplace 噪声。混合前分别把线性与 RFF 总分量中心化并缩放到单位标准差，使 λ 不被原始数值尺度主导；这会让 λ 成为两个标准化分量的凸混合系数，但由于分量可能相关，它不是严格的方差占比。λ=0 仍是线性非高斯 SEM。生成后统一逐列 z-score，不使用真图进行方法专用预处理。

实现阶段最初用 HGB 计算前 8 个数据集的 X-only 非线性增益，但 CPU 运行时间预计约 25 分钟；在训练 Router、查看 Router 结果之前终止，并从头改用规格允许的简单替代：一维三次样条（7 knots）+ Ridge。该调整只降低计算开销，未改变数据、CDFM/DirectLiNGAM 结果、标签定义或根据结果选参；最终 `meta_features.csv` 的 150 行全部来自样条版本。

CDFM 方向语义按官方源码的 `adjacency_to_edge_list(adj)` 核对为 `adj[source,target]`；DirectLiNGAM 的 `B[target,source]` 在评价前显式转置，并由单元 smoke 验证。SHD 对同一无序节点对的漏边、多边或反向计一次；F1 严格按有向位置比较。

官方来源：[CDFM 仓库](https://github.com/DMIRLAB-Group/CDFM)、[CDFM 论文](https://arxiv.org/abs/2607.11508)。环境与版本见 `results/environment.json`，逐数据集原始预测见 `results/raw_predictions/`，错误日志位置为 `results/errors.jsonl`（若不存在表示无运行异常）。

本轮实验没有验证 OOD robustness，也没有验证 Agent/MCP integration。它只验证 Hybrid CDFM-specialist 研究方向的必要条件。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run() -> None:
    benchmark = pd.read_csv(RESULTS_PATH)
    features = pd.read_csv(FEATURES_PATH)
    frame = benchmark.merge(features, on=["dataset_id", "base_seed", "lambda"], how="inner", validate="one_to_one")
    if len(frame) != 150 or frame["base_seed"].nunique() != 30:
        raise RuntimeError(f"expected 150 datasets and 30 groups, got {len(frame)} and {frame['base_seed'].nunique()}")

    x = frame[FEATURE_COLUMNS].to_numpy(float)
    y = (frame["lingam_f1"] >= frame["cdfm_f1"] + DEFER_MARGIN).astype(int).to_numpy()
    groups = frame["base_seed"].to_numpy(int)
    probabilities = np.zeros(len(frame), dtype=float)
    predictions = np.zeros(len(frame), dtype=int)
    fold_ids = np.zeros(len(frame), dtype=int)
    degenerate_folds: list[int] = []

    splitter = GroupKFold(n_splits=5)
    for fold_id, (train_indices, test_indices) in enumerate(splitter.split(x, y, groups), start=1):
        train_classes = np.unique(y[train_indices])
        fold_ids[test_indices] = fold_id
        if train_classes.size == 1:
            constant = int(train_classes[0])
            probabilities[test_indices] = float(constant)
            predictions[test_indices] = constant
            degenerate_folds.append(fold_id)
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=BOOTSTRAP_SEED,
                    ),
                ),
            ]
        )
        model.fit(x[train_indices], y[train_indices])
        probabilities[test_indices] = model.predict_proba(x[test_indices])[:, 1]
        predictions[test_indices] = (probabilities[test_indices] >= 0.5).astype(int)

    selected_solver = np.where(predictions == 1, "DirectLiNGAM", "CDFM")
    selected_f1 = np.where(predictions == 1, frame["lingam_f1"], frame["cdfm_f1"]).astype(float)
    frame["defer_target"] = y
    frame["router_fold"] = fold_ids
    frame["router_defer_probability"] = probabilities
    frame["router_prediction"] = predictions
    frame["selected_solver"] = selected_solver
    frame["hybrid_selected_f1"] = selected_f1
    router_columns = [
        "dataset_id",
        "base_seed",
        "lambda",
        *FEATURE_COLUMNS,
        "cdfm_f1",
        "lingam_f1",
        "defer_target",
        "router_fold",
        "router_defer_probability",
        "router_prediction",
        "selected_solver",
        "hybrid_selected_f1",
    ]
    frame[router_columns].to_csv(ROUTER_PATH, index=False)

    cdfm = frame["cdfm_f1"].to_numpy(float)
    lingam_values = frame["lingam_f1"].to_numpy(float)
    hybrid = selected_f1
    methods, best_solver = _method_summaries(frame)
    best_single = methods["best_single"]["mean"]
    oracle_gap = methods["oracle"]["mean"] - best_single
    hybrid_improvement = methods["hybrid_router"]["mean"] - best_single
    captured_gap = None if abs(oracle_gap) < 1e-12 else hybrid_improvement / oracle_gap

    lambda0 = frame[np.isclose(frame["lambda"], 0.0)]
    lambda1 = frame[np.isclose(frame["lambda"], 1.0)]
    lambda0_difference = float((lambda0["lingam_f1"] - lambda0["cdfm_f1"]).mean())
    lambda1_difference = float((lambda1["cdfm_f1"] - lambda1["lingam_f1"]).mean())
    decision_label, decision_reason = _gate_decision(
        lambda0_difference,
        lambda1_difference,
        oracle_gap,
        hybrid_improvement,
        captured_gap,
        frame,
    )

    roc_auc = float(roc_auc_score(y, probabilities)) if np.unique(y).size == 2 else None
    environment = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    summary: dict[str, object] = {
        "status": "complete",
        "dataset_count": int(len(frame)),
        "base_seed_count": int(frame["base_seed"].nunique()),
        "data_generation": {
            "n_samples": int(frame["n_samples"].iloc[0]),
            "n_variables": int(frame["n_variables"].iloc[0]),
            "lambda_values": [float(value) for value in sorted(frame["lambda"].unique())],
            "mean_edge_count": float(frame.groupby("base_seed")["edge_count"].first().mean()),
            "mean_total_degree": float(
                2.0 * frame.groupby("base_seed")["edge_count"].first().mean()
                / frame["n_variables"].iloc[0]
            ),
        },
        "best_single_solver": best_solver,
        "methods": methods,
        "per_lambda": _per_lambda_summary(frame),
        "bootstrap_differences": {
            "oracle_minus_best_single": _oracle_gap_bootstrap(cdfm, lingam_values),
            "hybrid_minus_best_single": _hybrid_improvement_bootstrap(hybrid, cdfm, lingam_values),
            "lingam_minus_cdfm_lambda0": _paired_difference_bootstrap(
                lambda0["lingam_f1"].to_numpy(float),
                lambda0["cdfm_f1"].to_numpy(float),
                30,
            ),
            "cdfm_minus_lingam_lambda1": _paired_difference_bootstrap(
                lambda1["cdfm_f1"].to_numpy(float),
                lambda1["lingam_f1"].to_numpy(float),
                31,
            ),
        },
        "router": {
            "features": FEATURE_COLUMNS,
            "target_definition": f"lingam_f1 >= cdfm_f1 + {DEFER_MARGIN}",
            "defer_rate": float(y.mean()),
            "predicted_defer_rate": float(predictions.mean()),
            "accuracy": float(accuracy_score(y, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "roc_auc": roc_auc,
            "captured_gap": None if captured_gap is None else float(captured_gap),
            "group_kfold_splits": 5,
            "degenerate_training_folds": degenerate_folds,
        },
        "runtime": {
            "mean_cdfm_sec": float(frame["cdfm_runtime_sec"].mean()),
            "mean_lingam_sec": float(frame["lingam_runtime_sec"].mean()),
            "total_cdfm_sec": float(frame["cdfm_runtime_sec"].sum()),
            "total_lingam_sec": float(frame["lingam_runtime_sec"].sum()),
        },
        "decision": {
            "label": decision_label,
            "reason": decision_reason,
            "criteria": {
                "lambda0_lingam_minus_cdfm_gte_0_05": lambda0_difference >= 0.05,
                "lambda1_cdfm_minus_lingam_gte_0_05": lambda1_difference >= 0.05,
                "oracle_gap_gte_0_03": oracle_gap >= 0.03,
                "hybrid_improvement_gte_0_015_or_captured_gap_gte_0_50": (
                    hybrid_improvement >= 0.015 or (captured_gap is not None and captured_gap >= 0.50)
                ),
            },
        },
        "environment": environment,
        "bootstrap": {"resamples": N_BOOTSTRAP, "unit": "dataset", "seed": BOOTSTRAP_SEED},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(summary)
    print(
        f"{decision_label}: oracle_gap={oracle_gap:.4f}, "
        f"hybrid_improvement={hybrid_improvement:.4f}, captured_gap={captured_gap}, "
        f"lambda0={lambda0_difference:.4f}, lambda1={lambda1_difference:.4f}"
    )


if __name__ == "__main__":
    run()
