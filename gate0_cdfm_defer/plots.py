"""生成 Gate-0 规格限定的三张核心图。"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_data import ROOT
from run_benchmark import RESULTS_PATH
from run_router import ROUTER_PATH, SUMMARY_PATH


FIGURES_DIR = ROOT / "figures"
PLOT_SEED = 20260916
PLOT_BOOTSTRAP = 2000


def _mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boot = np.empty(PLOT_BOOTSTRAP, dtype=float)
    for index in range(PLOT_BOOTSTRAP):
        boot[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    lower, upper = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper)


def plot_f1_vs_lambda(frame: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for method_index, (column, label, color, marker) in enumerate(
        [
            ("cdfm_f1", "CDFM", "#2563eb", "o"),
            ("lingam_f1", "DirectLiNGAM", "#dc2626", "s"),
        ]
    ):
        xs: list[float] = []
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for lambda_index, (lambda_value, group) in enumerate(frame.groupby("lambda", sort=True)):
            mean, lower, upper = _mean_ci(
                group[column].to_numpy(float),
                PLOT_SEED + method_index * 10 + lambda_index,
            )
            xs.append(float(lambda_value))
            means.append(mean)
            lowers.append(lower)
            uppers.append(upper)
        means_array = np.asarray(means)
        axis.plot(xs, means, marker=marker, linewidth=2.2, label=label, color=color)
        axis.fill_between(xs, lowers, uppers, alpha=0.18, color=color)
    axis.set_xlabel("Nonlinearity mixing coefficient λ")
    axis.set_ylabel("Directed edge F1")
    axis.set_xticks(sorted(frame["lambda"].unique()))
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("CDFM vs DirectLiNGAM across the linear–nonlinear continuum")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "f1_vs_lambda.png", dpi=180)
    plt.close(fig)


def plot_hybrid_vs_oracle(summary: dict[str, object]) -> None:
    labels = ["Always\nCDFM", "Always\nLiNGAM", "Best\nSingle", "Hybrid\nRouter", "Oracle"]
    keys = ["always_cdfm", "always_lingam", "best_single", "hybrid_router", "oracle"]
    methods = summary["methods"]
    means = np.array([methods[key]["mean"] for key in keys], dtype=float)
    lower = np.array([methods[key]["ci95_lower"] for key in keys], dtype=float)
    upper = np.array([methods[key]["ci95_upper"] for key in keys], dtype=float)
    colors = ["#2563eb", "#dc2626", "#64748b", "#7c3aed", "#059669"]
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    positions = np.arange(len(labels))
    axis.bar(positions, means, color=colors, width=0.68)
    axis.errorbar(positions, means, yerr=[means - lower, upper - means], fmt="none", ecolor="#111827", capsize=4)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean directed edge F1")
    axis.set_ylim(0.0, min(1.0, max(upper) + 0.10))
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Realized hybrid performance and oracle headroom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hybrid_vs_oracle.png", dpi=180)
    plt.close(fig)


def plot_router_detectability(router: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    colors = np.where(router["defer_target"].to_numpy(int) == 1, "#dc2626", "#2563eb")
    labels_seen: set[int] = set()
    for index, row in router.iterrows():
        target = int(row["defer_target"])
        label = None
        if target not in labels_seen:
            label = "Actual: defer to LiNGAM" if target == 1 else "Actual: use CDFM"
            labels_seen.add(target)
        axis.scatter(
            row["nonlinearity_gain_mean"],
            row["router_defer_probability"],
            c=colors[index],
            s=28,
            alpha=0.70,
            edgecolors="none",
            label=label,
        )
    axis.axhline(0.5, color="#111827", linestyle="--", linewidth=1.1, label="Router threshold")
    axis.set_xlabel("Mean nonlinear-regression gain (X only)")
    axis.set_ylabel("OOF probability of deferring to DirectLiNGAM")
    axis.set_ylim(-0.03, 1.03)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="best")
    axis.set_title("Router detectability without mechanism labels")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "router_detectability.png", dpi=180)
    plt.close(fig)


def run() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = pd.read_csv(RESULTS_PATH)
    router = pd.read_csv(ROUTER_PATH).reset_index(drop=True)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    plot_f1_vs_lambda(benchmark)
    plot_hybrid_vs_oracle(summary)
    plot_router_detectability(router)
    print(f"Wrote 3 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    run()
