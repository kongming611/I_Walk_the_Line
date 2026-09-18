"""绘制 ground truth、CDFM 与 DirectLiNGAM 的因果结构箭头图。

只读取 `results/raw_predictions/` 中已保存的邻接矩阵，不重新训练、不重新预测。
三张图共用由 ground-truth 拓扑序推导的同一套节点坐标，因此可以直接逐边对比。
绘制原语在 `graph_draw` 中，与 gate1 共用同一套风格。
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graph_draw as gd
from generate_data import BASE_SPEC_DIR, ROOT
from metrics import directed_graph_metrics
from run_benchmark import RAW_PREDICTIONS_DIR, RESULTS_PATH


GRAPHS_DIR = ROOT / "figures" / "graphs"


def _lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:.2f}".replace(".", "p")


def load_predictions(dataset_id: str) -> dict[str, np.ndarray]:
    """读取单个数据集的三套结构与连续分数。"""
    path = RAW_PREDICTIONS_DIR / f"{dataset_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing raw predictions for {dataset_id}: {path}")
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _panels_for(
    truth: np.ndarray,
    cdfm: np.ndarray,
    lingam: np.ndarray,
    cdfm_confidence: np.ndarray,
    lingam_confidence: np.ndarray,
    *,
    include_truth: bool,
) -> list[tuple[str, str, np.ndarray | None, np.ndarray | None, str]]:
    """统一的 (标题, 副标题, 预测矩阵, 连续分数, 颜色) 面板描述。"""
    panels: list[tuple[str, str, np.ndarray | None, np.ndarray | None, str]] = []
    if include_truth:
        panels.append(("Ground truth", gd.edge_count_label(int(truth.sum())), None, None, gd.TRUTH_COLOR))
    panels.append(
        ("CDFM", gd.metric_caption(directed_graph_metrics(cdfm, truth)), cdfm, cdfm_confidence, gd.CDFM_COLOR)
    )
    panels.append(
        (
            "DirectLiNGAM",
            gd.metric_caption(directed_graph_metrics(lingam, truth)),
            lingam,
            lingam_confidence,
            gd.LINGAM_COLOR,
        )
    )
    return panels


def _render(axes, panels, positions, truth: np.ndarray, *, style: str) -> None:
    for axis, (title, subtitle, predicted, confidence, color) in zip(np.atleast_1d(axes), panels):
        if predicted is None:
            edges = gd.truth_edges(truth)
        else:
            edges = gd.edges_for_panel(predicted, truth, confidence, style, color)
        gd.draw_edges(axis, edges, positions, curved=style != "plain")
        gd.draw_nodes(axis, positions)
        gd.finish_panel(axis, positions, title=title, subtitle=subtitle)


def plot_dataset(
    dataset_id: str,
    *,
    style: str = "diff",
    include_truth: bool = True,
    output_path: Path | None = None,
) -> Path:
    """画单个数据集的箭头图：ground truth | CDFM | DirectLiNGAM。"""
    payload = load_predictions(dataset_id)
    base_seed = int(dataset_id.split("_")[1])
    with np.load(BASE_SPEC_DIR / f"seed_{base_seed:02d}.npz") as spec:
        order = np.asarray(spec["topological_order"], dtype=int)

    truth = payload["truth_adjacency"]
    positions = gd.layered_layout(truth, order)
    panels = _panels_for(
        truth,
        payload["cdfm_adjacency"],
        payload["lingam_adjacency_source_target"],
        payload["cdfm_probabilities"],
        payload["lingam_coefficients_target_source"].T,
        include_truth=include_truth,
    )

    width, height = gd.panel_size(positions)
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), height))
    _render(axes, panels, positions, truth, style=style)

    lambda_value = float(dataset_id.split("_lambda_")[1].replace("p", "."))
    fig.suptitle(f"{dataset_id}   (topology seed {base_seed}, lambda={lambda_value:g})", fontsize=13, fontweight="bold")
    fig.legend(handles=gd.legend_handles(style), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))

    output_path = output_path or GRAPHS_DIR / f"{dataset_id}_{style}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_seed_grid(base_seed: int, *, style: str = "diff") -> Path:
    """画一个 seed 在 5 个 lambda 上的对比网格：行=λ，列=CDFM / DirectLiNGAM。"""
    lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
    with np.load(BASE_SPEC_DIR / f"seed_{base_seed:02d}.npz") as spec:
        order = np.asarray(spec["topological_order"], dtype=int)
        truth = np.asarray(spec["adjacency"], dtype=np.int8)
    positions = gd.layered_layout(truth, order)

    width, height = gd.panel_size(positions)
    rows = len(lambdas) + 1
    fig, axes = plt.subplots(rows, 2, figsize=(width * 2 + 0.4, height * rows * 0.80))
    gd.draw_edges(axes[0, 0], gd.truth_edges(truth), positions)
    gd.draw_nodes(axes[0, 0], positions)
    gd.finish_panel(
        axes[0, 0],
        positions,
        title="Ground truth",
        subtitle=gd.edge_count_label(int(truth.sum()), suffix=" (identical for all lambda)"),
    )
    axes[0, 1].axis("off")
    axes[0, 1].legend(handles=gd.legend_handles(style), loc="center", frameon=False, fontsize=10)
    axes[0, 1].set_title("How to read these panels", fontsize=11, fontweight="semibold")

    for row, lambda_value in enumerate(lambdas, start=1):
        payload = load_predictions(f"seed_{base_seed:02d}_lambda_{_lambda_tag(lambda_value)}")
        columns = [
            ("CDFM", payload["cdfm_adjacency"], payload["cdfm_probabilities"], gd.CDFM_COLOR),
            (
                "DirectLiNGAM",
                payload["lingam_adjacency_source_target"],
                payload["lingam_coefficients_target_source"].T,
                gd.LINGAM_COLOR,
            ),
        ]
        for column, (title, predicted, confidence, color) in enumerate(columns):
            axis = axes[row, column]
            edges = gd.edges_for_panel(predicted, truth, confidence, style, color)
            gd.draw_edges(axis, edges, positions, curved=style != "plain")
            gd.draw_nodes(axis, positions)
            gd.finish_panel(
                axis,
                positions,
                title=f"{title}  |  lambda={lambda_value:g}",
                subtitle=gd.metric_caption(directed_graph_metrics(predicted, truth)),
            )

    fig.suptitle(f"seed_{base_seed:02d}: CDFM vs DirectLiNGAM along the linear-nonlinear continuum", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = GRAPHS_DIR / f"seed_{base_seed:02d}_lambda_sweep_{style}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_gallery(paths: list[Path]) -> Path:
    """写一个本地 HTML 索引，方便逐个浏览生成的箭头图。"""
    cards = "\n".join(
        f'<figure><img src="{html.escape(p.relative_to(GRAPHS_DIR).as_posix())}" loading="lazy">'
        f"<figcaption>{html.escape(p.stem)}</figcaption></figure>"
        for p in sorted(paths)
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Gate-0 causal graph gallery</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
h1{{font-size:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}}
figure{{margin:0;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px}}
img{{width:100%;height:auto;display:block}}
figcaption{{font-size:12px;color:#475569;margin-top:8px;font-family:ui-monospace,monospace}}
</style></head><body>
<h1>Gate-0 causal graph gallery ({len(paths)} figures)</h1>
<div class="grid">{cards}</div>
</body></html>
"""
    output_path = GRAPHS_DIR / "index.html"
    output_path.write_text(document, encoding="utf-8")
    return output_path


def available_datasets() -> list[str]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"run run_benchmark.py first; missing {RESULTS_PATH}")
    frame = pd.read_csv(RESULTS_PATH)
    return sorted(frame["dataset_id"].astype(str))


def showcase_datasets() -> list[str]:
    """挑出最能体现两种算法差异的代表数据集：LiNGAM 领先 / CDFM 领先 / 接近平手。"""
    frame = pd.read_csv(RESULTS_PATH)
    frame["margin"] = frame["cdfm_f1"] - frame["lingam_f1"]
    picks = [
        frame.nsmallest(1, "margin")["dataset_id"].iloc[0],
        frame.nlargest(1, "margin")["dataset_id"].iloc[0],
        frame.iloc[(frame["margin"].abs()).argsort()[:1]]["dataset_id"].iloc[0],
    ]
    return list(dict.fromkeys(str(item) for item in picks))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", default=None, help="dataset_id，可重复传入")
    parser.add_argument("--seed", type=int, action="append", default=None, help="画该 seed 的 5 个 lambda 网格")
    parser.add_argument("--all", action="store_true", help="为全部 150 个数据集出图")
    parser.add_argument("--style", choices=["diff", "plain"], default="diff", help="diff=命中/误报/漏报着色，plain=纯预测箭头")
    parser.add_argument("--gallery", action="store_true", help="额外写出 figures/graphs/index.html")
    args = parser.parse_args()

    gd.apply_style(plt)
    written: list[Path] = []

    dataset_ids: list[str] = []
    if args.dataset:
        dataset_ids.extend(args.dataset)
    if args.all:
        dataset_ids.extend(available_datasets())
    if not args.dataset and not args.all and not args.seed:
        dataset_ids.extend(showcase_datasets())
        print(f"no target given; rendering showcase datasets: {dataset_ids}")

    for dataset_id in dataset_ids:
        written.append(plot_dataset(dataset_id, style=args.style))
        print(f"wrote {written[-1]}")

    for base_seed in args.seed or []:
        written.append(plot_seed_grid(base_seed, style=args.style))
        print(f"wrote {written[-1]}")

    if args.gallery:
        print(f"wrote {write_gallery(written)}")

    print(f"done: {len(written)} figures in {GRAPHS_DIR}")


if __name__ == "__main__":
    main()
