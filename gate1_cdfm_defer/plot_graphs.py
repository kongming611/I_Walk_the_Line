"""绘制 Gate-1 三个数据源的因果结构箭头图：ground truth | CDFM | DirectLiNGAM。

覆盖 `results/<source>/raw_predictions/` 下的 synthetic / tuebingen / causal_chamber。
只读取已保存的邻接矩阵，不重新训练、不重新预测。绘制原语来自 gate0 的 `graph_draw`，
与 gate0 的图共用同一套配色、布局与标注风格。
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
GATE0_ROOT = REPO_ROOT / "experiments" / "gate0_cdfm_defer"
sys.path.insert(0, str(GATE0_ROOT))

import graph_draw as gd  # noqa: E402
from metrics import directed_graph_metrics, lingam_target_source_to_source_target  # noqa: E402


GRAPHS_DIR = ROOT / "figures" / "graphs"
BASE_SPEC_DIR = ROOT / "data" / "synthetic" / "base_specs"
SOURCES = {
    "synthetic": ROOT / "results" / "synthetic" / "raw_predictions",
    "tuebingen": ROOT / "results" / "tuebingen" / "raw_predictions",
    "causal_chamber": ROOT / "results" / "causal_chamber" / "raw_predictions",
}
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_NODES_DEFAULT = 40


class DenseGraphSkipped(Exception):
    """节点数超过绘图上限；120 节点的分层箭头图无法保持可读。"""


def _lambda_tag(lambda_value: float) -> str:
    return f"{lambda_value:.2f}".replace(".", "p")


def load_predictions(source: str, dataset_id: str) -> dict[str, np.ndarray]:
    path = SOURCES[source] / f"{dataset_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing raw predictions for {dataset_id}: {path}")
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def available_datasets(source: str) -> list[str]:
    return sorted(path.stem for path in SOURCES[source].glob("*.npz"))


def truth_from_direction(value: str, n_variables: int) -> np.ndarray:
    """Tuebingen 只保存 `true_direction`（如 '0->1'），这里还原成 A[source, target]。"""
    source, target = (int(part) for part in str(value).split("->"))
    truth = np.zeros((n_variables, n_variables), dtype=np.int8)
    truth[source, target] = 1
    return truth


def case_frames(source: str, dataset_id: str) -> dict[str, object]:
    """把一个数据集归一化成统一的 (truth, cdfm, lingam, 连续分数, 拓扑序)。"""
    payload = load_predictions(source, dataset_id)

    if "truth_adjacency" in payload:
        truth = payload["truth_adjacency"]
    elif "true_direction" in payload:
        truth = truth_from_direction(str(payload["true_direction"]), payload["cdfm_adjacency"].shape[0])
    else:
        raise ValueError(f"{dataset_id} has neither truth_adjacency nor true_direction")

    # Tuebingen 没有保存二值化后的 LiNGAM 邻接，按 benchmark 的同一约定现算。
    if "lingam_adjacency_source_target" in payload:
        lingam = payload["lingam_adjacency_source_target"]
    else:
        lingam = lingam_target_source_to_source_target(payload["lingam_coefficients_target_source"])

    # synthetic 保留了生成期拓扑序；其余数据源从 ground truth 现推一个稳定的拓扑序。
    spec_path = BASE_SPEC_DIR / f"{dataset_id.split('_lambda_')[0]}.npz"
    if source == "synthetic" and spec_path.exists():
        with np.load(spec_path) as spec:
            order = np.asarray(spec["topological_order"], dtype=int)
    else:
        order = gd.topological_order(truth)

    return {
        "truth": truth,
        "cdfm": payload["cdfm_adjacency"],
        "lingam": lingam,
        "cdfm_confidence": payload["cdfm_probabilities"],
        "lingam_confidence": payload["lingam_coefficients_target_source"].T,
        "order": order,
    }


def _panels_for(frames: dict[str, object], *, include_truth: bool) -> list[tuple]:
    truth = frames["truth"]
    panels: list[tuple] = []
    if include_truth:
        panels.append(("Ground truth", gd.edge_count_label(int(truth.sum())), None, None, gd.TRUTH_COLOR))
    panels.append(
        (
            "CDFM",
            gd.metric_caption(directed_graph_metrics(frames["cdfm"], truth)),
            frames["cdfm"],
            frames["cdfm_confidence"],
            gd.CDFM_COLOR,
        )
    )
    panels.append(
        (
            "DirectLiNGAM",
            gd.metric_caption(directed_graph_metrics(frames["lingam"], truth)),
            frames["lingam"],
            frames["lingam_confidence"],
            gd.LINGAM_COLOR,
        )
    )
    return panels


def _render(axes, panels, positions, truth: np.ndarray, *, style: str) -> None:
    for axis, (title, subtitle, predicted, confidence, color) in zip(np.atleast_1d(axes), panels):
        edges = gd.truth_edges(truth) if predicted is None else gd.edges_for_panel(
            predicted, truth, confidence, style, color
        )
        gd.draw_edges(axis, edges, positions, curved=style != "plain")
        gd.draw_nodes(axis, positions)
        gd.finish_panel(axis, positions, title=title, subtitle=subtitle)


def plot_case(
    source: str,
    dataset_id: str,
    *,
    style: str = "diff",
    include_truth: bool = True,
    max_nodes: int = MAX_NODES_DEFAULT,
    output_path: Path | None = None,
) -> Path:
    """画单个数据集：ground truth | CDFM | DirectLiNGAM。"""
    frames = case_frames(source, dataset_id)
    truth = frames["truth"]
    if truth.shape[0] > max_nodes:
        raise DenseGraphSkipped(f"{dataset_id} has {truth.shape[0]} nodes (limit {max_nodes})")

    positions = gd.layered_layout(truth, frames["order"])
    panels = _panels_for(frames, include_truth=include_truth)

    width, height = gd.panel_size(positions)
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), height))
    _render(axes, panels, positions, truth, style=style)

    cdfm_f1 = directed_graph_metrics(frames["cdfm"], truth)["f1"]
    lingam_f1 = directed_graph_metrics(frames["lingam"], truth)["f1"]
    fig.suptitle(
        f"{dataset_id}   [{source}]   (CDFM F1={cdfm_f1:.2f} vs DirectLiNGAM F1={lingam_f1:.2f})",
        fontsize=13,
        fontweight="bold",
    )
    fig.legend(handles=gd.legend_handles(style), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))

    output_path = output_path or GRAPHS_DIR / source / f"{dataset_id}_{style}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sweep(
    source: str,
    prefix: str,
    *,
    style: str = "diff",
    max_nodes: int = MAX_NODES_DEFAULT,
    output_path: Path | None = None,
) -> Path:
    """画同一个数据集在 5 个 lambda 上的对比网格：行=λ，列=CDFM / DirectLiNGAM。"""
    datasets = [f"{prefix}_lambda_{_lambda_tag(value)}" for value in LAMBDAS]
    missing = [item for item in datasets if not (SOURCES[source] / f"{item}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{prefix} is not a lambda sweep; missing {missing}")

    first = case_frames(source, datasets[0])
    truth = first["truth"]
    if truth.shape[0] > max_nodes:
        raise DenseGraphSkipped(f"{prefix} has {truth.shape[0]} nodes (limit {max_nodes})")
    positions = gd.layered_layout(truth, first["order"])

    width, height = gd.panel_size(positions)
    rows = len(LAMBDAS) + 1
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

    for row, (lambda_value, dataset_id) in enumerate(zip(LAMBDAS, datasets), start=1):
        frames = case_frames(source, dataset_id)
        columns = [
            ("CDFM", frames["cdfm"], frames["cdfm_confidence"], gd.CDFM_COLOR),
            ("DirectLiNGAM", frames["lingam"], frames["lingam_confidence"], gd.LINGAM_COLOR),
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

    fig.suptitle(f"{prefix} [{source}]: CDFM vs DirectLiNGAM along the lambda continuum", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = output_path or GRAPHS_DIR / source / f"{prefix}_lambda_sweep_{style}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return output_path


def showcase(source: str, *, max_nodes: int) -> list[str]:
    """每个数据源挑 CDFM 领先与 LiNGAM 领先各一个，作为默认出图对象。"""
    scored: list[tuple[str, float]] = []
    for dataset_id in available_datasets(source):
        try:
            frames = case_frames(source, dataset_id)
        except (ValueError, KeyError):
            continue
        truth = frames["truth"]
        if truth.shape[0] > max_nodes:
            continue
        margin = directed_graph_metrics(frames["cdfm"], truth)["f1"] - directed_graph_metrics(frames["lingam"], truth)["f1"]
        scored.append((dataset_id, margin))
    if not scored:
        return []
    scored.sort(key=lambda item: item[1])
    return list(dict.fromkeys([scored[0][0], scored[-1][0]]))


def sources_containing(dataset_id: str, sources: list[str]) -> list[str]:
    """把 dataset_id 解析到真正拥有它的数据源，避免在错误的数据源里找文件。"""
    return [source for source in sources if (SOURCES[source] / f"{dataset_id}.npz").exists()]


def write_gallery(paths: list[Path]) -> Path:
    cards = "\n".join(
        f'<figure><img src="{html.escape(p.relative_to(GRAPHS_DIR).as_posix())}" loading="lazy">'
        f"<figcaption>{html.escape(p.relative_to(GRAPHS_DIR).with_suffix('').as_posix())}</figcaption></figure>"
        for p in sorted(paths)
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Gate-1 causal graph gallery</title>
<style>
body{{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:24px;background:#f8fafc;color:#0f172a}}
h1{{font-size:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}}
figure{{margin:0;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:10px}}
img{{width:100%;height:auto;display:block}}
figcaption{{font-size:12px;color:#475569;margin-top:8px;font-family:ui-monospace,monospace}}
</style></head><body>
<h1>Gate-1 causal graph gallery ({len(paths)} figures)</h1>
<div class="grid">{cards}</div>
</body></html>
"""
    output_path = GRAPHS_DIR / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), action="append", default=None, help="数据源，可重复传入")
    parser.add_argument("--dataset", action="append", default=None, help="dataset_id，可重复传入（需配合 --source）")
    parser.add_argument("--sweep", action="append", default=None, help="形如 router_ood_tanh_seed_1000 的 lambda 前缀")
    parser.add_argument("--all", action="store_true", help="为选定数据源的全部数据集出图")
    parser.add_argument("--list", action="store_true", help="只列出可用 dataset_id，不出图")
    parser.add_argument("--style", choices=["diff", "plain"], default="diff")
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES_DEFAULT, help="节点数上限，超过则跳过")
    parser.add_argument("--gallery", action="store_true", help="额外写出 figures/graphs/index.html")
    args = parser.parse_args()

    gd.apply_style(plt)
    sources = args.source or sorted(SOURCES)

    if args.list:
        for source in sources:
            ids = available_datasets(source)
            print(f"=== {source}: {len(ids)} datasets")
            for dataset_id in ids:
                print(f"    {dataset_id}")
        return

    written: list[Path] = []
    skipped: list[str] = []

    def render(source: str, dataset_id: str) -> None:
        try:
            path = plot_case(source, dataset_id, style=args.style, max_nodes=args.max_nodes)
        except DenseGraphSkipped as exc:
            skipped.append(str(exc))
            print(f"skip {exc} -- raise --max-nodes to force")
            return
        written.append(path)
        print(f"wrote {path}")

    for dataset_id in args.dataset or []:
        owners = sources_containing(dataset_id, sources)
        if not owners:
            parser.error(f"unknown dataset_id {dataset_id!r} in sources {sources}")
        for source in owners:
            render(source, dataset_id)

    for prefix in args.sweep or []:
        owners = sources_containing(f"{prefix}_lambda_{_lambda_tag(LAMBDAS[0])}", sources)
        if not owners:
            parser.error(f"unknown lambda sweep prefix {prefix!r} in sources {sources}")
        for source in owners:
            path = plot_sweep(source, prefix, style=args.style, max_nodes=args.max_nodes)
            written.append(path)
            print(f"wrote {path}")
    if args.all:
        for source in sources:
            for dataset_id in available_datasets(source):
                render(source, dataset_id)
    if not args.dataset and not args.sweep and not args.all:
        for source in sources:
            picks = showcase(source, max_nodes=args.max_nodes)
            print(f"{source}: rendering showcase datasets {picks}")
            for dataset_id in picks:
                render(source, dataset_id)

    if args.gallery:
        print(f"wrote {write_gallery(written)}")

    print(f"done: {len(written)} figures in {GRAPHS_DIR}" + (f"; {len(skipped)} skipped as too dense" if skipped else ""))


if __name__ == "__main__":
    main()
