"""gate0 与 gate1 共用的因果结构箭头图绘制原语。

只依赖 numpy 与 matplotlib；不导入 torch / cdfm / causal-learn，因此可以离线单独运行。
gate1 通过 `sys.path` 引入本模块（与它引入 `metrics`、`meta_features` 的方式一致），
以保证两个 Gate 的图在配色、布局和标注上完全同风格。
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# 与 plots.py 保持同一套方法配色。
CDFM_COLOR = "#2563eb"
LINGAM_COLOR = "#dc2626"
TRUTH_COLOR = "#111827"
TRUE_POSITIVE_COLOR = "#059669"
FALSE_POSITIVE_COLOR = "#dc2626"
FALSE_NEGATIVE_COLOR = "#9ca3af"
PANEL_BACKGROUND = "#ffffff"
NODE_FILL = "#f8fafc"
NODE_TEXT_COLOR = "#0f172a"

NODE_SIZE = 900
NODE_FONTSIZE = 9
ARROW_SCALE = 13
MIN_EDGE_WIDTH = 1.4
MAX_EDGE_WIDTH = 4.0


def apply_style(plt) -> None:
    """统一两张 Gate 图表的 rcParams。"""
    plt.rcParams.update(
        {
            "figure.facecolor": PANEL_BACKGROUND,
            "axes.facecolor": PANEL_BACKGROUND,
            "font.size": 10,
            "font.family": ["DejaVu Sans", "Microsoft YaHei", "SimHei"],
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
        }
    )


def topological_order(adjacency: np.ndarray) -> np.ndarray:
    """Kahn 拓扑排序，同层按节点下标升序打破平局，结果可复现。

    没有保存生成期拓扑序的数据集（Tuebingen、CausalChamber）用它兜底。
    """
    adjacency = np.asarray(adjacency)
    n_variables = adjacency.shape[0]
    indegree = adjacency.sum(axis=0).astype(int)
    resolved = np.zeros(n_variables, dtype=bool)
    order: list[int] = []

    while len(order) < n_variables:
        ready = [node for node in range(n_variables) if indegree[node] == 0 and not resolved[node]]
        if not ready:
            raise ValueError("adjacency contains a cycle; cannot build a topological order")
        node = min(ready)
        resolved[node] = True
        order.append(node)
        indegree -= adjacency[node]
    return np.asarray(order, dtype=int)


def _sweep(
    layers: list[list[int]],
    adjacency: np.ndarray,
    position: dict[int, int],
    *,
    forward: bool,
) -> tuple[list[list[int]], dict[int, int]]:
    """沿一个方向按邻居的平均位置重排每层（barycenter 启发式，减少边交叉）。"""
    new_layers = [list(layer) for layer in layers]
    new_position = dict(position)
    indices = range(len(layers)) if forward else range(len(layers) - 1, -1, -1)

    for layer_index in indices:
        if forward and layer_index == 0:
            continue
        if not forward and layer_index == len(layers) - 1:
            continue
        scored: list[tuple[float, int]] = []
        for node in layers[layer_index]:
            neighbours = np.flatnonzero(adjacency[:, node] if forward else adjacency[node, :])
            references = [position[parent] for parent in neighbours if parent in position]
            key = float(np.mean(references)) if references else float(position[node])
            scored.append((key, node))
        scored.sort()
        new_layers[layer_index] = [node for _, node in scored]
        for index, node in enumerate(new_layers[layer_index]):
            new_position[node] = index
    return new_layers, new_position


def layered_layout(
    adjacency: np.ndarray,
    order: np.ndarray,
    *,
    sweep_iterations: int = 6,
) -> dict[int, tuple[float, float]]:
    """按最长路径分层，再用 barycenter 启发式减少交叉，得到节点坐标。

    分层保证每条边都从低层指向高层，因此不存在同层内互相缠绕的箭头；
    同一套坐标供 ground truth 与各方法共用，可以直接逐边对比。
    """
    adjacency = np.asarray(adjacency)
    n_variables = adjacency.shape[0]
    depth = np.zeros(n_variables, dtype=int)
    for node in order:
        parents = np.flatnonzero(adjacency[:, node])
        if parents.size:
            depth[node] = int(depth[parents].max()) + 1

    layers = [[int(node) for node in order if depth[node] == layer] for layer in np.unique(depth)]
    position = {node: index for layer in layers for index, node in enumerate(layer)}
    for iteration in range(sweep_iterations):
        layers, position = _sweep(layers, adjacency, position, forward=iteration % 2 == 0)

    positions: dict[int, tuple[float, float]] = {}
    for layer_index, layer in enumerate(layers):
        for index, node in enumerate(layer):
            offset = index - (len(layer) - 1) / 2.0
            positions[node] = (float(layer_index), float(offset))
    return positions


def _spans(positions: dict[int, tuple[float, float]]) -> tuple[float, float]:
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    return max(xs) - min(xs), max(ys) - min(ys)


def _padding(span_x: float, span_y: float) -> tuple[float, float]:
    """节点的像素尺寸固定，所以留白按数据跨度取比例，而不是固定数据单位。"""
    return 0.16 * span_x + 0.05, 0.20 * span_y + 0.28


def panel_size(positions: dict[int, tuple[float, float]]) -> tuple[float, float]:
    """按图的数据跨度选面板尺寸，避免窄图被拉成细缝、宽图被压扁。"""
    span_x, span_y = _spans(positions)
    width = float(np.clip(1.30 * (span_x + 1.6), 2.9, 7.0))
    height = float(np.clip(1.05 * (span_y + 1.6), 2.6, 6.2))
    return width, height


def _edge_width(value: float, maximum: float) -> float:
    if maximum <= 0:
        return MIN_EDGE_WIDTH
    scaled = MIN_EDGE_WIDTH + (MAX_EDGE_WIDTH - MIN_EDGE_WIDTH) * (abs(value) / maximum)
    return float(np.clip(scaled, MIN_EDGE_WIDTH, MAX_EDGE_WIDTH))


def draw_edges(
    axis,
    edges: list[tuple[int, int, str, float]],
    positions: dict[int, tuple[float, float]],
    *,
    curved: bool = True,
) -> None:
    """edges 为 (source, target, color, width) 列表。"""
    for source, target, color, width in edges:
        axis.add_patch(
            FancyArrowPatch(
                positions[source],
                positions[target],
                arrowstyle="-|>",
                mutation_scale=ARROW_SCALE,
                linewidth=width,
                color=color,
                shrinkA=14.0,
                shrinkB=16.0,
                connectionstyle="arc3,rad=0.12" if curved else "arc3,rad=0.0",
                linestyle="--" if color == FALSE_NEGATIVE_COLOR else "-",
                zorder=2,
            )
        )


def draw_nodes(axis, positions: dict[int, tuple[float, float]], labels: dict[int, str] | None = None) -> None:
    nodes = sorted(positions)
    if labels is None:
        labels = {node: f"X{node}" for node in nodes}
    axis.scatter(
        [positions[node][0] for node in nodes],
        [positions[node][1] for node in nodes],
        s=NODE_SIZE,
        c=NODE_FILL,
        edgecolors=TRUTH_COLOR,
        linewidths=1.6,
        zorder=3,
    )
    for node in nodes:
        axis.annotate(
            labels[node],
            positions[node],
            ha="center",
            va="center",
            fontsize=NODE_FONTSIZE,
            fontweight="semibold",
            color=NODE_TEXT_COLOR,
            zorder=4,
        )


def finish_panel(axis, positions: dict[int, tuple[float, float]], *, title: str, subtitle: str) -> None:
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    pad_x, pad_y = _padding(*_spans(positions))
    axis.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    axis.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    axis.set_aspect("auto")
    axis.axis("off")
    axis.set_title(f"{title}\n{subtitle}", linespacing=1.6, pad=10, fontsize=10)


def edge_count_label(count: int, *, suffix: str = "") -> str:
    """`1 directed edge` / `7 directed edges`；Tuebingen 这种二元图会用到单数。"""
    noun = "edge" if count == 1 else "edges"
    return f"{count} directed {noun}{suffix}"


def metric_caption(metrics: dict[str, float | int]) -> str:
    """分两行，否则相邻面板的标题会互相碰撞。"""
    return (
        f"F1={metrics['f1']:.2f}   P={metrics['precision']:.2f}   R={metrics['recall']:.2f}\n"
        f"TP/FP/FN = {metrics['tp']}/{metrics['fp']}/{metrics['fn']}"
    )


def reconcile_edges(
    predicted: np.ndarray,
    truth: np.ndarray,
    confidence: np.ndarray | None,
) -> list[tuple[int, int, str, float]]:
    """把预测边分成命中/误报/漏报三类，供着色使用。"""
    predicted = (predicted != 0).astype(np.int8)
    truth = (truth != 0).astype(np.int8)
    if confidence is None:
        confidence = predicted.astype(float)
    maximum = float(np.abs(confidence).max()) if confidence.size else 0.0

    edges: list[tuple[int, int, str, float]] = []
    n_variables = truth.shape[0]
    for source in range(n_variables):
        for target in range(n_variables):
            if source == target:
                continue
            if predicted[source, target] and truth[source, target]:
                edges.append((source, target, TRUE_POSITIVE_COLOR, _edge_width(confidence[source, target], maximum)))
            elif predicted[source, target]:
                edges.append((source, target, FALSE_POSITIVE_COLOR, _edge_width(confidence[source, target], maximum)))
            elif truth[source, target]:
                edges.append((source, target, FALSE_NEGATIVE_COLOR, MIN_EDGE_WIDTH))
    return edges


def edges_for_panel(
    predicted: np.ndarray,
    truth: np.ndarray,
    confidence: np.ndarray | None,
    style: str,
    color: str,
) -> list[tuple[int, int, str, float]]:
    """diff 风格按命中/误报/漏报着色；plain 风格只画该方法的预测边。"""
    if style != "plain":
        return reconcile_edges(predicted, truth, confidence)

    maximum = float(np.abs(confidence).max()) if confidence is not None and confidence.size else 0.0
    edges: list[tuple[int, int, str, float]] = []
    n_variables = predicted.shape[0]
    for source in range(n_variables):
        for target in range(n_variables):
            if source == target or not predicted[source, target]:
                continue
            width = _edge_width(confidence[source, target], maximum) if confidence is not None else MIN_EDGE_WIDTH
            edges.append((source, target, color, width))
    return edges


def truth_edges(truth: np.ndarray) -> list[tuple[int, int, str, float]]:
    return [(int(s), int(t), TRUTH_COLOR, 2.0) for s, t in zip(*np.nonzero(truth))]


def legend_handles(style: str) -> list[Line2D]:
    if style == "plain":
        return [
            Line2D([], [], color=TRUTH_COLOR, linewidth=2.0, label="ground truth edge"),
            Line2D([], [], color=CDFM_COLOR, linewidth=2.0, label="CDFM edge"),
            Line2D([], [], color=LINGAM_COLOR, linewidth=2.0, label="DirectLiNGAM edge"),
        ]
    return [
        Line2D([], [], color=TRUE_POSITIVE_COLOR, linewidth=2.4, label="true positive"),
        Line2D([], [], color=FALSE_POSITIVE_COLOR, linewidth=2.4, label="false positive (extra arrow)"),
        Line2D([], [], color=FALSE_NEGATIVE_COLOR, linewidth=2.0, linestyle="--", label="false negative (missed arrow)"),
    ]
