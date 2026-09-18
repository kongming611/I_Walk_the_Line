# Gate-0：CDFM–DirectLiNGAM Defer 可行性实验

文档职责：冻结并说明 Gate-0 的数据生成、算法调用、评价、Router、bootstrap 与复现实验入口。

适用范围：仅用于 `experiments/gate0_cdfm_defer/` 下的离线合成数据研究实验；不修改或验证 CausalAgent 生产代码、MCP、Agent 路由、真实数据或 OOD 能力。

## 冻结协议

- 30 个 base seed，每个 seed 生成一个 D=10、总期望度约为 2 的稀疏 ER 风格 DAG。
- 每个 seed 固定拓扑、边权、RFF 参数与 Laplace 外生噪声；对同一个 seed 生成 `lambda ∈ {0, 0.25, 0.5, 0.75, 1}` 共 5 个 N=1000 数据集。
- `lambda=0` 是线性、非高斯、无环 SEM；非线性项是每条边 K=30 的随机 Fourier features。线性项和 RFF 项在混合前分别中心化并按标准差缩放。
- 生成后对每列 z-score；同一个标准化 X 同时交给 CDFM 与 DirectLiNGAM，不做方法专用预处理。
- CDFM 使用 `CDFM.from_pretrained("DMIRLAB/CDFM")` 和 `model.predict(X)` 的官方自动阈值，不根据本实验调阈值。
- DirectLiNGAM 使用 causal-learn 的 `DirectLiNGAM(measure="pwling")`；其 `B[target, source]` 显式转置为统一的 `A[source, target]`。
- 主要指标是 directed edge F1；同时记录 precision、recall、SHD、运行时间、邻接矩阵、CDFM probabilities 与 threshold。
- Router 特征仅来自 X：绝对偏度、绝对 excess kurtosis、绝对 Pearson/Spearman 相关，以及 top-10 相关变量对双向 3-fold CV 的非线性回归增益 median/mean/max。非线性回归器固定为一维三次样条（7 knots）+ Ridge；它属于预定义允许的简单非线性回归器，并避免 CPU-only 环境下 HGB 的不必要开销。
- Router 是 fold 内标准化的 balanced logistic regression；5-fold `GroupKFold` 以 base seed 分组。标签仅定义为 `LiNGAM_F1 >= CDFM_F1 + 0.03`，不会作为输入特征。
- 统计不确定性使用固定随机种子的 2000 次 dataset-level bootstrap。

该协议在运行完整 benchmark 前写定；不得依据结果回头修改生成机制或模型阈值。

## 环境

推荐 Python 3.10+。本次冻结依赖见 `requirements.txt`。CDFM 自动检测设备；当前脚本显式将检测结果传给官方加载接口。无 CUDA 时使用 CPU。

```powershell
python -m pip install -r experiments/gate0_cdfm_defer/requirements.txt
```

## 运行

从仓库根目录依次执行：

```powershell
python experiments/gate0_cdfm_defer/generate_data.py
python experiments/gate0_cdfm_defer/test_gate0.py
python experiments/gate0_cdfm_defer/run_benchmark.py
python experiments/gate0_cdfm_defer/meta_features.py
python experiments/gate0_cdfm_defer/run_router.py
python experiments/gate0_cdfm_defer/plots.py
python experiments/gate0_cdfm_defer/validate_results.py
```

### 可视化因果结构箭头图（可选）

`plot_graphs.py` 只读取 `results/raw_predictions/` 中已保存的邻接矩阵，不重新拟合模型，因此可以在 benchmark 完成后随时单独运行。默认渲染三个代表数据集；`--dataset` 指定单个数据集，`--seed` 画该 seed 的 5 个 lambda 对比网格，`--all` 为全部 150 个数据集出图，`--gallery` 额外写出 `figures/graphs/index.html` 索引页。

```powershell
python experiments/gate0_cdfm_defer/plot_graphs.py --dataset seed_06_lambda_1p00
python experiments/gate0_cdfm_defer/plot_graphs.py --seed 07 --gallery
```

`--style diff`（默认）按 true positive / false positive / false negative 着色，直接暴露多画的箭头和漏掉的箭头；`--style plain` 只画各方法自己的预测箭头。两种风格下 ground truth、CDFM 与 DirectLiNGAM 都共用同一套节点坐标，可以逐边对比。

绘制原语集中在 `graph_draw.py`（只依赖 numpy 与 matplotlib，不导入 torch / cdfm / causal-learn）。Gate-1 通过 `sys.path` 复用同一模块，因此两个 Gate 的箭头图配色、分层布局与标注完全一致；改动绘图风格只需要改这一处。节点分层按最长路径计算，再用 barycenter 启发式减少边的交叉。

`run_benchmark.py` 会在每个数据集后原子更新 CSV，支持重复执行时跳过已成功的数据集。若真实算法失败，异常会写入 `results/errors.jsonl`，不会生成伪结果。最终数值、图和决策写入 `results/summary.json`、`figures/` 与 `REPORT.md`。

## 方向约定

整个实验统一使用：

```text
A[source, target] = 1  表示 source -> target
```

CDFM 官方 `adjacency_to_edge_list` 直接把 `adj[i,j]` 解释为 `i -> j`。causal-learn DirectLiNGAM 的 `adjacency_matrix_[target,source]` 是结构系数，因此转换为 `(abs(B).T > 1e-8)`。`test_gate0.py` 用手工矩阵和真实小型 LiNGAM smoke 同时检查该转换。

## 输出边界

- `data/manifest.csv`：150 个冻结数据集索引。
- `data/base_specs/`：每个 seed 的拓扑、权重、噪声与 RFF 参数。
- `data/datasets/`：标准化 X 与 ground-truth adjacency。
- `results/raw_predictions/`：逐数据集原始预测矩阵。
- `results/per_dataset_results.csv`：逐数据集指标与运行信息。
- `results/router_predictions.csv`：严格 OOF Router 预测。
- `results/summary.json`：汇总、bootstrap CI、版本和 Gate 决策。
- `figures/`：规格要求的三张核心图。
- `figures/graphs/`：`plot_graphs.py` 生成的因果结构箭头图，不参与 Gate 判定。
