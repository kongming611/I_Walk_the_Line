# Gate-0 实验报告：CDFM vs DirectLiNGAM + 简单 Defer Router

文档职责：报告冻结 Gate-0 实验的实际结果、统计不确定性、失败记录和 GO/BORDERLINE/STOP 决策。

适用范围：30×5 个合成数据集上的 CDFM–DirectLiNGAM 必要条件筛选；不外推到真实数据、OOD、Agent 或 MCP。

## 结论

**GO**。两个端点均存在至少 0.05 F1 的相反优势，oracle headroom 至少 0.03，且简单 Router 达到预设利用门槛。

完整 150/150 个数据集已运行成功。CDFM 使用官方 0.1.0 权重和逐数据集自动阈值；DirectLiNGAM 使用 causal-learn、`measure="pwling"`。所有图均转成 `A[source,target]` 后计算 directed edge F1。

## 六个问题

**Q1. DirectLiNGAM 在线性非高斯区域是否明显超过 CDFM？**  
λ=0 时 `LiNGAM - CDFM = 0.1686`，95% bootstrap CI `[0.1185, 0.2239]`。该差值达到预设 0.05 门槛。

**Q2. 非线性增加后是否出现 CDFM 反超？**  
λ=1 时 `CDFM - LiNGAM = 0.5692`，95% CI `[0.5077, 0.6255]`。端点均值为：λ=0 CDFM `0.8103` / LiNGAM `0.9789`；λ=1 CDFM `0.7165` / LiNGAM `0.1473`。

**Q3. Oracle 比 best single 高多少？**  
Always CDFM `0.7614`，Always LiNGAM `0.5804`，best single 是 **CDFM**、F1 `0.7614`；Oracle `0.8313`。Oracle gap 为 `0.0699`，95% CI `[0.0519, 0.0903]`。

**Q4. 仅使用 X 的简单特征能否预测何时 defer？**  
严格按 base seed 的 5-fold GroupKFold OOF 结果：accuracy `0.8933`，balanced accuracy `0.8964`，ROC-AUC `0.9387`。Router 特征只有分布矩和相关性，以及 top-10 相关变量对双向回归的 `nonlinearity_gain`；λ、seed、真图和算法输出没有进入特征矩阵。

**Q5. Hybrid 是否提高最终 directed F1？**  
Hybrid F1 `0.8245`，相对 best single 改善 `0.0632`，95% CI `[0.0427, 0.0854]`；捕获 Oracle gap 的 `90.4%`。

**Q6. 最终判断？**  
**GO**。决策直接使用预注册门槛，没有依据结果修改数据生成器、CDFM threshold 或 Router 特征。

## 解释边界与实现说明

数据生成使用同一 seed 下固定 DAG、边权、RFF 参数和 Laplace 噪声。混合前分别把线性与 RFF 总分量中心化并缩放到单位标准差，使 λ 不被原始数值尺度主导；这会让 λ 成为两个标准化分量的凸混合系数，但由于分量可能相关，它不是严格的方差占比。λ=0 仍是线性非高斯 SEM。生成后统一逐列 z-score，不使用真图进行方法专用预处理。

实现阶段最初用 HGB 计算前 8 个数据集的 X-only 非线性增益，但 CPU 运行时间预计约 25 分钟；在训练 Router、查看 Router 结果之前终止，并从头改用规格允许的简单替代：一维三次样条（7 knots）+ Ridge。该调整只降低计算开销，未改变数据、CDFM/DirectLiNGAM 结果、标签定义或根据结果选参；最终 `meta_features.csv` 的 150 行全部来自样条版本。

CDFM 方向语义按官方源码的 `adjacency_to_edge_list(adj)` 核对为 `adj[source,target]`；DirectLiNGAM 的 `B[target,source]` 在评价前显式转置，并由单元 smoke 验证。SHD 对同一无序节点对的漏边、多边或反向计一次；F1 严格按有向位置比较。

官方来源：[CDFM 仓库](https://github.com/DMIRLAB-Group/CDFM)、[CDFM 论文](https://arxiv.org/abs/2607.11508)。环境与版本见 `results/environment.json`，逐数据集原始预测见 `results/raw_predictions/`，错误日志位置为 `results/errors.jsonl`（若不存在表示无运行异常）。

本轮实验没有验证 OOD robustness，也没有验证 Agent/MCP integration。它只验证 Hybrid CDFM-specialist 研究方向的必要条件。
