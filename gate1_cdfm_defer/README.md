# Gate-1：CDFM–DirectLiNGAM External Validity

文档职责：记录 Gate-1 外部效度实验的冻结边界、执行顺序、数据来源、评价口径和复现入口。

适用范围：只覆盖 CDFM、DirectLiNGAM 与 Gate-0 冻结 Router 在 synthetic OOD、CDFM scale OOD、Tübingen 和 Causal Chamber 上的外部验证；不修改 Gate-0、生产 Agent 或 MCP。

## 强制执行顺序

1. 用 Gate-0 全部 150 个数据集的已保存 meta-features 和原标签训练最终 Router。
2. 保存 `frozen/router.joblib` 与 `frozen/freeze_manifest.json`。冻结脚本没有覆盖模式；重复调用只允许验证现有冻结物。
3. 冻结完成后，才允许生成 Gate-1 synthetic 数据、读取 real benchmark label 或运行两个因果发现算法。
4. Gate-1 全部 Router 推理只能加载冻结的 scaler + LogisticRegression；不得 `fit`、调整 0.5 分类阈值或改变 Gate-0 feature pipeline。

## 冻结 Router

```powershell
python experiments/gate1_cdfm_defer/freeze_router.py
python experiments/gate1_cdfm_defer/freeze_router.py --verify
```

Router 输入固定为 Gate-0 的七个 X-only 特征。`lambda`、mechanism、dataset ID、ground truth、CDFM/DirectLiNGAM 输出与 F1 均不得进入模型输入；`base_seed` 只用于 Gate-0 OOF 审计，不进入最终模型。

## Synthetic 执行

```powershell
python experiments/gate1_cdfm_defer/generate_synthetic.py
python experiments/gate1_cdfm_defer/run_synthetic_benchmark.py
```

## Real benchmark 数据准备

```powershell
python experiments/gate1_cdfm_defer/prepare_real_data.py
```

脚本下载 Tübingen 官方 `pairs_1.0.zip`，固定前 100 对并排除 52–55、71；同时从 CDFM 官方仓库固定 commit 下载其 Causal Chamber A1/A2 准备数据。所有输入均保存 URL 与 SHA-256，且在进入模型前验证 95 个 strict-bivariate pair、A1/A2 行数、变量名和 39-edge 真图。

```powershell
# 先运行 A1 sanity gate；只有通过后 runner 才允许 A2
python experiments/gate1_cdfm_defer/run_causal_chamber_benchmark.py --stage a1
python experiments/gate1_cdfm_defer/run_causal_chamber_benchmark.py --stage a2

# strict-95；失败或平局均保留在官方加权分母中
python experiments/gate1_cdfm_defer/run_tuebingen_benchmark.py
```

Causal Chamber A2 的正式可执行聚合口径固定为当前官方 notebook 的
`20` 个逐环境自动阈值邻接图投票、`vote >= 5/20`。同时另存论文文字所述
probability-average 和 notebook 中的 `mean(logits) > 0.7`，只用于复现差异核验，
不得事后择优替换正式口径。逐环境 Directed F1 是 Gate-1 新增分析，不冒充论文已报告结果。

A2 强干预可能产生常数列。算法仍使用完整官方矩阵；若冻结 Gate-0 特征提取器
因常数列拒绝输入，不得改变、插补或重训特征，而是记录 Router 技术失败，并按
“CDFM 为默认 generalist、只有有效 Router 决策才 defer”的语义 fail closed 到 CDFM。

## 预注册 Gate-1 分层

- `router_ood_tanh`：15 个新 seed × 5 个 λ；Gate-0 RFF 分量替换为逐边 additive `tanh`，噪声仍为单位方差 Laplace。
- `router_ood_student_t`：15 个新 seed × 5 个 λ；仍使用 Gate-0 RFF，噪声替换为标准化 Student-t(df=3)。
- `scale_ood_n40_d10`：D=10、N=40、λ∈{0,1}、10 个新 seed；RFF + Laplace。
- `scale_ood_d120_n1000`：D=120、N=1000、λ∈{0,1}、10 个新 seed；RFF + Laplace，ER 总期望度仍约为 2。
- `tuebingen_strict95`：官方标准 95 个 strict-bivariate cause-effect pairs，使用官方 metadata 权重和 weighted causal-direction accuracy。
- `causal_chamber_a1` / `causal_chamber_a2`：以 CDFM 官方论文/代码所用版本和评价协议为准；A1 先复现论文 sanity check，A2 同时保存逐 environment 结果。

每个分层分别报告 Always CDFM、Always LiNGAM、Best Single、Frozen Hybrid、Oracle、Oracle gap、Hybrid improvement、captured gap 和算法胜率。Synthetic/Causal Chamber 使用 directed F1；Tübingen 使用官方 weighted causal-direction accuracy。

最终 Gate 判定首先看 external/OOD Oracle headroom。若 complementarity 仍存在而 Frozen Router 失败，结论必须写成 `complementarity survives, routing does not generalize`，不得误判为 portfolio STOP。

Synthetic 的 seed 空间、mechanism 与噪声规则在 `generate_synthetic.py` 中固定。tanh 组使用逐边 `w·tanh(s·x+b)` 的 additive mechanism；Student-t 组将 df=3 噪声除以 `sqrt(3)` 以得到单位方差。线性与非线性总分量仍分别中心化并缩放后按 λ 混合，最后统一逐列 z-score。

Tübingen 的 forced-direction decoder 同样在运行前固定：CDFM 比较两个 off-diagonal probability，较大者决定方向；完全相等视为无决定并按错误计入加权分母。DirectLiNGAM 使用其 `causal_order_` 的首个变量作为 cause；失败或非法顺序按错误计入分母。不得根据 aggregate sanity 结果在 causal order、稀疏系数或阈值邻接之间切换。

## 可视化因果结构箭头图（可选）

`plot_graphs.py` 为三个数据源输出 ground truth | CDFM | DirectLiNGAM 的箭头对比图。它只读取 `results/<source>/raw_predictions/` 中已保存的矩阵，不重新拟合模型；绘制原语来自 Gate-0 的 `graph_draw`，因此与 Gate-0 的同类图配色、布局、标注完全一致，且不引入 torch / cdfm / causal-learn 依赖。

```powershell
# 默认：每个数据源各挑 CDFM 领先与 LiNGAM 领先各一个
python experiments/gate1_cdfm_defer/plot_graphs.py --gallery

# 指定数据集与数据源；--dataset 会自动解析到真正拥有它的数据源
python experiments/gate1_cdfm_defer/plot_graphs.py --dataset a1_uniform_reference
python experiments/gate1_cdfm_defer/plot_graphs.py --source tuebingen --dataset pair0050

# 某个 synthetic seed 在 5 个 λ 上的对比网格
python experiments/gate1_cdfm_defer/plot_graphs.py --sweep router_ood_tanh_seed_1000

# 列出全部可用 dataset_id
python experiments/gate1_cdfm_defer/plot_graphs.py --list
```

三个数据源的记录格式并不一致，脚本按各自约定归一化：

- `synthetic` / `causal_chamber` 直接使用保存的 `truth_adjacency` 与 `lingam_adjacency_source_target`。
- `tuebingen` 不保存真图与二值化的 LiNGAM 邻接：真图由 `true_direction`（如 `0->1`，含义是 `x[:,0]` 导致 `x[:,1]`）还原，LiNGAM 邻接按 benchmark 相同的 `(|B|.T > 1e-8)` 约定现算。两项都与 `run_tuebingen_benchmark.py` 的判定口径一致。
- 拓扑序优先取 synthetic `base_specs` 中保存的生成期顺序；Tübingen 与 Causal Chamber 没有该文件，改用 ground truth 上的确定性 Kahn 排序。
- `scale_ood_d120_n1000` 为 D=120，分层箭头图无法保持可读，默认跳过（`--max-nodes`，默认 40）。

`--style diff`（默认）按 true positive / false positive / false negative 着色，直接暴露多画的箭头与漏掉的箭头；`--style plain` 只画各方法自己的预测箭头。两种风格下三张图共用同一套节点坐标，可以逐边对比。输出写入 `figures/graphs/<source>/`，`--gallery` 额外写出 `figures/graphs/index.html`；这些图不参与任何 Gate 判定。
