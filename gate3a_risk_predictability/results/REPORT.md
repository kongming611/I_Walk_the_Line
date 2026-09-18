# Gate-3A：冻结 CDFM 的结构错误风险可预测性

最终判定：**BORDERLINE**。primary Ridge passed 2/4 rho gates; mean rho=0.200。

## 科学问题与必要性

本轮检验：测试时完全不知道真实 DAG 时，仅凭观测数据、CDFM 预测图、CDFM 直接提供的概率、与 DirectLiNGAM 的预测分歧及可选 bootstrap 稳定性，能否预测 `1 - directed F1`，并泛化到 risk predictor 未见的机制族。若连风险排序都不能跨机制迁移，后续 risk certificate 即使形式上校准，也缺少可用的个体化风险信号。

## 数据与防泄漏设计

使用 Gate-2 generator 产生 60 个 D=10、N=1000、Laplace 噪声任务。15 个 graph seed 在 linear、tanh、RFF、interaction 间共享完全相同的 DAG、边权与外生噪声；linear 使用 lambda=0，其余使用 lambda=1。interaction 沿用 Gate-2 定义：逐边 tanh，并在入度至少为 2 时加入父变量交互项。

主评估是四折 leave-one-mechanism-family-out，因为随机切分会让同机制分布同时出现在训练和测试中，不能回答跨未见机制族泛化。真实图只用于监督 target 和最终评价；risk 输入不含真实图、CDFM/LiNGAM F1、真实机制标签、oracle 或其他标签派生量。所有 scaler、OOD 中心/协方差和模型均只在三个训练机制族上拟合。

CDFM 未训练、未微调、未改源码；使用已安装 `cdfm-base` 的官方自动阈值。API 直接返回 probabilities/logits，因此加入概率摘要；bootstrap B 值见 predictions.csv。预注册主候选是全诊断 Ridge，固定 RF 仅为辅助敏感性分析。

## 四个未见机制族结果

| held-out family | CDFM mean F1 | risk variance | candidate rho | metadata rho | OOD rho | confidence rho | candidate risk@50% | full risk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear | 0.806 | 0.0135 | 0.286 | 0.125 | -0.123 | 0.713 | 0.148 | 0.194 |
| tanh | 0.771 | 0.0105 | 0.440 | 0.326 | -0.233 | 0.193 | 0.175 | 0.229 |
| rff | 0.751 | 0.0084 | -0.373 | -0.036 | -0.563 | 0.695 | 0.293 | 0.249 |
| interaction | 0.767 | 0.0113 | 0.447 | 0.692 | -0.084 | 0.190 | 0.173 | 0.233 |

## 解释与 Gate

平均 Spearman（常数的未定义相关按 0 计）：candidate Ridge=0.200，metadata-only=0.277，OOD-only=-0.251，confidence-only=0.448，secondary RF=0.458。constant predictor 没有排序能力；其相关为 NA，risk-coverage 等价于随机期望。

按预注册 rho>0.30 标准，主候选成功 family：tanh, interaction；失败 family：linear, rff。主候选平均 risk@50%=0.197，低于 full/random=0.226 和 OOD-only=0.250，但高于 confidence-only=0.182；其平均 rho 也低于 metadata-only 与 confidence-only。因此它不是单纯 OOD detector，但新增诊断在 Ridge 中没有稳定提供超越 cheap baselines 的增量信息。

固定 RF 的辅助结果平均 rho=0.458，3/4 family 超过 0.30，平均 risk@50%=0.178；它达到点估计 GO 数值条件，但这是预先声明的 secondary sensitivity，不替换主候选，而且 RFF rho 仍只有 0.161。因此 Gate 保持 **BORDERLINE**，没有按测试结果更换主模型。bootstrap 95% CI、Pearson、MAE、25/50/75/100% coverage 均在 `fold_results.csv`。

## 失败模式与下一信号

最重要的失败模式是 RFF 的线性跨族负迁移：主候选 rho=-0.373，risk@50%=0.293 反而高于 full risk=0.249；同一 fold 的 confidence-only rho=0.695，说明全特征线性组合压坏了一个本来有用的直接信号。若继续，下一轮最值得增加的单一信号是 CDFM 的小扰动一致性（对输入加入幅度受控的观测噪声后比较图变化）；它仍不需要真实图，并比继续堆叠分布矩更直接地探测决策边界脆弱性。
