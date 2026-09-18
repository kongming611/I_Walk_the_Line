# Gate-3A.1：跨机制、跨图实例的 invariant-risk 独立确认

最终判定：**BORDERLINE**。confirmation retained signal but missed GO: mean rho=0.448。

## Gate-3A 审计边界

Gate-3A 有三个需要显式修正但不应称为数据泄漏的问题：

1. 固定 D=10 时，`cdfm_edge_count`、`cdfm_edge_density`、`cdfm_mean_degree` 是线性重复信息。
2. `cdfm_mean_confidence = 0.5 + cdfm_mean_margin`，二者也是线性重复信息。
3. Gate-3A 的 train/test mechanism folds 共享相同 graph seeds；它是 mechanism holdout，但存在 train/test group dependence / repeated underlying graph instances，不是 graph-instance-independent confirmation。

这些问题影响模型设定与独立性解释，但这里不将它们描述为数据泄漏。

## 1. Post-hoc development evidence

主候选 `stable_ridge_v1` 使用固定五特征与 `StandardScaler + Ridge(alpha=1.0)`，在旧 Gate-3A cache 上重放四个 LOFO folds。旧 60 个 CDFM tasks 没有重新 inference。

这些 feature 是在观察 Gate-3A 后确定的，因此旧 60 tasks 上的结果是 post-hoc exploratory，不得作为最终验证。

| held-out family | Spearman | risk@50 | full risk |
|---|---:|---:|---:|
| linear | 0.593 | 0.154 | 0.194 |
| tanh | 0.501 | 0.175 | 0.229 |
| rff | 0.692 | 0.207 | 0.249 |
| interaction | 0.413 | 0.186 | 0.233 |

### 相同 graph seed 的旧风险相关 sanity check

四个机制共享 15 个底层 graph seeds。所有 pairwise Spearman 见 `same_seed_risk_correlations.csv`；尤其 tanh vs interaction 为 **0.892**。该分析只量化旧实验的 graph-instance dependence，不进入任何 predictor feature。

### Threshold-aware diagnostics

原 Gate-3A 的 margin 以 0.5 为中心，而 CDFM 使用 official auto threshold。本轮从旧 raw cache 与新 raw cache 的 `cdfm_probabilities`、`cdfm_threshold` 计算 threshold、margin mean/p10/p25 与 near-fraction(0.02/0.05)。它们只进入预先冻结的 secondary Ridge；没有替换 primary。

## 2. Pre-registered independent confirmation

训练集仅为旧 Gate-3A 的 linear/tanh/RFF/interaction 60 tasks。测试集为 25 个 softsign（320000–320024）和 25 个 sine（321000–321024），均为 Laplace、D=10、N=1000；两个 family 不共享 seed，也不与旧 310000–310014 重叠。因此测试同时是 unseen mechanism family 与 unseen graph instances。CDFM 保持冻结；每个新任务运行一次原始 inference 和 B=2 bootstrap。

所有 scaler、OOD 统计量和 Ridge 均只 fit 旧 60 tasks。`model_fit_receipt.json` 在加载 confirmatory truth 前写出。没有 hyperparameter search。

### Primary stable_ridge_v1

| family | rho (95% bootstrap CI) | Pearson | MAE | risk@50 (95% CI) | full risk | relative reduction@50 |
|---|---:|---:|---:|---:|---:|---:|
| softsign | 0.665 [0.369, 0.841] | 0.696 | 0.062 | 0.132 [0.081, 0.200] | 0.198 | 33.3% |
| sine | 0.231 [-0.176, 0.589] | 0.312 | 0.085 | 0.271 [0.226, 0.329] | 0.300 | 9.9% |

Primary mean rho=0.448；平均 relative risk reduction@50=21.6%。25/50/75/100% coverage 的完整结果见 `confirmatory_results.csv`。

### Frozen baselines and secondary

| family | confidence-only rho / risk@50 | threshold-aware rho / risk@50 |
|---|---:|---:|
| softsign | 0.596 / 0.149 | 0.522 / 0.145 |
| sine | -0.115 / 0.303 | 0.537 / 0.247 |

confidence-only 平均 rho=0.241、平均 risk@50=0.226；primary 平均 risk@50=0.201。

constant mean-risk、metadata-only、OOD-only、old all-feature candidate Ridge 的完整对照，以及 secondary threshold-aware Ridge，均在 `confirmatory_results.csv`。

## Gate 与不可变性声明

按 `frozen_protocol.json` 的预注册判据，结果为 **BORDERLINE**。测试结果产生后，feature set、model、alpha、threshold 与 normalization 的修改：**No**。
