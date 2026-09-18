# Gate-1 实验报告：CDFM–DirectLiNGAM External Validity

文档职责：报告冻结 Gate-0 Router 在 synthetic OOD、scale OOD 与两个真实 benchmark 上的外部效度、复现核验和生死决策。

适用范围：只覆盖 CDFM、DirectLiNGAM 与冻结 Router；不涉及生产 Agent、MCP 或重新训练。

## 结论

**BORDERLINE**。External complementarity survives, but the frozen Gate-0 Router does not generalize across noise shift and real intervention environments.

全部 synthetic `190/190`、Causal Chamber A1/A2 `1+20/1+20`、Tübingen strict-95 `95/95` 已完成。Router 在读取任何 Gate-1 标签前冻结；Gate-0 文件哈希在各 runner 启动时重新验证。

## 各 benchmark 主结果

| Benchmark | Always CDFM | Always LiNGAM | Best Single | Frozen Hybrid | Oracle | Oracle gap | Hybrid improvement | Captured gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| router_ood_student_t | 0.8273 | 0.6499 | 0.8273 | 0.7462 | 0.8657 | 0.0384 | -0.0812 | -2.1159 |
| router_ood_tanh | 0.7905 | 0.8116 | 0.8116 | 0.8594 | 0.9053 | 0.0937 | 0.0478 | 0.5102 |
| scale_ood_d120_n1000 | 0.4141 | 0.5440 | 0.5440 | 0.6331 | 0.6331 | 0.0890 | 0.0890 | 1.0000 |
| scale_ood_n40_d10 | 0.3876 | 0.3968 | 0.3968 | 0.4244 | 0.4802 | 0.0833 | 0.0276 | 0.3308 |
| causal_chamber_a2 | 0.7136 | 0.5603 | 0.7136 | 0.6119 | 0.7185 | 0.0049 | -0.1016 | -20.6537 |
| tuebingen_strict95 | 0.6347 | 0.5050 | 0.6347 | 0.6699 | 0.8062 | 0.1714 | 0.0352 | 0.2051 |

Synthetic 与 Causal Chamber 的指标是 directed edge F1；Tübingen 是官方 metadata 权重下的 forced-direction accuracy，不能直接把两种指标跨 benchmark 求总平均。

## 外部效度判断

- tanh mechanism OOD：Oracle gap `0.0937`，Hybrid improvement `0.0478`，captured gap `0.5102`。
- Student-t(df=3) noise OOD：Oracle gap `0.0384`，但 Hybrid improvement `-0.0812`。这说明互补性仍在，Router 的分布特征发生迁移。
- D=120 scale OOD：Router 在 λ=0/1 上完整捕获 Oracle；N=40 的低样本 OOD 只捕获 `0.3308`。
- Causal Chamber A1：CDFM F1 `0.7273`，与官方 notebook 的 `0.7273` 一致；A2 notebook ensemble CDFM F1 `0.7353`、AUROC `0.9686`，复现通过。
- A2 的 20 个环境上，Frozen Hybrid 相对 best single 为 `-0.1016`；其中 `2` 个强干预环境产生常数列，冻结特征管线按约束拒绝输入并显式回退默认 CDFM。
- Tübingen CDFM/DirectLiNGAM 加权准确率为 `0.6347` / `0.5050`；aggregate sanity `PASS`，具体偏差保留在结果 JSON，未据此更换 decoder。

## 为什么不是 GO 或 STOP

这不是 `STOP`：四个 synthetic OOD/scale strata 仍有可利用的 Oracle headroom，算法 portfolio 的互补性没有消失。也不是 `GO`：冻结 Router 在 Student-t OOD 和 Causal Chamber 真实干预环境上显著低于 best single，因此最准确的表述是 **complementarity survives, routing does not generalize**。下一阶段只有在不读取测试标签的前提下解决分布不变特征、常数列契约与安全 abstain/fallback，才值得继续；不应直接扩展到更多 solver 或 Agent/MCP。

## 复现与边界

Causal Chamber A2 同时保存了当前官方 notebook 的 adjacency-vote、平均 logits 阈值和论文文字 probability-average 三条路径；正式口径在运行前固定为 notebook adjacency-vote，没有择优。Tübingen 的公开材料没有 pair-level decoder，Gate-1 使用预先固定的 off-diagonal probability 比较与 LiNGAM causal order；失败和平局均保留在 95 对加权分母中。

CDFM 的 A2 notebook 路径可精确复现；DirectLiNGAM 的论文 A2 聚合代码未公开。按与 CDFM 对齐的 `5/20` 投票得到 F1 `0.6250`，不等于论文 `0.349`。为避免后验择阈值，结果文件完整保存 `1/20` 到 `20/20` 的投票 sweep；其中 `1/20` 为 `0.3422`，虽接近论文值，也只视为协议诊断，不能冒充精确复现。

Bootstrap 使用 dataset/environment/pair 为单位、2,000 次、种子 `20260916`。完整区间、胜率、技术失败、环境版本和 raw predictions 位于 `results/`。本轮没有验证生产 Agent/MCP integration，也没有在 Gate-1 上重新拟合 Router。
