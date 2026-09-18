# Gate-2：机制迁移下的安全整图路由实验

文档职责：记录 Gate-2 的独立任务生成、求解、冻结、测试、风险校准与生死判据。
适用范围：只覆盖 `experiments/gate2_reliable_routing/`；不修改或重解释 Gate-0/Gate-1 与生产 Agent。

最终决策：**BORDERLINE**。safe substantive candidate=l2d_weighted

## 固定协议

任务单位是独立 DAG 图；源域 train/dev/calibration 为 600/300/600，六个未见迁移域各 300 个测试图。所有方法共享 X-only 诊断特征与 CDFM/DirectLiNGAM 计算结果，测试只运行一次。
风险门槛固定为平均错误切换损失 H ≤ 0.02、严重损失 T（损失 > 0.10）≤ 5%；失败任务保留在技术失败账本和总分母中。

## 测试汇总

测试任务总数 `1800`，成功 `1800`，技术失败 `0`（失败率 `0.0000`）。

| 方法 | H | T | 净收益 vs CDFM | 切换率 | 同时校正 H 上界 | 同时校正 T 上界 |
|---|---:|---:|---:|---:|---:|---:|
| always_cdfm | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| always_lingam | 0.2029 | 0.4639 | -0.1180 | 1.0000 | 0.4198 | 0.9354 |
| old_gate0_router | 0.0902 | 0.2328 | -0.0552 | 0.5089 | 0.3223 | 0.8620 |
| logistic_router | 0.0574 | 0.1433 | 0.0063 | 0.5494 | 0.2207 | 0.4800 |
| gain_regression | 0.0005 | 0.0011 | 0.0274 | 0.1917 | 0.0040 | 0.0200 |
| l2d_weighted | 0.0013 | 0.0028 | 0.0356 | 0.2422 | 0.0136 | 0.0267 |
| gain_crc | 0.0024 | 0.0072 | 0.0274 | 0.2122 | 0.0140 | 0.0487 |
| gain_ltt | 0.0005 | 0.0011 | 0.0273 | 0.1894 | 0.0041 | 0.0200 |
| conformal_uncertain | 0.0013 | 0.0028 | 0.0356 | 0.2422 | 0.0131 | 0.0267 |
| support_gate | 0.0001 | 0.0000 | 0.0116 | 0.0650 | 0.0017 | 0.0000 |
| conservative_min_ltt | 0.0017 | 0.0056 | -0.0017 | 0.0061 | 0.0227 | 0.0687 |
| support_gain_ltt | 0.0001 | 0.0000 | 0.0116 | 0.0650 | 0.0021 | 0.0000 |

逐任务预测、路由分数和损失见 `results/router_predictions.csv`；域级统计见 `results/domain_summary.csv`；2,000 次任务级 bootstrap 的同时校正结果见 `results/bootstrap_summary.csv`。

## 机制迁移与消融

`results/mechanism_summary.csv` 按目标机制/噪声汇总了两种基础算法的胜率、Oracle gap 和所有路由策略。互补性并非均匀存在：interaction+exponential 的 LiNGAM 胜率为 81.7%，softsign+Gaussian 的 CDFM 胜率为 95.3%；因此仅凭训练域上的总体收益不能推出跨域安全。

`results/ablation_summary.csv` 是不重新拟合测试标签的策略消融：普通收益预测、L2D-inspired 加权分类、CRC/LTT、conformal 回退、支持距离门控和跨源最小收益门控均使用同一冻结特征与测试结果。普通收益回归和 LTT 的同时校正风险上界分别为 H=0.0040/T=0.0200 与 H=0.0041/T=0.0200，净收益约 0.027；跨源保守候选的同时上界为 H=0.0227/T=0.0687（softsign+Gaussian 域贡献），且总体净收益为负。

## 生死结论

本轮不是 `GO`：确有可复现的互补收益，且多个普通收益预测/风险控制基线满足六域风险约束并达到至少 0.015 的总体净收益；但新候选没有超过同信息强基线，跨源保守化还牺牲了收益并在一个迁移域超出风险预算。因此结论为 `BORDERLINE`，下一步应优先研究机制识别/特征可迁移性，而不是宣称已得到可靠通用路由器。

## 复现与边界

冻结账本 `frozen/freeze_manifest.json` 保存输入哈希、特征 schema、阈值和模型哈希；`verify_gate2.py` 负责独立复算。验证结果见 `results/verification.json`：任务、solver、raw、特征、预测公式和风险校准检查均通过。唯一警告是 test diagnostics checkpoint 重写全量特征 CSV 后产生源特征的字节级 digest 变化；源任务 identity、solver digest、schema、模型哈希和阈值均保持一致，未重新拟合或改变测试结果。技术失败没有从分母删除，诊断失败强制回退 CDFM。

本实验是严格合成筛选。Gate-1 的真实数据结果只作问题诊断，本轮不把目标真实数据加入训练、校准或测试。
