# Gate-2：机制迁移下的安全整图路由实验

文档职责：说明 Gate-2 的独立任务、冻结边界、运行顺序、产物和验证方法。
适用范围：仅覆盖本目录；不修改 Gate-0、Gate-1 或生产代码。

## 固定协议

- 任务单位是独立 DAG 图，而不是矩阵行；源域为 train/dev/calibration = 600/300/600。
- 测试包含 6 个未见机制/噪声迁移域，每域 300 个独立任务，共 1,800 个测试任务。
- 每个任务含 `X (1000,10)`、真实邻接矩阵、独立 graph seed；训练、开发、校准和测试 seed 不重叠。
- 所有路由共享 CDFM、DirectLiNGAM、七个 Gate-0 数值特征和本目录新增的 X-only 诊断特征。
- 冻结后才打开测试标签；测试只运行一次。技术失败留在分母，诊断失败强制回退 CDFM。
- 固定风险门槛：平均错误切换损失 `H <= 0.02`，严重损失比例 `T <= 0.05`。

## 运行顺序

```powershell
python experiments\gate2_reliable_routing\generate_tasks.py
python experiments\gate2_reliable_routing\run_solvers.py --smoke
python experiments\gate2_reliable_routing\compute_diagnostics.py --split smoke
python experiments\gate2_reliable_routing\run_solvers.py --split train
python experiments\gate2_reliable_routing\run_solvers.py --split dev
python experiments\gate2_reliable_routing\run_solvers.py --split calibration
python experiments\gate2_reliable_routing\compute_diagnostics.py --split train
python experiments\gate2_reliable_routing\compute_diagnostics.py --split dev
python experiments\gate2_reliable_routing\compute_diagnostics.py --split calibration
python experiments\gate2_reliable_routing\fit_and_evaluate.py --fit
python experiments\gate2_reliable_routing\run_solvers.py --split test
python experiments\gate2_reliable_routing\compute_diagnostics.py --split test --checkpoint-every 20
python experiments\gate2_reliable_routing\fit_and_evaluate.py --evaluate
python experiments\gate2_reliable_routing\summarize_results.py
python experiments\gate2_reliable_routing\verify_gate2.py
```

`run_solvers.py`、`compute_diagnostics.py` 和生成器都支持 checkpoint/resume；已经存在且完整的任务不会被重复运行。`fit_and_evaluate.py --fit` 在 frozen 目录存在时拒绝覆盖模型、阈值和输入账本。

## 方法和产物

比较方法包括 Always CDFM/LiNGAM、旧 Gate-0 Router、同特征 Logistic、收益回归、L2D-CD-inspired 收益加权分类、CRC、LTT、conformal 不确定集合回退、支持距离门控、跨源机制保守收益评分及其支持门控变体。

- `data/manifest.csv`：完整任务清单和 seed 隔离账本。
- `results/raw_predictions/`、`solver_results.csv`：逐任务基础模型和矩阵方向证据。
- `results/diagnostic_features.csv`：只从 `X` 和重采样图稳定性生成的诊断特征。
- `frozen/models.joblib`、`frozen/freeze_manifest.json`：模型、特征顺序、阈值、风险预算和哈希。
- `results/router_predictions.csv`：逐任务路由分数、选择、F1、错误切换损失和 Oracle gap。
- `results/domain_summary.csv`、`results/bootstrap_summary.csv`：六域统计和 2,000 次任务级 bootstrap 的同时校正上界。
- `results/mechanism_summary.csv`、`results/ablation_summary.csv`：机制/噪声分析和预注册策略消融。
- `results/risk_gain_frontier.png`、`REPORT.md`：风险—收益图和 GO/BORDERLINE/STOP 报告。
- `results/verification.json`：独立一致性检查结果。
- `results/sha256_manifest.json`：包含任务、raw、源代码、冻结模型、汇总和报告的 6,629 个文件 SHA-256。

本次运行的报告结论由封存测试给出：`BORDERLINE`。普通收益预测与风险控制方法可以在多个迁移域保留实质互补收益；跨源最小收益候选没有超过这些强基线，且在 softsign+Gaussian 域的同时校正风险上界不达标。
