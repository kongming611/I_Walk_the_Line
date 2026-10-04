# Sachs Self-Compatibility / LLM + Jev

独立实验：同一 Sachs CSV 上，CDFM 官方自动阈值与固定 0.5 生成候选；比较论文 κG 与 DeepSeek 表头背景 + Jev 的选图表现。无需、也不导入旧 Gate 实验结果。

## 执行

运行环境：`D:\miniconda3\python.exe`。所有命令在仓库根目录执行。

```powershell
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\bootstrap.py
& 'D:\miniconda3\python.exe' -m pytest -q experiments\sachs_selfcompat_jev\test_experiment.py
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\run.py prepare
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\run.py infer
& experiments\sachs_selfcompat_jev\configure_credentials.ps1
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\run.py analyze
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\run.py judge
& 'D:\miniconda3\python.exe' experiments\sachs_selfcompat_jev\run.py evaluate
```

`analyze` 和 `judge` 需要网络及配置凭据的同一个 Windows 用户；CPU 推理可在无网络的沙箱中执行。加密凭据只保存在本目录 `.credentials.xml`，使用 Windows DPAPI，不提交。程序不输出密钥。

`infer` 自动断点续算；`--limit 1` 只用于首份数据的完整技术检查，不改变协议。两个阈值共享一次前向推理。任何已存在预测均不因重跑改变。软件/权重变化会拒绝继续冻结实验。

## 冻结口径

- 一次7466行全量实验，20次固定种子的1000行无放回抽样；始终11个变量。
- 每份数据40个6变量子集（论文向上取整）及40个5变量子集（源码向下取整）；子集及原始列序预先保存。
- 完整图和边际图采用相同CDFM及确定性无环处理。原始阈值图也保留，不把模型输出环伪装成混杂。
- κG 直接使用未修改作者 `ADMG.marginalize`、`ADMG.shd`、`SelfCompatibilityScorer._graphical_compatibility`。作者ADMG的互逆弧表达限制仍然存在。
- Jev只接收背景、数值证据和候选图，不接收阈值、κG、参考网络和旧有示例；交换顺序两次一致才接受，平局回退auto。
- SHD反向计2；Skeleton F1对无向骨架计算。参考图按作者修正PIP3→PIP2。仅evaluate解析参考文件。
- 两方法共用数据与候选，但语义信息、计算预算不同。20次重复不是独立生物系统，不支持全面超过结论。
- DeepSeek峰值费率输入$0.30/M、输出$1.20/M；Jev输入$0.042/M。预算按保守字节上界累计预留，限$2；实际账单金额未知。每请求至多3次尝试，原始响应缓存后不因结构校验失败重新付费。

## 产物

`protocol.json` 冻结数据、权重、软件、随机子集与提示词哈希。`results/inference/` 保存所有前向结果、数值证据与κG；`results/api/` 保存原始响应；`results/judgments/` 保存两次选择。所有预测冻结后才写 `truth_open_receipt.json` 并生成 `per_run_results.csv`、`evaluation.json`、`REPORT.md`。

源码依赖复制自已核验干净的作者提交 `9084041f9c1e238250ce281868c4f6d2971d5e0c`，仅软件，不复用旧实验。Sachs来自CDT公开CSV，下载字节哈希见 `data/provenance.json`。

来源：[论文](https://proceedings.mlr.press/v238/faller24a.html)、[作者代码](https://github.com/amazon-science/causal-self-compatibility)、[CDT数据](https://github.com/FenTechSolutions/CausalDiscoveryToolbox/tree/master/cdt/data/resources)、[CDFM](https://github.com/DMIRLAB-Group/CDFM)、[Jev Choice](https://docs.typesafe.ai/primitives/choice)、[DeepSeek费率](https://api-docs.deepseek.com/quick_start/pricing/)。
