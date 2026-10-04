# Gate 6B：两种评分经校准后预测 SHD 的实验

## 从这里开始

- [中文实验报告](results/REPORT.md)
- [DeepSeek 分析报告（可读版）](analysis/deepseek_report.md)
- [DeepSeek 原始结构化报告](analysis/semantic_report.json)，以及 analysis/api、analysis/requests 中的原始请求响应
- [Sachs 原始数据：7466 行、11 变量](data/cyto_full_data.csv)
- [论文参考网络原始边表](data/cyto_full_target.csv)：评价时执行 PIP2→PIP3 改为 PIP3→PIP2 的作者修正
- [20 次抽样的实际 CSV](data/samples/)，[零起始原始数据行索引](data/sampling_row_indices.json)

两个 Gate 使用同一组原始数据、抽样行、候选图和同一份 DeepSeek 报告。这是已有报告的完整副本，归档没有再次调用任何模型。

## 克隆后离线复核

安装 `requirements-offline.txt` 中的 NumPy 后，在任意工作目录运行：

```powershell
python "Gate 6B/offline_verify.py"
```

也可以传入该脚本的绝对路径。脚本只读，不需要密钥、CDFM 权重或网络；验证整个归档的相对路径哈希、每份抽样数据、DeepSeek 原始响应与报告的一致性、图对应的真实 SHD，并重新计算本轮评价。

Gate 6A 验证缓存的 κG 距离均值、双顺序选择及选图指标；完整子集图和原作者边际化审计均保留。Gate 6B 使用本目录 data/candidates 中的缓存独立复算留组线性回归和 MAE/RMSE，不依赖 Gate 6A 的位置。

## 原始代码与冻结记录

本目录根部的 run.py、core.py、api.py、README.md 等是当时实际运行的代码与说明，按字节保留。results 和 protocol_history 保存原始结果、冻结、修复与核验记录。**历史 README 和脚本中的绝对路径不是克隆后的复跑入口**；已完成实验的可移植复核入口是上面的 offline_verify.py。

原始收据中保留当时机器的路径与权重指纹，以免改写实验历史。archive_provenance.json 逐文件记录原始目录及原始哈希；ARCHIVE_MANIFEST.json 使用本目录相对路径，适用于任意克隆位置。

不上传 Windows 加密凭据、API 密钥或模型权重。原 experiments 目录保留在原机器，未因归档修改。若未来重新采集模型评分或重新生成候选，应建立新的运行目录和协议，不能覆盖这些冻结结果。

## 解释边界

Sachs 为论文使用的混合实验数据版本；参考网络不等于绝对因果真相。20 次抽样来自同一系统且相互重叠，不是20个独立真实数据集。

本轮主结果：Self-Compatibility＋校准 MAE 为 2.863652，LLM＋Jev＋校准为 2.996735；Jev 在全量两图上更准。新增监督校准不是原论文方法；这是已知旧结果之后开展的探索实验。

逐图结果见 [per_graph_predictions.csv](results/per_graph_predictions.csv)。原 CSV BOM 编码修复完整保存在 protocol_history/csv_bom_fix 中；未改变评分或数值规则。
