# I_Walk_the_Line

## Sachs 对照实验

| 实验 | 内容 | 数据与 DeepSeek 分析 | 结果 |
|---|---|---|---|
| [Gate 6A](Gate%206A/START_HERE.md) | Self-Compatibility 与 LLM＋Jev 的选图对照 | [数据](Gate%206A/data/) · [DeepSeek 报告](Gate%206A/analysis/deepseek_report.md) | [中文报告](Gate%206A/results/REPORT.md) |
| [Gate 6B](Gate%206B/START_HERE.md) | 两种评分经相同校准后预测真实 SHD | [数据](Gate%206B/data/) · [DeepSeek 报告](Gate%206B/analysis/deepseek_report.md) | [中文报告](Gate%206B/results/REPORT.md) |

两个目录均包含实际数据、20 份抽样 CSV、DeepSeek 原始分析、代码、冻结协议、原始模型响应和评价结果。
目录中的 `offline_verify.py` 可在任意克隆位置离线复核，不调用 CDFM、DeepSeek 或 Jev；详见各目录的 `START_HERE.md`。
凭据、API 密钥和模型权重不包含在仓库中。
