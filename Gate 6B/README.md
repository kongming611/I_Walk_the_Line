# Sachs SHD prediction: calibrated Self-Compatibility vs LLM + Jev

本实验复用 `../sachs_selfcompat_jev/` 全部输入。不会加载或调用 CDFM，不重新调用 DeepSeek。
旧数据只读。新 Jev 请求为每图独立五级 Score；confidence 不用作质量分数。

## 固定执行顺序

在本目录使用 `D:\miniconda3\python.exe`：

1. `-m unittest -v test_experiment`：离线验证。
2. `run.py prepare`：验证旧收据，冻结代码、提示词、缓存哈希及协议。
3. `run.py score`：唯一付费步骤，须以配置凭据的 Windows 用户运行；自动续用完整响应缓存。最多 42 次成功请求，累计旧新保守预算不超过 2 美元。
4. `run.py evaluate`：仅在全部评分冻结后解析标签，留一抽样组交叉验证。
5. `audit.py`：独立矩阵 SHD、最小二乘、请求与哈希核验，并运行测试。
6. `deliver.py`：中文报告、逐图表、图像及交付哈希。

已有 `score_freeze.json` 时重复 score 只验证缓存。冻结源文件改变会拒绝执行。技术失败保留原始响应与预算占用，不补默认评分。失败记录不等于方法失败。

## 统计定义

20 组重复共 40 张图，按组留出，每折用其余 38 张图拟合带截距的一元普通最小二乘。
两路线各自拟合，允许负斜率，不调参。分数完全恒定时回退训练均值。所有预测裁剪到 [0,110]，不取整。
全量两图不参加训练标签拟合，用 40 个重复样本训练后的公式单独预测。
主 MAE，辅 RMSE、有符号误差；逐图/逐组胜平负，绝对误差差值 1e-8 内平局。
训练均值和中位数作朴素参照；子集 5 的 κG 仅敏感性检查。原作者自环行为保留，不在本轮改写原分数。

这是原始结果已知后提出的探索实验。新评分程序与标签隔离是软件流程隔离，不是系统级保密。
数据来自同一 Sachs 系统且抽样重叠；不得将记录当独立系统做显著性或泛化声明。
新增监督校准不是 Self-Compatibility 原论文的方法。Jev 等级只是预测特征，尚无图质量校准保证。

来源：[Self-Compatibility](https://proceedings.mlr.press/v238/faller24a.html)、[Jev Score](https://docs.typesafe.ai/primitives/score)、[统计学习导论](https://www.statlearning.com/)。
