"""Offline final report additions and scientific plots; no model calls."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import read, save, digest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def main():
    frame = pd.read_csv(RESULTS / 'per_run_results.csv')
    result = read(RESULTS / 'evaluation.json')
    audit = read(RESULTS / 'final_audit.json')
    differences = []
    for row in frame.to_dict('records'):
        for size in (6, 5):
            diag = audit['path_definition_diagnostic_scores'][row['run_id']][str(size)]
            selected = diag['selection']['selected']
            prefix = 'auto' if selected == 'auto' else 'fixed'
            differences.append({'run_id': row['run_id'], 'subset_size': size, 'path_auto_kappa': diag['scores']['auto'],
                                'path_fixed_kappa': diag['scores']['fixed_0.5'], 'path_selected': selected,
                                'path_shd': row[prefix + '_shd'], 'path_f1': row[prefix + '_skeleton_f1'],
                                'author_selected': row[f'sc{size}_selected'], 'selection_changed': selected != row[f'sc{size}_selected'],
                                'jev_minus_path_shd': row['jev_shd'] - row[prefix + '_shd'],
                                'jev_minus_path_f1': row['jev_skeleton_f1'] - row[prefix + '_skeleton_f1']})
    diagnostic = pd.DataFrame(differences)
    diagnostic.to_csv(RESULTS / 'path_definition_diagnostic.csv', index=False, encoding='utf-8-sig')
    repeat = frame.iloc[1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    x = np.arange(20)
    axes[0].plot(x, repeat.sc6_shd, 'o-', label='Self-Compatibility (6 variables)', color='#3178b9')
    axes[0].plot(x, repeat.jev_shd, 's--', label='DeepSeek + Jev', color='#e2822a')
    axes[0].set(xlabel='Sachs subsample seed', ylabel='Selected graph SHD (lower is better)', xticks=range(0, 20, 2))
    axes[0].legend(fontsize=8)
    delta = repeat.jev_minus_sc6_shd.to_numpy()
    axes[1].bar(x, delta, color=['#299d75' if v < 0 else '#c96060' if v > 0 else '#a2a2a2' for v in delta])
    axes[1].axhline(0, color='#555555', linewidth=.8)
    axes[1].set(xlabel='Sachs subsample seed', ylabel='SHD difference: Jev minus Self-Compatibility', xticks=range(0, 20, 2))
    fig.suptitle('One biological system, 20 fixed subsamples; paired candidate graphs')
    fig.savefig(RESULTS / 'paired_results.png', dpi=180)
    plt.close(fig)
    marker = '\n## 独立审计与作者实现差异\n'
    base = (RESULTS / 'REPORT.md').read_text(encoding='utf-8').split(marker)[0]
    lines = [marker, '',
             f'选图与回退解释：Self-Compatibility 和 Jev 在全部21次运行中最终均选择了自动阈值候选。20次重复中，Jev 有 {int(repeat.jev_undecided.sum())}/20 次因无法区分或换序不一致而回退默认配置；全量主实验也使用了回退。因此最终指标打平不等于两种判断能力已被证明等价，本轮也未展示相对始终采用自动阈值的额外收益。', '',
             f'验证了 {audit["checked_forward_caches"]} 次前向缓存、{audit["checked_graphs"]} 张无环候选图、{audit["independently_recounted_distances"]} 个子集距离及42项参考图指标。', '',
             f'CDFM 前向累计耗时：{audit["cdfm_sum_forward_seconds"]/60:.2f} 分钟（各次调用墙钟时间之和，不包含 API、审计和其他开销）。', '',
             f'独立路径定义与作者边际化实现共有 {len(audit["path_oracle_vs_author_marginalization_mismatches"])} 次差异。该检查在读取参考网络前登记，原作者实现仍是主结果。', '',
             '| 范围 | 子集变量数 | 按路径定义复算后改变选择的次数 | Jev−路径基线平均SHD | Jev−路径基线平均F1 |', '|---|---:|---:|---:|---:|']
    for full, label in ((True, '全量'), (False, '20次重复')):
        for size in (6, 5):
            g = diagnostic[(diagnostic.run_id.eq('full') == full) & (diagnostic.subset_size == size)]
            lines.append(f'| {label} | {size} | {int(g.selection_changed.sum())} | {g.jev_minus_path_shd.mean():.3f} | {g.jev_minus_path_f1.mean():.4f} |')
    primary_diagnostic = diagnostic[(diagnostic.run_id != 'full') & (diagnostic.subset_size == 6)]
    robust = bool(result['repeats']['jev_vs_sc6']['positive_signal'] and primary_diagnostic.jev_minus_path_shd.mean() < 0 and primary_diagnostic.jev_minus_path_f1.mean() >= 0)
    lines += ['', '优势对路径定义核对是否稳健：' + ('是，仍满足SHD更低且F1不降低。' if robust else '没有形成同时满足主结果与路径核对的稳健积极信号。'), '',
              '软件修订记录在 protocol_history：001仅修正参考CSV表头读取及首次开启回执；002登记作者边际化可能产生自环的独立诊断。没有修改抽样、候选、提示词、主评分规则或预算。', '',
              '![配对结果](paired_results.png)', '', '逐次完整指标：per_run_results.csv；路径诊断：path_definition_diagnostic.csv；原始审计：final_audit.json。']
    (RESULTS / 'REPORT.md').write_text(base + '\n'.join(lines) + '\n', encoding='utf-8')
    save(RESULTS / 'delivery_receipt.json', {'files': {name: digest(RESULTS / name) for name in ('REPORT.md', 'per_run_results.csv', 'evaluation.json', 'final_audit.json', 'paired_results.png', 'path_definition_diagnostic.csv')},
                                          'robust_positive_signal': robust})
    print('Offline audited report and paired plot delivered.')


if __name__ == '__main__':
    main()
