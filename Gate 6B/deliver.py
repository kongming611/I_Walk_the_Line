"""Offline Chinese report and scientific plots; requires successful independent audit."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
from run import verify, check, manifest

LABELS={'sc6':'Self-Compatibility＋校准（主）','jev':'LLM＋Jev＋校准','sc5':'Self-Compatibility＋校准（子集5）','mean':'训练均值预测','median':'训练中位数预测'}

def deliver():
    verify();check(read(RESULTS/'score_freeze.json')['files'])
    audit=read(RESULTS/'audit.json');assert audit['passed']
    ev=read(RESULTS/'evaluation.json');s=ev['summary'];rs=ev['records']
    repeats=[r for r in rs if r['run_id']!='full'];full=[r for r in rs if r['run_id']=='full']
    delta=s['repeats']['jev']['mae']-s['repeats']['sc6']['mae']
    winner='LLM＋Jev＋校准' if delta < -1e-8 else 'Self-Compatibility＋校准' if delta>1e-8 else '两方法打平'
    relative=abs(delta)/s['repeats']['sc6']['mae']*100 if s['repeats']['sc6']['mae'] else 0
    lines=['# Sachs SHD 预测实验报告','',f'主评价：**{winner}**。Jev MAE 减去 Self-Compatibility MAE 为 **{delta:.6f}**；绝对差为 {abs(delta):.6f} 个 SHD 单位（相对 Self-Compatibility MAE 的 {relative:.2f}%）。',
           '', '这是已有候选图上的事后探索实验；两种原始分数都经过相同形式的一元线性回归校准。未重新调用 CDFM 或 DeepSeek。',
           '', '## 20 次抽样：40 张图的留组预测','', '| 方法 | MAE ↓ | RMSE ↓ | 有符号误差 |','|---|---:|---:|---:|']
    for name in LABELS:
        m=s['repeats'][name];lines.append(f"| {LABELS[name]} | {m['mae']:.6f} | {m['rmse']:.6f} | {m['bias']:.6f} |")
    w=s['repeats']['graph_wins'];g=s['repeat_group_wins']
    lines+=['',f"以 Jev 相对 Self-Compatibility 主结果计：逐图胜／平／负 **{w['jev_win']}／{w['tie']}／{w['sc_win']}**；逐抽样组平均绝对误差胜／平／负 **{g['jev_win']}／{g['tie']}／{g['sc_win']}**。",'', '## 按候选类型拆分','', '| 候选 | Self-Compatibility MAE | Jev MAE | Jev−SC |','|---|---:|---:|---:|']
    for config in CONFIGS:
        a=s[config]['sc6']['mae'];b=s[config]['jev']['mae'];lines.append(f'| {config} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines+=['','## 全量数据两张图（单独评价）','','| 候选 | 真实 SHD | κG | Jev 原分数 | SC预计SHD | Jev预计SHD | SC绝对误差 | Jev绝对误差 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in full:
        lines.append(f"| {r['config']} | {r['truth_shd']:.0f} | {r['sc6_score']:.4f} | {r['jev_score']:.4f} | {r['sc6_prediction']:.4f} | {r['jev_prediction']:.4f} | {r['sc6_absolute_error']:.4f} | {r['jev_absolute_error']:.4f} |")
    lines+=['',f"全量两图平均绝对误差：SC **{s['full']['sc6']['mae']:.6f}**，Jev **{s['full']['jev']['mae']:.6f}**。",'', '## 解释与稳定性','']
    for name in ('sc6','jev'):
        better=[LABELS[b] for b in ('mean','median') if s['repeats'][name]['mae']<s['repeats'][b]['mae']-1e-8]
        lines.append(f"- {LABELS[name]}："+('MAE 优于'+ '、'.join(better) if better else 'MAE 未超过两种常数参照，没有发现其优于这些简单预测的证据')+'。')
    if delta*(s['repeats']['jev']['rmse']-s['repeats']['sc6']['rmse'])<0:
        lines.append('- MAE 与 RMSE 的优劣方向冲突；主结论按冻结的 MAE，同时保留误差分布的权衡。')
    d=[s[c]['jev']['mae']-s[c]['sc6']['mae'] for c in CONFIGS]
    lines.append('- 两种候选类型的 MAE 优劣方向'+('一致。' if d[0]*d[1]>0 else '不一致或存在平局，优势依赖候选类型。'))
    lines.append(f"- 子集5敏感性 MAE：{s['repeats']['sc5']['mae']:.6f}；不替代子集6主结果。")
    models=read(RESULTS/'fold_models.json')
    for name in ('sc6','jev','sc5'):
        slopes=[m['models'][name]['slope'] for m in models if m['held_out_group']!='full']
        lines.append(f'- {LABELS[name]}校准斜率范围 [{min(slopes):.6f}, {max(slopes):.6f}]；负斜率折数 {sum(v<0 for v in slopes)}/20。负斜率如出现，说明校准在反转原始分数方向。')
    scores=[r['jev_score'] for r in repeats]
    lines.append(f'- Jev 原始分数范围 [{min(scores):.6f}, {max(scores):.6f}]，不同分值数 {len(set(scores))}；confidence 未作预测值。')
    lines+=['','## 费用与核验','',f"新增 API 请求尝试 {audit['api_attempts']} 次，42 次成功；接口累计耗时 {audit['api_runtime_sec']:.2f} 秒。按返回用量和单价估算的新费用上界 ${audit['new_usage_price_bound_usd']:.8f}，旧新用量费用上界合计 ${audit['new_usage_price_bound_usd']+audit['old_usage_price_bound_usd']:.8f}；旧新保守预算占用 ${audit['cumulative_reserved_usd']:.8f}，低于 $2。以上不是账单实付金额。",'',
           '17 项离线测试通过；独立重算 42 个真实 SHD、126 个回归预测、42 个请求白名单与图哈希。所有复用输入哈希未变。原始响应、协议、评分冻结和交付收据均保存。',
           '', '## 局限与复跑','', '这不是 Self-Compatibility 原生输出 SHD 的复现，而是两条路线加相同监督校准后的比较。五级 Jev 标准是本实验固定的设计，不是已验证的因果图误差量表。真实 SHD 使用论文参考网络，反向边计两次；参考网络不是绝对因果真相。',
           '', '20 次抽样来自同一 Sachs 系统且数据行重叠，真实 SHD 已在旧实验揭晓；这是探索性同系统预测，不是独立新系统测试。不把40张图当独立系统做显著性声明，不声称全面超过。',
           '', '原作者潜变量边际化的自环行为仍保留在已有 κG 中；前轮诊断曾发现该差异，本轮未改写原始分数。',
           '', '详见 [逐图结果](per_graph_predictions.csv)、[逐组误差](per_group_errors.csv)、[各折回归系数](fold_models.json)、[独立核验](audit.json) 和上级 README 的复跑步骤。',
           '', '![预测SHD与真实SHD](predicted_vs_true.png)','', '![配对误差](paired_errors.png)','']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    y=[r['truth_shd'] for r in repeats]
    for ax,name,title in zip(axes,['sc6','jev'],['Self-Compatibility + OLS','LLM + Jev + OLS']):
        for config,marker in zip(CONFIGS,['o','^']):
            subset=[r for r in repeats if r['config']==config]
            ax.scatter([r['truth_shd'] for r in subset],[r[name+'_prediction'] for r in subset],label=config,marker=marker,alpha=.75)
        allvalues=y+[r[name+'_prediction'] for r in repeats];lo=min(allvalues)-1;hi=max(allvalues)+1
        ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set(xlabel='True SHD',ylabel='Held-out predicted SHD',title=title,xlim=(lo,hi),ylim=(lo,hi));ax.legend();ax.grid(alpha=.2)
    fig.savefig(RESULTS/'predicted_vs_true.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
    x=np.arange(40)
    for name,color,label in [('sc6','#2969ad','Self-Compatibility + OLS'),('jev','#cd6a27','LLM + Jev + OLS')]:
        axes[0].plot(x,[r[name+'_absolute_error'] for r in repeats],'.-',label=label,color=color,lw=1)
    axes[0].set(ylabel='Absolute SHD prediction error',xlabel='Record (2 per repeat)');axes[0].legend();axes[0].grid(alpha=.2)
    groups=ev['groups'][:20];ds=[g['jev_minus_sc6_mae'] for g in groups]
    axes[1].bar(np.arange(20),ds,color=['#cd6a27' if v>0 else '#2969ad' for v in ds]);axes[1].axhline(0,color='black',lw=1)
    axes[1].set(xlabel='Repeat seed',ylabel='Jev MAE minus SC MAE',title='Negative: Jev better; positive: SC better',xticks=np.arange(20))
    fig.savefig(RESULTS/'paired_errors.png',dpi=180);plt.close(fig)
    paths=[p for p in RESULTS.iterdir() if p.is_file() and p.name not in ('delivery_receipt.json','last_failure.json')]
    save(RESULTS/'delivery_receipt.json',{'complete':True,'protocol_sha256':digest(ROOT/'protocol.json'),'files':manifest(paths)})
    print('Report, plots, and delivery hashes saved')

if __name__=='__main__':deliver()
