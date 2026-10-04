"""Sachs two-judge experiment, isolated from historical Gate experiment modules."""
from __future__ import annotations
import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
UPSTREAM = ROOT / 'vendor/causal-self-compatibility'
sys.path[:0] = [str(ROOT / 'vendor/python_deps'), str(UPSTREAM), str(UPSTREAM / 'src')]

import numpy as np
import pandas as pd
import networkx as nx
from core import CONFIGS, canonical, digest, save, read, project, metrics, evidence, adjacency_hash, select_sc, combine_choices

RESULTS = ROOT / 'results'
PROTOCOL = ROOT / 'protocol.json'
CHECKPOINT = Path('C:/Users/charc/.cache/huggingface/hub/models--DMIRLAB--CDFM/snapshots/fd3ac4f72e617c9b2165082493f2494c9f980d04')


def software_hashes():
    paths = [ROOT / n for n in ('core.py', 'api.py', 'run.py', 'prompts.json', 'bootstrap.py')]
    paths.extend(p for p in UPSTREAM.rglob('*') if p.is_file() and p.suffix == '.py')
    paths.extend(p for p in (ROOT / 'vendor/python_deps').rglob('*.py'))
    import cdfm
    paths.extend(Path(cdfm.__file__).parent.rglob('*.py'))
    paths.extend(CHECKPOINT / f for f in ('config.json', 'model.safetensors'))
    return {str(p): digest(p) for p in sorted(paths)}


def verify():
    p = read(PROTOCOL)
    for name, expected in p['software_hashes'].items():
        if digest(name) != expected:
            raise RuntimeError('Frozen software or checkpoint changed: ' + name)
    if digest(ROOT / 'data/cyto_full_data.csv') != p['observations_sha256']:
        raise RuntimeError('Observations changed')
    return p


def prepare():
    if PROTOCOL.exists():
        verify()
        print('Existing protocol verified.', flush=True)
        return
    frame = pd.read_csv(ROOT / 'data/cyto_full_data.csv')
    headers = frame.columns.tolist()
    expected = ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 'p44/42', 'pakts473', 'PKA', 'PKC', 'P38', 'pjnk']
    if frame.shape != (7466, 11) or headers != expected or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError('Unexpected public Sachs data contract')
    runs = []
    for index in range(21):
        full = index == 0
        row_seed = None if full else index - 1
        rows = list(range(len(frame))) if full else np.random.default_rng(row_seed).choice(len(frame), 1000, replace=False).tolist()
        rng = np.random.default_rng(10400 + index)
        subsets = {str(size): [rng.choice(11, size, replace=False).tolist() for _ in range(40)] for size in (6, 5)}
        runs.append({'id': 'full' if full else f'repeat_{row_seed:02d}', 'row_seed': row_seed, 'row_indices': rows,
                     'subset_seed': 10400 + index, 'subsets': subsets})
    p = {'schema': 'sachs_selfcompat_jev_v1', 'created_at': time.time(), 'headers': headers,
         'alias_map': {'p44/42': 'pff_ft'}, 'observations_sha256': digest(ROOT / 'data/cyto_full_data.csv'),
         'reference_sha256': read(ROOT / 'data/provenance.json')['downloads']['cyto_full_target.csv']['sha256'],
         'source': read(ROOT / 'data/provenance.json'), 'runs': runs, 'configs': list(CONFIGS),
         'cdfm': {'checkpoint': str(CHECKPOINT), 'device': 'cpu', 'threads': 4, 'standardize': True,
                  'fixed_threshold': .5, 'matrix': 'source,target', 'projection': 'greedy descending probability then original column index; skip cycle-forming edges'},
         'compatibility': {'primary_size': 6, 'sensitivity_size': 5, 'num_subsets': 40,
             'implementation': 'unchanged author ADMG.marginalize and SelfCompatibilityScorer._graphical_compatibility',
             'limitations': 'Author ADMG uses reciprocal arcs for bidirected edges and cannot distinguish bows. CDFM outputs DAGs; removing variables introduces latent confounding not represented by subset CDFM outputs.'},
         'judges': {'llm': 'deepseek-flash', 'jev': 'jev-1.13.0', 'question': 'pairwise smaller SHD',
                    'swap_check': True, 'tie_fallback': 'auto', 'semantic_report': 'once from headers only',
                    'api_budget_usd': 2., 'max_attempts': 3, 'max_payload_bytes': 30000,
                    'rates_peak_usd_per_million': {'deepseek_input': .3, 'deepseek_output': 1.2, 'jev_input': .042}},
         'evaluation': {'shd': 'directed adjacency Hamming; reverse=2', 'f1': 'skeleton', 'sc_tie_tolerance': 1e-12,
                        'truth_isolation': 'software stages and manifests, not OS isolation; known public reference'},
         'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'torch', 'cdfm-base', 'networkx', 'scipy', 'cdt')},
         'software_hashes': software_hashes()}
    save(PROTOCOL, p)
    RESULTS.mkdir(exist_ok=True)
    print('Frozen: 21 runs, 2 configurations, 40 subsets each of size 6 and 5.', flush=True)


def write_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def graph(a, nodes):
    from src.causal_graphs.admg import ADMG
    g = nx.DiGraph()
    g.add_nodes_from(int(i) for i in nodes)
    g.add_edges_from((int(nodes[i]), int(nodes[j])) for i, j in zip(*np.where(a)))
    return ADMG(g)


def infer(limit=None):
    p = verify()
    if (RESULTS / 'inference_freeze.json').exists():
        verify_manifest(RESULTS / 'inference_freeze.json')
        print('All inference already complete and verified.', flush=True)
        return
    import torch
    from cdfm import CDFM
    from cdfm.utils import suggest_threshold
    from self_compatibility import SelfCompatibilityScorer
    torch.set_num_threads(4)
    np.random.seed(10400)
    torch.manual_seed(10400)
    model = CDFM.from_pretrained(str(CHECKPOINT), device='cpu')
    data = pd.read_csv(ROOT / 'data/cyto_full_data.csv').to_numpy(float)
    started = time.perf_counter()
    calls = 0
    def predict(x, nodes, path):
        nonlocal calls
        if path.exists():
            return dict(np.load(path, allow_pickle=False))
        start = time.perf_counter()
        result = model.predict(x, standardize=True)
        # Official auto threshold, and fixed 0.5, derived from the exact same forward pass.
        threshold = float(result.threshold)
        raw, dags = [], []
        for t in (threshold, .5):
            r, a = project(result.probabilities, t, nodes)
            raw.append(r)
            dags.append(a)
        arrays = {'probabilities': result.probabilities, 'logits': result.logits, 'raw': np.array(raw),
                  'dags': np.array(dags), 'thresholds': np.array([threshold, .5]),
                  'nodes': np.array(nodes), 'runtime_sec': np.array(time.perf_counter() - start)}
        write_npz(path, **arrays)
        calls += 1
        return arrays
    selected_runs = p['runs'] if limit is None else p['runs'][:limit]
    for ordinal, run in enumerate(selected_runs):
        directory = RESULTS / 'inference' / run['id']
        x = data[run['row_indices']]
        full = predict(x, list(range(11)), directory / 'full.npz')
        save(directory / 'evidence.json', evidence(x, p['headers']))
        for size in (6, 5):
            marginals = [[], []]
            distances = [[], []]
            for k, nodes in enumerate(run['subsets'][str(size)]):
                sub = predict(x[:, nodes], nodes, directory / f'size{size}_{k:02d}.npz')
                for config in range(2):
                    marginal = graph(sub['dags'][config], nodes)
                    joint = graph(full['dags'][config], list(range(11)))
                    marginals[config].append(marginal)
                    distances[config].append(float(marginal.shd(joint.marginalize(nodes))))
                if (k + 1) % 10 == 0:
                    print(f'{run["id"]} size={size} subsets={k+1}/40 new_forward_calls={calls} elapsed={time.perf_counter()-started:.1f}s', flush=True)
            scores = [SelfCompatibilityScorer._graphical_compatibility(graph(full['dags'][i], list(range(11))), marginals[i]) for i in range(2)]
            if not np.allclose(scores, np.mean(distances, axis=1), atol=1e-12):
                raise RuntimeError('Author scorer disagreement')
            identical = bool(np.array_equal(*full['dags']))
            save(directory / f'sc_size{size}.json', {'scores': dict(zip(CONFIGS, scores)), 'distances': dict(zip(CONFIGS, distances)),
                 'selection': select_sc(scores, identical), 'candidate_hashes': [adjacency_hash(a) for a in full['dags']]})
        print(f'COMPLETE {run["id"]}: edges={full["dags"].sum(axis=(1,2)).tolist()}', flush=True)
    if limit is None or limit >= 21:
        freeze(RESULTS / 'inference_freeze.json', list((RESULTS / 'inference').rglob('*.npz')) + list((RESULTS / 'inference').rglob('*.json')))
    print(f'Inference invocation finished; {calls} new forward calls.', flush=True)


def freeze(path, files):
    save(path, {'protocol_sha256': digest(PROTOCOL), 'created_at': time.time(),
                'files': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}})


def verify_manifest(path):
    record = read(path)
    if record['protocol_sha256'] != digest(PROTOCOL):
        raise RuntimeError('Protocol mismatch in receipt')
    for name, expected in record['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen result changed: ' + name)
    return record


def analyze():
    p = verify()
    if (RESULTS / 'semantic_report.json').exists():
        verify_manifest(RESULTS / 'semantic_freeze.json')
        print('Semantic report already frozen.', flush=True)
        return
    from api import request, validate_report
    prompt = read(ROOT / 'prompts.json')
    payload = {'model': 'deepseek-flash', 'temperature': 0, 'thinking': {'type': 'disabled'},
               'max_tokens': 4096, 'response_format': {'type': 'json_object'},
               'messages': [{'role': 'system', 'content': prompt['deepseek_system']},
                            {'role': 'user', 'content': canonical({'columns': p['headers']}).decode('utf-8')}]}
    response, key = request('deepseek', payload)
    report = validate_report(response, p['headers'])
    save(RESULTS / 'semantic_report.json', {'report': report, 'request_sha256': key, 'actual_model': response.get('model')})
    freeze(RESULTS / 'semantic_freeze.json', [RESULTS / 'semantic_report.json', RESULTS / 'api' / f'deepseek_{key}.json'])
    print('Header-only semantic report generated and frozen.', flush=True)


def judge_payload(report, numerical, dags, headers, reverse=False):
    order = [1, 0] if reverse else [0, 1]
    candidates = {}
    for label, i in zip(('A', 'B'), order):
        a = dags[i]
        candidates[label] = {'nodes': headers, 'edges': [[headers[s], headers[t]] for s, t in zip(*np.where(a))],
                             'edge_count': int(a.sum())}
    return {'model': 'jev-1.13.0', 'state': canonical({'background': report, 'numerical_evidence': numerical,
            'candidates': candidates}).decode('utf-8'), 'questions': {'better_graph': read(ROOT / 'prompts.json')['jev_question']}}


def judge():
    p = verify()
    verify_manifest(RESULTS / 'inference_freeze.json')
    verify_manifest(RESULTS / 'semantic_freeze.json')
    if (RESULTS / 'prediction_freeze.json').exists():
        verify_manifest(RESULTS / 'prediction_freeze.json')
        print('Judgments already frozen.', flush=True)
        return
    from api import request, validate_choice
    report = read(RESULTS / 'semantic_report.json')['report']
    for run in p['runs']:
        output = RESULTS / 'judgments' / (run['id'] + '.json')
        if output.exists():
            continue
        directory = RESULTS / 'inference' / run['id']
        dags = np.load(directory / 'full.npz')['dags']
        identical = bool(np.array_equal(*dags))
        choices, requests = [], []
        if not identical:
            for reverse in (False, True):
                payload = judge_payload(report, read(directory / 'evidence.json'), dags, p['headers'], reverse)
                response, key = request('jev', payload)
                choices.append(validate_choice(response))
                requests.append(key)
        decision = combine_choices(*(choices if choices else ['unclear', 'unclear']), identical=identical)
        save(output, {'selection': decision, 'choices_forward_reverse': choices, 'request_hashes': requests,
                      'candidate_hashes': [adjacency_hash(a) for a in dags]})
        print('JEV ' + run['id'] + ': ' + decision['selected'] + ' (' + decision['reason'] + ')', flush=True)
    files = list((RESULTS / 'judgments').glob('*.json')) + list((RESULTS / 'api').glob('*')) + list((RESULTS / 'requests').glob('*.json'))
    files += [RESULTS / 'semantic_report.json', RESULTS / 'inference_freeze.json', RESULTS / 'semantic_freeze.json', RESULTS / 'api_ledger.json']
    freeze(RESULTS / 'prediction_freeze.json', files)


def evaluate():
    p = verify()
    verify_manifest(RESULTS / 'inference_freeze.json')
    verify_manifest(RESULTS / 'prediction_freeze.json')
    if digest(ROOT / 'data/cyto_full_target.csv') != p['reference_sha256']:
        raise RuntimeError('Reference bytes changed')
    # FIRST reference parsing in the pipeline, after every prediction is frozen.
    if not (RESULTS / 'truth_open_receipt.json').exists():
        save(RESULTS / 'truth_open_receipt.json', {'time': time.time(), 'prediction_freeze_sha256': digest(RESULTS / 'prediction_freeze.json'),
             'reference_sha256': p['reference_sha256']})
    edges = pd.read_csv(ROOT / 'data/cyto_full_target.csv')
    if edges.shape[1] != 2:
        raise RuntimeError('Unexpected reference edge file')
    index = {h: i for i, h in enumerate(p['headers'])}
    truth = np.zeros((11, 11), np.int8)
    for s, t in edges.itertuples(index=False, name=None):
        truth[index[s], index[t]] = 1
    if truth[index['PIP2'], index['PIP3']] != 1:
        raise RuntimeError('Expected author reference correction cannot be applied')
    truth[index['PIP2'], index['PIP3']] = 0
    truth[index['PIP3'], index['PIP2']] = 1
    rows = []
    for run in p['runs']:
        directory = RESULTS / 'inference' / run['id']
        full = dict(np.load(directory / 'full.npz'))
        measurement = [metrics(a, truth) for a in full['dags']]
        sc6, sc5 = read(directory / 'sc_size6.json'), read(directory / 'sc_size5.json')
        jev = read(RESULTS / 'judgments' / (run['id'] + '.json'))
        hashes = [adjacency_hash(a) for a in full['dags']]
        if not all(r['candidate_hashes'] == hashes for r in (sc6, sc5, jev)):
            raise RuntimeError('Judges did not evaluate identical candidates')
        row = {'run_id': run['id'], 'n': len(run['row_indices']), 'identical_candidates': bool(np.array_equal(*full['dags'])),
               'truth_shd_tie': measurement[0]['shd'] == measurement[1]['shd'],
               'auto_threshold': float(full['thresholds'][0])}
        for i, config in enumerate(CONFIGS):
            prefix = 'auto' if i == 0 else 'fixed'
            row.update({prefix + '_' + k: v for k, v in measurement[i].items()})
            row[prefix + '_edges'] = int(full['dags'][i].sum())
            row[prefix + '_removed_cycle_edges'] = int(full['raw'][i].sum() - full['dags'][i].sum())
            row[prefix + '_kappa6'] = sc6['scores'][config]
            row[prefix + '_kappa5'] = sc5['scores'][config]
        for method, record in [('sc6', sc6), ('sc5', sc5), ('jev', jev)]:
            selection = record['selection']
            chosen = CONFIGS.index(selection['selected'])
            row[method + '_selected'] = selection['selected']
            row[method + '_undecided'] = selection['undecided']
            row[method + '_reason'] = selection['reason']
            row[method + '_shd'] = measurement[chosen]['shd']
            row[method + '_skeleton_f1'] = measurement[chosen]['skeleton_f1']
            difference = measurement[chosen]['shd'] - measurement[1 - chosen]['shd']
            row[method + '_selection_quality'] = 'better' if difference < 0 else 'worse' if difference > 0 else 'equal'
        row['jev_minus_sc6_shd'] = row['jev_shd'] - row['sc6_shd']
        row['jev_minus_sc6_skeleton_f1'] = row['jev_skeleton_f1'] - row['sc6_skeleton_f1']
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / 'per_run_results.csv', index=False, encoding='utf-8-sig')
    def summary(group):
        result = {'n': len(group), 'identical_candidates': int(group.identical_candidates.sum()), 'true_shd_ties': int(group.truth_shd_tie.sum())}
        for method in ('sc6', 'sc5', 'jev'):
            result[method] = {'mean_shd': float(group[method + '_shd'].mean()),
              'mean_skeleton_f1': float(group[method + '_skeleton_f1'].mean()),
              'undecided': int(group[method + '_undecided'].sum()),
              'selection_quality_counts': group[method + '_selection_quality'].value_counts().to_dict()}
        result['jev_vs_sc6'] = {'mean_shd_difference': float(group.jev_minus_sc6_shd.mean()),
             'mean_f1_difference': float(group.jev_minus_sc6_skeleton_f1.mean()),
             'wins': int((group.jev_minus_sc6_shd < 0).sum()), 'ties': int((group.jev_minus_sc6_shd == 0).sum()),
             'losses': int((group.jev_minus_sc6_shd > 0).sum()),
             'positive_signal': bool(group.jev_minus_sc6_shd.mean() < 0 and group.jev_minus_sc6_skeleton_f1.mean() >= 0)}
        return result
    from api import ledger
    costs = ledger()
    result = {'full': summary(frame.iloc[:1]), 'repeats': summary(frame.iloc[1:]),
         'reference_edges': int(truth.sum()), 'api_attempts': len(costs),
         'api_conservative_budget_usd': sum(e['budget_charge_usd'] for e in costs),
         'api_usage_peak_price_bound_usd': sum(e.get('peak_price_usage_bound_usd', e['budget_charge_usd']) for e in costs),
         'actual_billed_amount_known': False, 'protocol_sha256': digest(PROTOCOL),
         'limitations': ['Single known biological system; repeats not independent systems.', 'Sachs mixed-condition CSV and public reference network.',
          'Semantic priors may include benchmark familiarity; no ablation in this experiment.', 'DAG discovery is not closed under latent marginalization.',
          'Different semantic and computational information budgets; compare complete systems only.']}
    save(RESULTS / 'evaluation.json', result)
    report(result, frame)
    print(canonical(result).decode(), flush=True)


def report(result, frame):
    lines = ['# Sachs：Self-Compatibility 与 LLM＋Jev 对照结果', '',
        '这是原论文图形不相容评分在 CDFM 上的应用；不是原论文四种发现算法的完整复现。两种方法评价同一对候选图。', '',
        '## 全量主实验与20次抽样重复', '',
        '| 范围 | 方法 | 平均 SHD ↓ | 平均 Skeleton F1 ↑ | 严格选优 / 同 SHD / 选差 | 无法区分 |',
        '|---|---|---:|---:|---|---:|']
    for scope, label in [('full', '全量主实验'), ('repeats', '20次重复')]:
        for key, name in [('sc6', 'Self-Compatibility（6变量，主口径）'), ('sc5', 'Self-Compatibility（5变量，敏感性）'), ('jev', 'DeepSeek＋Jev')]:
            s = result[scope][key]
            c = s['selection_quality_counts']
            lines.append(f'| {label} | {name} | {s["mean_shd"]:.3f} | {s["mean_skeleton_f1"]:.4f} | {c.get("better",0)} / {c.get("equal",0)} / {c.get("worse",0)} | {s["undecided"]} |')
    comparison = result['repeats']['jev_vs_sc6']
    lines += ['', f'20次重复中，Jev 相对主基线胜/平/负：{comparison["wins"]}/{comparison["ties"]}/{comparison["losses"]}；平均 SHD 差（Jev−基线）={comparison["mean_shd_difference"]:.3f}，平均 Skeleton F1 差={comparison["mean_f1_difference"]:.4f}。',
      '', ('达到预先定义的积极信号：平均 SHD 更低且 Skeleton F1 不下降。' if comparison['positive_signal'] else '未达到预先定义的积极信号；不得将本轮描述为全面超过基线。'),
      '', '## 完整逐次结果', '', '| 运行 | 自动/固定 SHD | 主基线选择 | Jev选择 | SHD差 |', '|---|---|---|---|---:|']
    for row in frame.to_dict('records'):
        lines.append(f'| {row["run_id"]} | {row["auto_shd"]}/{row["fixed_shd"]} | {row["sc6_selected"]} | {row["jev_selected"]} | {row["jev_minus_sc6_shd"]} |')
    lines += ['', '## 口径与限制', '',
      '- SHD 按作者 CDT 口径，反向计2；Skeleton F1不看方向。平局及Jev换序不一致均回退自动阈值，并单列无法区分。',
      '- 子集6变量对应论文向上取整，5变量对应作者源码向下取整；两套均预先冻结。',
      '- 21次运行属于同一Sachs系统；20次重复不能当20个独立真实世界任务，不计算跨系统显著性。',
      '- 使用作者修正后的公开参考网络及混合实验条件数据；相对参考网络的误差不等于生物学绝对真值。',
      '- CDFM输出DAG，变量子集可能产生隐藏混杂；κG可能反映这个表达能力限制。作者ADMG以互逆弧表示混杂，不能区分同时存在直接边的情况。',
      '- 没有匿名消融，不能把提升单独归因于变量背景或Jev；熟知公开Sachs网络的可能影响也不能排除。',
      '- 两方法原始数据、候选一致，语义先验和计算预算不同；报告总耗时和API费用。', '',
      f'API尝试次数：{result["api_attempts"]}；按峰值单价及用量估算的费用上界 ${result["api_usage_peak_price_bound_usd"]:.6f}；保守预算占用 ${result["api_conservative_budget_usd"]:.6f}。这不是账户账单实扣金额。',
      '', '## 复现与来源', '', '[原论文](https://proceedings.mlr.press/v238/faller24a.html) · [作者代码](https://github.com/amazon-science/causal-self-compatibility) · [CDT数据](https://github.com/FenTechSolutions/CausalDiscoveryToolbox/tree/master/cdt/data/resources)', '',
      '协议见 ../protocol.json；逐次明细见 per_run_results.csv；软件、权重、数据及预测哈希分别见协议和各阶段 freeze 回执。参考网络首次解析在所有预测冻结之后；这是代码阶段隔离，并非操作系统隔离。']
    (RESULTS / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'infer', 'analyze', 'judge', 'evaluate', 'verify'])
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.stage == 'infer':
        infer(args.limit)
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
