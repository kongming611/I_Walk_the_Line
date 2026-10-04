"""Pure numerical contracts; no models, credentials, or reference-network reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import rankdata

CONFIGS = ('auto', 'fixed_0.5')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(canonical(value))
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def adjacency_hash(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int8).tobytes()).hexdigest()


def project(probabilities, threshold, nodes):
    """Greedy maximum-confidence DAG; deterministic tie order in global node IDs."""
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(nodes), len(nodes)) or not np.isfinite(p).all():
        raise ValueError('invalid probability matrix')
    if not 0 <= threshold <= 1 or np.any((p < 0) | (p > 1)):
        raise ValueError('invalid probability/threshold range')
    raw = (p > threshold).astype(np.int8)
    np.fill_diagonal(raw, 0)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))
    kept = np.zeros_like(raw)
    edges = sorted(zip(*np.where(raw)), key=lambda e: (-p[e], nodes[e[0]], nodes[e[1]]))
    for i, j in edges:
        if not nx.has_path(graph, j, i):
            graph.add_edge(i, j)
            kept[i, j] = 1
    return raw, kept


def metrics(predicted, truth):
    """Author CDT SHD: each directed adjacency mismatch counts once; reversal=2."""
    a, b = np.asarray(predicted, bool), np.asarray(truth, bool)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('mismatched graph shapes')
    if np.diag(a).any() or np.diag(b).any():
        raise ValueError('self loops')
    ap, bp = (a | a.T)[np.triu_indices(len(a), 1)], (b | b.T)[np.triu_indices(len(a), 1)]
    tp = int((ap & bp).sum())
    fp, fn = int((ap & ~bp).sum()), int((~ap & bp).sum())
    denominator = 2 * tp + fp + fn
    return {'shd': int((a != b).sum()), 'skeleton_f1': 2 * tp / denominator if denominator else 1.,
            'skeleton_tp': tp, 'skeleton_fp': fp, 'skeleton_fn': fn}


def select_sc(scores, identical=False):
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite compatibility score')
    tie = identical or abs(scores[0] - scores[1]) <= 1e-12
    return {'selected': CONFIGS[0 if tie or scores[0] < scores[1] else 1],
            'undecided': bool(tie), 'reason': 'identical' if identical else 'score_tie' if tie else 'lower_kappa'}


def combine_choices(forward, reverse, identical=False):
    if identical:
        return {'selected': 'auto', 'undecided': True, 'reason': 'identical'}
    if forward not in ('A', 'B', 'unclear') or reverse not in ('A', 'B', 'unclear'):
        raise ValueError('invalid Jev choice')
    a = {'A': 'auto', 'B': 'fixed_0.5', 'unclear': None}[forward]
    b = {'A': 'fixed_0.5', 'B': 'auto', 'unclear': None}[reverse]
    agreement = a is not None and a == b
    return {'selected': a if agreement else 'auto', 'undecided': not agreement,
            'reason': 'order_consistent' if agreement else 'unclear_or_order_disagreement'}


def evidence(x, headers):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or np.any(x.std(axis=0) == 0):
        raise ValueError('Sachs data must be complete, finite and nonconstant')
    pearson = np.corrcoef(x.T)
    spearman = np.corrcoef(np.apply_along_axis(rankdata, 0, x).T)
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    pairs = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            controls = np.column_stack([np.ones(len(x)), np.delete(z, [i, j], axis=1)])
            residual = z[:, [i, j]] - controls @ np.linalg.lstsq(controls, z[:, [i, j]], rcond=None)[0]
            partial = float(np.corrcoef(residual.T)[0, 1]) if np.all(residual.std(axis=0) > 1e-12) else None
            pairs.append({'variables': [headers[i], headers[j]], 'pearson': round(float(pearson[i, j]), 6),
                          'spearman': round(float(spearman[i, j]), 6),
                          'partial_pearson_controlling_all_other_variables': round(partial, 6) if partial is not None else None})
    variables = []
    for i, name in enumerate(headers):
        variables.append({'name': name, 'missing_count': 0, 'mean': round(float(x[:, i].mean()), 6),
                          'std_population': round(float(x[:, i].std()), 6),
                          'min_q25_median_q75_max': np.quantile(x[:, i], [0, .25, .5, .75, 1]).round(6).tolist()})
    return {'n_samples': len(x), 'variables': variables, 'pairs': pairs,
            'limitations': 'Correlations and linear partial correlations do not establish causal direction or absence of nonlinear effects. Background semantics are uncertain prior knowledge, not an observed reference graph. Rows may mix experimental conditions.'}
