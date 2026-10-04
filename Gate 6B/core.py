"""Offline contracts; no model runtime or reference-network access."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'sachs_selfcompat_jev'
RESULTS = ROOT / 'results'
RUNS = [f'repeat_{i:02d}' for i in range(20)] + ['full']
CONFIGS = ['auto', 'fixed_0.5']

def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_bytes(canonical(x))
    tmp.replace(p)

def graph_hash(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int8).tobytes()).hexdigest()

def fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid regression inputs')
    dx = x - x.mean()
    b = 0.0 if np.ptp(x) == 0 else float(dx @ (y-y.mean()) / (dx @ dx))
    return {'intercept': float(y.mean()-b*x.mean()), 'slope': b, 'constant_fallback': bool(np.ptp(x) == 0)}

def predict(model, x):
    return np.clip(model['intercept'] + model['slope'] * np.asarray(x, float), 0, 110)

def metrics(y, pred):
    d = np.asarray(pred, float)-np.asarray(y, float)
    return {'mae': float(np.mean(abs(d))), 'rmse': float(np.sqrt(np.mean(d*d))), 'bias': float(np.mean(d))}

def outcome(delta):
    return 'tie' if abs(delta) <= 1e-8 else ('jev_win' if delta < 0 else 'sc_win')

def splits(records):
    for group in RUNS:
        test = [i for i, r in enumerate(records) if r['run_id'] == group]
        train = [i for i, r in enumerate(records) if r['run_id'] != group and r['run_id'] != 'full']
        if len(test) != 2 or len(train) != (40 if group == 'full' else 38):
            raise ValueError('Unexpected fold sizes')
        yield group, train, test

def payload(report, numerical, headers, dag, question):
    # Strict allowlist; no result tables, thresholds, run IDs, or sibling graphs.
    numerical = {k: numerical[k] for k in ('limitations', 'n_samples', 'pairs', 'variables')}
    numerical['variables'] = [{k: v[k] for k in ('name', 'mean', 'std_population', 'missing_count', 'min_q25_median_q75_max')} for v in numerical['variables']]
    numerical['pairs'] = [{k: v[k] for k in ('variables', 'pearson', 'spearman', 'partial_pearson_controlling_all_other_variables')} for v in numerical['pairs']]
    report = {'research_context': report['research_context'], 'limitations': report['limitations'],
              'variables': [{k: v[k] for k in ('name', 'possible_meaning', 'measurement_interpretation', 'uncertainty')} for v in report['variables']]}
    graph = {'nodes': headers, 'edges': [[headers[s], headers[t]] for s, t in zip(*np.where(dag))]}
    return {'model': 'jev-1.13.0', 'state': canonical({'background': report, 'numerical_evidence': numerical, 'candidate': graph}).decode('utf-8'),
            'questions': {'structural_error': question}}

def validate_score(response, criteria):
    if response.get('model') != 'jev-1.13.0' or set(response.get('answers', {})) != {'structural_error'}:
        raise ValueError('Unexpected model or answer keys')
    a = response['answers']['structural_error']
    if a.get('type') != 'score':
        raise ValueError('Expected Score')
    def number(v, lo, hi):
        return type(v) in (int, float) and np.isfinite(v) and lo <= v <= hi
    if not number(a.get('score'), 0, 4) or not number(a.get('confidence'), 0, 1):
        raise ValueError('Invalid score or confidence')
    probs = a.get('probabilities', {})
    if set(probs) != set(map(str, range(5))) or not all(number(v, 0, 1) for v in probs.values()):
        raise ValueError('Invalid probabilities')
    # API displays rounded probability values; tolerate that rounding only.
    if abs(sum(probs.values())-1) > .026 or abs(sum(int(k)*v for k,v in probs.items())-a['score']) > .056:
        raise ValueError('Inconsistent probability distribution')
    if a.get('legend') != {str(i): c for i,c in enumerate(criteria)}:
        raise ValueError('Unexpected legend')
    return a
