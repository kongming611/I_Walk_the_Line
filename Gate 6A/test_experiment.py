"""Small oracle examples and failure cases, independent of Sachs reference truth."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from core import project, metrics, select_sc, combine_choices, evidence, canonical
from api import validate_choice, validate_report


def test_shd_orientation_and_skeleton():
    truth = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    reversed_one = np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
    assert metrics(reversed_one, truth)['shd'] == 2
    assert metrics(reversed_one, truth)['skeleton_f1'] == 1
    assert metrics(np.zeros((3, 3)), truth)['skeleton_f1'] == 0
    mixed = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    assert metrics(mixed, truth)['shd'] == 2
    assert metrics(mixed, truth)['skeleton_f1'] == .5
    assert run.graph(reversed_one, [0, 1, 2]).shd(run.graph(truth, [0, 1, 2])) == 2


def test_author_latent_chain():
    a = np.zeros((3, 3), int)
    a[0, 1] = a[1, 2] = 1
    marginal = run.graph(a, [0, 1, 2]).marginalize([0, 2]).graph
    assert set(marginal.edges) == {(0, 2)}


def test_author_latent_common_cause():
    a = np.zeros((3, 3), int)
    a[1, 0] = a[1, 2] = 1
    marginal = run.graph(a, [0, 1, 2]).marginalize([0, 2]).graph
    assert set(marginal.edges) == {(0, 2), (2, 0)}


def test_upstream_self_loop_discrepancy_is_exposed_not_silently_fixed():
    from audit import latent_dag_oracle
    a = np.zeros((7, 7), int)
    for i, j in [(0, 2), (0, 4), (1, 3), (1, 6), (3, 4), (3, 5), (3, 6)]:
        a[i, j] = 1
    nodes = [2, 6, 5, 4]
    upstream = run.graph(a, list(range(7))).marginalize(nodes).graph
    assert upstream.has_edge(6, 6)
    assert np.trace(latent_dag_oracle(a, nodes)) == 0


def test_cycle_and_global_index_ties():
    p = np.array([[0, .8, 0], [0, 0, .8], [.8, 0, 0]])
    raw, a = project(p, .5, [2, 0, 1])
    assert raw.sum() == 3 and a.sum() == 2
    assert a[1, 2] and a[2, 0] and not a[0, 1]
    assert project(p, .8, [2, 0, 1])[1].sum() == 0


@pytest.mark.parametrize('forward,reverse,selected,unclear', [
    ('A', 'B', 'auto', False), ('B', 'A', 'fixed_0.5', False),
    ('A', 'A', 'auto', True), ('B', 'B', 'auto', True),
    ('unclear', 'B', 'auto', True), ('A', 'unclear', 'auto', True)])
def test_order_swap(forward, reverse, selected, unclear):
    result = combine_choices(forward, reverse)
    assert result['selected'] == selected and result['undecided'] == unclear


def test_ties_identical_and_failure():
    assert select_sc([3, 2])['selected'] == 'fixed_0.5'
    assert select_sc([2, 2])['undecided']
    assert select_sc([3, 2], identical=True)['selected'] == 'auto'
    assert combine_choices('B', 'A', identical=True)['undecided']
    with pytest.raises(ValueError):
        select_sc([np.nan, 0])


def test_partial_correlation_shared_cause():
    rng = np.random.default_rng(44)
    z = rng.normal(size=10000)
    x = np.column_stack([z + rng.normal(size=len(z)) * .3, z + rng.normal(size=len(z)) * .3, z])
    e = evidence(x, ['x', 'y', 'z'])
    assert e['pairs'][0]['pearson'] > .8
    assert abs(e['pairs'][0]['partial_pearson_controlling_all_other_variables']) < .04


def test_request_isolation_and_swap():
    dags = np.zeros((2, 3, 3), int)
    dags[0, 0, 1] = 1
    dags[1, 1, 0] = 1
    headers = ['x', 'y', 'z']
    a = run.judge_payload({'research_context': 'unknown'}, {}, dags, headers)
    b = run.judge_payload({'research_context': 'unknown'}, {}, dags, headers, reverse=True)
    import json
    sa, sb = json.loads(a['state']), json.loads(b['state'])
    assert sa['candidates']['A'] == sb['candidates']['B']
    assert sa['candidates']['B'] == sb['candidates']['A']
    assert set(sa) == {'background', 'numerical_evidence', 'candidates'}
    for forbidden in ('kappa', 'threshold', 'reference_edges', 'true_shd'):
        assert forbidden not in a['state']


def test_api_choice_contract():
    response = {'model': 'jev-1.13.0', 'answers': {'better_graph': {'type': 'choice', 'choice': 'A',
                 'probabilities': {'A': .8, 'B': .1, 'unclear': .1}}}}
    assert validate_choice(response) == 'A'
    response['answers']['better_graph']['probabilities']['B'] = -1
    with pytest.raises(ValueError):
        validate_choice(response)


def test_budget_and_cache_do_not_require_secret(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, 'RESULTS', tmp_path)
    monkeypatch.setattr(api, 'credential', lambda _: 'test-never-network')
    api.save(tmp_path / 'api_ledger.json', [{'request_sha256': 'other', 'budget_charge_usd': 2}])
    with pytest.raises(RuntimeError, match='budget'):
        api.request('jev', {'model': 'jev-1.13.0'})


def test_frozen_results_reject_new_paid_call(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr(api, 'RESULTS', tmp_path)
    api.save(tmp_path / 'prediction_freeze.json', {})
    with pytest.raises(RuntimeError, match='frozen'):
        api.request('jev', {'model': 'jev-1.13.0'})
