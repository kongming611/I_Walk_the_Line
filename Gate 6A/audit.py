"""Independent cache/metric audit; only reads reference after prediction freeze."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import networkx as nx
import run
from core import read, save, digest, project, adjacency_hash, metrics


def latent_dag_oracle(a, subset):
    """Path-based marginalization oracle, unlike upstream's sequential node removal."""
    g = nx.from_numpy_array(np.asarray(a), create_using=nx.DiGraph)
    hidden = set(g) - set(subset)
    output = np.zeros((len(subset), len(subset)), np.int8)
    for i, x in enumerate(subset):
        for j, y in enumerate(subset):
            if x == y:
                continue
            directed = nx.has_path(g.subgraph(hidden | {x, y}), x, y)
            common = any(nx.has_path(g.subgraph(hidden | {x}), h, x) and nx.has_path(g.subgraph(hidden | {y}), h, y) for h in hidden)
            output[i, j] = directed or common
    return output


def main(evaluated=False):
    started = time.time()
    p = run.verify()
    run.verify_manifest(run.RESULTS / 'inference_freeze.json')
    num_forward = 0
    checked_graphs = 0
    distances = 0
    runtime = 0.
    mismatches = []
    path_scores = {}
    for record in p['runs']:
        directory = run.RESULTS / 'inference' / record['id']
        full = dict(np.load(directory / 'full.npz'))
        path_scores[record['id']] = {}
        descriptors = [('full', list(range(11)), full)]
        for size in (6, 5):
            scores = read(directory / f'sc_size{size}.json')
            calculated = [[], []]
            path_distances = [[], []]
            for k, nodes in enumerate(record['subsets'][str(size)]):
                sub = dict(np.load(directory / f'size{size}_{k:02d}.npz'))
                descriptors.append((f'size{size}_{k:02d}', nodes, sub))
                for i in range(2):
                    expected = latent_dag_oracle(full['dags'][i], nodes)
                    official = run.graph(full['dags'][i], list(range(11))).marginalize(nodes).graph
                    actual = nx.to_numpy_array(official, nodelist=nodes, dtype=np.int8)
                    if not np.array_equal(expected, actual):
                        mismatches.append({'run_id': record['id'], 'size': size, 'subset': k, 'config': i,
                                           'upstream_self_loops': int(np.trace(actual)),
                                           'differing_entries': int(np.count_nonzero(expected != actual))})
                    path_distances[i].append(int(np.count_nonzero(sub['dags'][i] != expected)))
                    # Independently re-count directed mismatches, not calling upstream SHD.
                    distance = int(np.count_nonzero(sub['dags'][i] != actual))
                    assert distance == scores['distances'][run.CONFIGS[i]][k]
                    calculated[i].append(distance)
                    distances += 1
            for i in range(2):
                assert abs(float(np.mean(calculated[i])) - scores['scores'][run.CONFIGS[i]]) < 1e-12
            values = np.mean(path_distances, axis=1).tolist()
            path_scores[record['id']][str(size)] = {'scores': dict(zip(run.CONFIGS, values)),
                'selection': run.select_sc(values, bool(np.array_equal(*full['dags'])))}
        for _, nodes, z in descriptors:
            assert z['nodes'].tolist() == nodes
            assert np.isfinite(z['probabilities']).all()
            for i, threshold in enumerate(z['thresholds']):
                raw, dag = project(z['probabilities'], float(threshold), nodes)
                assert np.array_equal(raw, z['raw'][i]) and np.array_equal(dag, z['dags'][i])
                assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))
                checked_graphs += 1
            runtime += float(z['runtime_sec'])
            num_forward += 1
    assert num_forward == 1701 and checked_graphs == 3402 and distances == 3360
    result = {'checked_forward_caches': num_forward, 'checked_graphs': checked_graphs,
              'independently_recounted_distances': distances, 'cdfm_sum_forward_seconds': runtime,
              'path_oracle_vs_author_marginalization_mismatches': mismatches,
              'path_definition_diagnostic_scores': path_scores,
              'author_implementation_preserved': True, 'evaluated': evaluated,
              'note': 'Any path-oracle mismatch is reported, not used to replace frozen upstream scores.'}
    if evaluated:
        run.verify_manifest(run.RESULTS / 'prediction_freeze.json')
        assert (run.RESULTS / 'truth_open_receipt.json').exists()
        frame = pd.read_csv(run.RESULTS / 'per_run_results.csv')
        edges = pd.read_csv(run.ROOT / 'data/cyto_full_target.csv')
        truth = nx.DiGraph()
        truth.add_nodes_from(range(11))
        index = {h: i for i, h in enumerate(p['headers'])}
        truth.add_edges_from((index[s], index[t]) for s, t in edges.itertuples(index=False, name=None))
        truth.remove_edge(index['PIP2'], index['PIP3'])
        truth.add_edge(index['PIP3'], index['PIP2'])
        from src.causal_graphs.admg import ADMG
        for row in frame.to_dict('records'):
            z = np.load(run.RESULTS / 'inference' / row['run_id'] / 'full.npz')
            for i, prefix in enumerate(('auto', 'fixed')):
                g = run.graph(z['dags'][i], list(range(11)))
                assert g.shd(truth) == row[prefix + '_shd']
                assert abs(g.skeleton_f1(truth) - row[prefix + '_skeleton_f1']) < 1e-12
            receipt = read(run.RESULTS / 'judgments' / (row['run_id'] + '.json'))
            assert receipt['candidate_hashes'] == [adjacency_hash(a) for a in z['dags']]
            for key in receipt['request_hashes']:
                request = read(run.RESULTS / 'requests' / f'jev_{key}.json')
                import json
                state = json.loads(request['state'])
                assert set(state) == {'background', 'numerical_evidence', 'candidates'}
        result['reference_metric_checks'] = 42
        result['paired_candidate_hash_checks'] = 21
    result['audit_seconds'] = time.time() - started
    save(run.RESULTS / ('final_audit.json' if evaluated else 'inference_audit.json'), result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluated', action='store_true')
    main(parser.parse_args().evaluated)
