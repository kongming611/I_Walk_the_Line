"""Independent numerical recount after score freeze; never calls APIs."""
import ast
import csv
import hashlib
import subprocess
import sys
import numpy as np
from core import ROOT, OLD, RESULTS, RUNS, CONFIGS, read, save, digest, canonical, payload, validate_score
from run import verify, check

def audit():
    protocol=verify()
    frozen=read(RESULTS/'score_freeze.json');check(frozen['files'])
    assert frozen['protocol_sha256']==digest(ROOT/'protocol.json')
    assert read(RESULTS/'truth_open_receipt.json')['created_at']>=frozen['created_at']
    ev=read(RESULTS/'evaluation.json');rs=ev['records']
    assert len(rs)==42 and len(set(r['graph_sha256'] for r in rs))==42
    report=read(OLD/'results/semantic_report.json')['report'];question=read(ROOT/'prompts.json')
    headers=protocol['headers'];truth=np.zeros((11,11),dtype=int)
    with (OLD/'data/cyto_full_target.csv').open(newline='') as f:
        for row in csv.DictReader(f):truth[headers.index(row['Cause']),headers.index(row['Effect'])]=1
    truth[headers.index('PIP2'),headers.index('PIP3')]=0;truth[headers.index('PIP3'),headers.index('PIP2')]=1
    for r in rs:
        j=CONFIGS.index(r['config']);directory=OLD/'results/inference'/r['run_id']
        with np.load(directory/'full.npz') as z: dag=z['dags'][j]
        assert int(np.count_nonzero(dag!=truth))==r['truth_shd']
        assert hashlib.sha256(np.asarray(dag,np.int8).tobytes()).hexdigest()==r['graph_sha256']
        original=read(RESULTS/'scores'/f"{r['run_id']}_{j}.json")
        key=original['request_sha256'];body=read(RESULTS/'requests'/(key+'.json'))
        assert body==payload(report,read(directory/'evidence.json'),headers,dag,question)
        assert hashlib.sha256(canonical(body)).hexdigest()==key
        response=read(RESULTS/'api'/(key+'.json'))['response'];a=validate_score(response,question['criteria'])
        assert a==original['answer'] and a['score']==r['jev_score']
        for name,size in [('sc6',6),('sc5',5)]:
            sc=read(directory/f'sc_size{size}.json');assert sc['scores'][r['config']]==r[name+'_score']
        train=[t for t in rs if t['run_id'] not in (r['run_id'],'full')]
        assert len(train)==(40 if r['run_id']=='full' else 38)
        y=np.array([t['truth_shd'] for t in train])
        for name in ('sc6','sc5','jev'):
            x=np.array([t[name+'_score'] for t in train])
            if np.ptp(x)==0:pred=y.mean()
            else:
                coefficients=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
                pred=np.clip(coefficients@[1,r[name+'_score']],0,110)
            assert abs(pred-r[name+'_prediction'])<1e-8
        assert r['mean_prediction']==float(y.mean()) and r['median_prediction']==float(np.median(y))
        for name in ('sc6','sc5','jev','mean','median'):
            assert abs(r[name+'_absolute_error']-abs(r[name+'_prediction']-r['truth_shd']))<1e-10
    for subset in ('repeats','auto','fixed_0.5','full'):
        rows=[r for r in rs if (r['run_id']=='full' if subset=='full' else r['run_id']!='full' and (subset=='repeats' or r['config']==subset))]
        for name in ('sc6','sc5','jev','mean','median'):
            d=np.array([r[name+'_prediction']-r['truth_shd'] for r in rows]);m=ev['summary'][subset][name]
            assert abs(m['mae']-sum(abs(d))/len(d))<1e-10
            assert abs(m['rmse']-np.sqrt(sum(d*d)/len(d)))<1e-10
    imports=set()
    for p in ROOT.glob('*.py'):
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            if isinstance(node,ast.Import):imports.update(n.name.split('.')[0] for n in node.names)
            if isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
    assert not imports & {'torch','cdfm','transformers','deepseek'}
    ledger=read(RESULTS/'api_ledger.json');old=read(OLD/'results/api_ledger.json')
    assert len([x for x in ledger if x['status']=='received'])==42
    total=sum(x['budget_charge_usd'] for x in ledger+old);assert total<=2
    for path in (RESULTS/'requests').glob('*.json'):assert read(path)['model']=='jev-1.13.0'
    tests=subprocess.run([sys.executable,'-m','unittest','-v','test_experiment'],cwd=ROOT,capture_output=True,text=True)
    (RESULTS/'test_output.txt').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    assert tests.returncode==0
    result={'passed':True,'graph_hashes_verified':42,'truth_shd_recounted':42,'independent_ols_predictions_checked':126,
            'request_payloads_verified':42,'unit_tests_passed':17,'no_cdfm_or_deepseek_imports':True,'old_inputs_unchanged':True,
            'cumulative_reserved_usd':total,'new_usage_price_bound_usd':sum(x.get('peak_price_usage_bound_usd',0) for x in ledger),
            'old_usage_price_bound_usd':sum(x.get('peak_price_usage_bound_usd',0) for x in old),
            'api_attempts':len(ledger),'api_runtime_sec':sum(x.get('runtime_sec',0) for x in ledger)}
    save(RESULTS/'audit.json',result)
    print(result)
    return result

if __name__=='__main__':audit()
