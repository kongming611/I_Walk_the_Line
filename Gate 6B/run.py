"""Stages: prepare -> score (only paid stage) -> evaluate -> deliver. No discovery runtime."""
import argparse
import csv
import hashlib
import time
import sys
import numpy as np
from core import *

def manifest(paths):
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}

def check(files):
    for name, sha in files.items():
        if digest(name) != sha:
            raise RuntimeError('Hash mismatch: '+name)

def verify():
    p = read(ROOT/'protocol.json')
    check(p['source_hashes'])
    check(p['input_hashes'])
    return p

def prepare():
    if (ROOT/'protocol.json').exists():
        verify()
        print('Protocol already frozen', flush=True)
        return
    # Validate original delivery before reusing any cached materials.
    receipt = read(OLD/'results/delivery_receipt.json')
    check({str(OLD/'results'/k): v for k,v in receipt['files'].items()})
    paths = [OLD/'protocol.json', OLD/'results/semantic_report.json', OLD/'results/api_ledger.json', OLD/'results/per_run_results.csv', OLD/'results/delivery_receipt.json']
    paths += [OLD/'results'/f for f in ('inference_freeze.json','prediction_freeze.json','final_audit.json')]
    for run in RUNS:
        paths += [OLD/'results/inference'/run/f for f in ('full.npz','evidence.json','sc_size6.json','sc_size5.json')]
    # Audit every available original freeze manifest using its recorded path convention.
    for filename in ('inference_freeze.json','prediction_freeze.json'):
        rec = read(OLD/'results'/filename)
        files = rec.get('files', {})
        for name, sha in files.items():
            from pathlib import Path
            target = Path(name) if Path(name).is_absolute() else OLD/name
            if not target.exists():
                target = OLD/'results'/name
            if digest(target) != sha:
                raise RuntimeError('Original freeze mismatch: '+name)
    p = {'created_at':time.time(), 'design':'exploratory post-hoc on one known Sachs system',
         'runs':RUNS, 'headers':read(OLD/'protocol.json')['headers'], 'configs':CONFIGS,
         'primary':'MAE of leave-one-run-out predictions on 40 repeat records',
         'calibration':'unconstrained univariate OLS with intercept; exact constant fallback to training mean; clip [0,110]',
         'ties':1e-8, 'primary_sc_subset_size':6,'sensitivity_sc_subset_size':5,
         'model':'jev-1.13.0','max_success_requests':42,'cumulative_budget_usd':2,'max_attempts':3,
         'old_budget_charge_usd':sum(e['budget_charge_usd'] for e in read(OLD/'results/api_ledger.json')),
         'no_new_cdfm_or_deepseek':True, 'input_hashes':manifest(paths),
         'source_hashes':manifest(list(ROOT.glob('*.py'))+[ROOT/'prompts.json',ROOT/'README.md']),
         'references':['https://docs.typesafe.ai/primitives/score','https://proceedings.mlr.press/v238/faller24a.html','https://www.statlearning.com/']}
    save(ROOT/'protocol.json',p)
    print('Protocol frozen; cached inputs and sources hashed', flush=True)

def score():
    p=verify()
    if (RESULTS/'score_freeze.json').exists():
        frozen=read(RESULTS/'score_freeze.json')
        assert frozen['protocol_sha256']==digest(ROOT/'protocol.json')
        check(frozen['files'])
        print('All scores already frozen; no requests',flush=True)
        return
    from api import request
    report=read(OLD/'results/semantic_report.json')['report']
    question=read(ROOT/'prompts.json')
    paths=[]
    count=0
    for run in RUNS:
        directory=OLD/'results/inference'/run
        with np.load(directory/'full.npz', allow_pickle=False) as z:
            dags=z['dags']
        numerical=read(directory/'evidence.json')
        for j, config in enumerate(CONFIGS):
            body=payload(report,numerical,p['headers'],dags[j],question)
            response,key=request(body)
            a=validate_score(response,question['criteria'])
            dest=RESULTS/'scores'/f'{run}_{j}.json'
            record={'run_id':run,'config':config,'graph_sha256':graph_hash(dags[j]),'request_sha256':key,'answer':a}
            if dest.exists() and read(dest)!=record:
                raise RuntimeError('Prior score changed')
            save(dest,record)
            paths.extend([dest,RESULTS/'requests'/(key+'.json'),RESULTS/'api'/(key+'.json'),RESULTS/'api'/(key+'.raw')])
            count+=1
            print(f'Scored {count}/42: {run} candidate {j}',flush=True)
    verify()
    paths.append(RESULTS/'api_ledger.json')
    save(RESULTS/'score_freeze.json',{'created_at':time.time(),'protocol_sha256':digest(ROOT/'protocol.json'),'files':manifest(paths),'count':count})
    print('All 42 scores frozen; truth table may now be parsed',flush=True)

def evaluate():
    verify()
    frozen=read(RESULTS/'score_freeze.json')
    assert frozen['count']==42 and frozen['protocol_sha256']==digest(ROOT/'protocol.json')
    check(frozen['files'])
    if not (RESULTS/'truth_open_receipt.json').exists():
        save(RESULTS/'truth_open_receipt.json',{'created_at':time.time(),'score_freeze_sha256':digest(RESULTS/'score_freeze.json'),'source_sha256':digest(OLD/'results/per_run_results.csv'),'note':'Source truth was already public in previous experiment; isolation applies to new scorer only'})
    with (OLD/'results/per_run_results.csv').open(newline='',encoding='utf-8-sig') as f:
        labels={r['run_id']:r for r in csv.DictReader(f)}
    records=[]
    for run in RUNS:
        for j, config in enumerate(CONFIGS):
            score_record=read(RESULTS/'scores'/f'{run}_{j}.json')
            prefix='auto' if j==0 else 'fixed'
            sc6=read(OLD/'results/inference'/run/'sc_size6.json')['scores'][config]
            sc5=read(OLD/'results/inference'/run/'sc_size5.json')['scores'][config]
            assert sc6==float(labels[run][prefix+'_kappa6']) and sc5==float(labels[run][prefix+'_kappa5'])
            records.append({'run_id':run,'config':config,'graph_sha256':score_record['graph_sha256'],'truth_shd':float(labels[run][prefix+'_shd']),
                            'sc6_score':sc6,'sc5_score':sc5,'jev_score':score_record['answer']['score'],'jev_confidence':score_record['answer']['confidence']})
    models=[]
    for group,train,test in splits(records):
        ys=[records[i]['truth_shd'] for i in train]
        fold={'held_out_group':group,'training_groups':sorted(set(records[i]['run_id'] for i in train)),'training_count':len(train),'models':{}}
        for name in ('sc6','sc5','jev'):
            m=fit([records[i][name+'_score'] for i in train],ys)
            fold['models'][name]=m
            for i in test:
                records[i][name+'_prediction']=float(predict(m,records[i][name+'_score']))
        for i in test:
            records[i]['mean_prediction']=float(np.mean(ys))
            records[i]['median_prediction']=float(np.median(ys))
            for name in ('sc6','sc5','jev','mean','median'):
                records[i][name+'_absolute_error']=abs(records[i][name+'_prediction']-records[i]['truth_shd'])
            records[i]['jev_minus_sc6_error']=records[i]['jev_absolute_error']-records[i]['sc6_absolute_error']
            records[i]['winner']=outcome(records[i]['jev_minus_sc6_error'])
        models.append(fold)
    summary={}
    for subset in ('repeats','auto','fixed_0.5','full'):
        rs=[r for r in records if (r['run_id']=='full' if subset=='full' else r['run_id']!='full' and (subset=='repeats' or r['config']==subset))]
        summary[subset]={name:metrics([r['truth_shd'] for r in rs],[r[name+'_prediction'] for r in rs]) for name in ('sc6','jev','sc5','mean','median')}
        summary[subset]['graph_wins']={w:sum(r['winner']==w for r in rs) for w in ('jev_win','tie','sc_win')}
    group_results=[]
    for run in RUNS:
        rs=[r for r in records if r['run_id']==run]
        delta=float(np.mean([r['jev_minus_sc6_error'] for r in rs]))
        group_results.append({'run_id':run,'jev_minus_sc6_mae':delta,'winner':outcome(delta)})
    summary['repeat_group_wins']={w:sum(r['winner']==w for r in group_results if r['run_id']!='full') for w in ('jev_win','tie','sc_win')}
    write_csv(RESULTS/'per_graph_predictions.csv',records)
    write_csv(RESULTS/'per_group_errors.csv',group_results)
    save(RESULTS/'fold_models.json',models)
    save(RESULTS/'evaluation.json',{'summary':summary,'records':records,'groups':group_results,'score_freeze_sha256':digest(RESULTS/'score_freeze.json')})
    print(json_summary(summary),flush=True)

def json_summary(x):
    import json
    return json.dumps(x,ensure_ascii=False,indent=2)

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','score','evaluate'])
    args=parser.parse_args()
    try:
        globals()[args.stage]()
    except Exception as exc:
        save(RESULTS/'last_failure.json',{'stage':args.stage,'type':type(exc).__name__,'message':str(exc),'time':time.time()})
        raise
