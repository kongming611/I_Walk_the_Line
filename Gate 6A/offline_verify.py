"""Portable read-only archive verification. No model/API/credential calls."""
from pathlib import Path
import csv, json, hashlib, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
CONFIGS=['auto','fixed_0.5']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    package=read(ROOT/'ARCHIVE_MANIFEST.json')
    for name,expected in package['files'].items():
        if sha(ROOT/name)!=expected:raise RuntimeError('Archive hash mismatch: '+name)
    origin=read(ROOT/'archive_provenance.json')
    for name,record in origin['copied_files'].items():assert sha(ROOT/name)==record['sha256']
    samples=read(ROOT/'data/sampling_row_indices.json');headers=samples['headers']
    with (ROOT/'data/cyto_full_data.csv').open(newline='',encoding='utf-8-sig') as f: data=list(csv.reader(f))
    assert data[0]==headers and len(data)==7467
    for run in samples['runs']:
        if run['run_id']=='full':continue
        with (ROOT/'data/samples'/(run['run_id']+'.csv')).open(newline='',encoding='utf-8-sig') as f:sub=list(csv.reader(f))
        assert sub==[headers]+[data[i+1] for i in run['row_indices']]
    semantic=read(ROOT/'analysis/semantic_report.json')
    key=semantic['request_sha256'];request=read(ROOT/'analysis/requests'/('deepseek_'+key+'.json'))
    canonical=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
    assert hashlib.sha256(canonical).hexdigest()==key
    response=read(ROOT/'analysis/api'/('deepseek_'+key+'.json'))['response']
    assert json.loads(response['choices'][0]['message']['content'])==semantic['report']
    truth=np.zeros((11,11),dtype=int)
    for row in csvrows(ROOT/'data/cyto_full_target.csv'):
        truth[headers.index(row['Cause']),headers.index(row['Effect'])]=1
    assert truth[headers.index('PIP2'),headers.index('PIP3')]==1
    truth[headers.index('PIP2'),headers.index('PIP3')]=0;truth[headers.index('PIP3'),headers.index('PIP2')]=1
    checked=0
    if origin['gate']=='Gate 6A':
        rows=csvrows(ROOT/'results/per_run_results.csv')
        for row in rows:
            folder=ROOT/'results/inference'/row['run_id']
            with np.load(folder/'full.npz',allow_pickle=False) as z:dags=z['dags']
            for i,prefix in enumerate(('auto','fixed')):
                assert np.count_nonzero(dags[i]!=truth)==float(row[prefix+'_shd'])
                a=(dags[i]+dags[i].T)>0;b=(truth+truth.T)>0
                mask=np.triu(np.ones((11,11),bool),1)
                tp=np.count_nonzero(a&b&mask);fp=np.count_nonzero(a&~b&mask);fn=np.count_nonzero(~a&b&mask)
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.
                assert abs(f1-float(row[prefix+'_skeleton_f1']))<1e-10
                checked+=1
            for size in (6,5):
                sc=read(folder/f'sc_size{size}.json')
                for prefix,config in zip(('auto','fixed'),CONFIGS):
                    assert len(sc['distances'][config])==40
                    assert abs(np.mean(sc['distances'][config])-sc['scores'][config])<1e-10
                    assert float(row[f'{prefix}_kappa{size}'])==sc['scores'][config]
                selected=CONFIGS[0 if abs(sc['scores']['auto']-sc['scores']['fixed_0.5'])<=1e-12 or sc['scores']['auto']<sc['scores']['fixed_0.5'] else 1]
                assert selected==row[f'sc{size}_selected']
            judgment=read(ROOT/'results/judgments'/(row['run_id']+'.json'))
            f,r=judgment['choices_forward_reverse']
            a={'A':'auto','B':'fixed_0.5','unclear':None}[f];b={'A':'fixed_0.5','B':'auto','unclear':None}[r]
            selected=a if a is not None and a==b else 'auto'
            assert selected==row['jev_selected']
        summary={'sc6_mean_shd':float(np.mean([float(r['sc6_shd']) for r in rows if r['run_id']!='full'])),
                 'jev_mean_shd':float(np.mean([float(r['jev_shd']) for r in rows if r['run_id']!='full']))}
    else:
        ev=read(ROOT/'results/evaluation.json');rows=ev['records']
        for row in rows:
            folder=ROOT/'data/candidates'/row['run_id']
            with np.load(folder/'full.npz',allow_pickle=False) as z:dag=z['dags'][CONFIGS.index(row['config'])]
            assert np.count_nonzero(dag!=truth)==row['truth_shd']
            assert hashlib.sha256(np.asarray(dag,np.int8).tobytes()).hexdigest()==row['graph_sha256']
            for name,size in [('sc6',6),('sc5',5)]:assert read(folder/f'sc_size{size}.json')['scores'][row['config']]==row[name+'_score']
            original=read(ROOT/'results/scores'/f"{row['run_id']}_{CONFIGS.index(row['config'])}.json")
            assert original['answer']['score']==row['jev_score']
            train=[t for t in rows if t['run_id'] not in ('full',row['run_id'])]
            assert len(train)==(40 if row['run_id']=='full' else 38)
            y=np.array([t['truth_shd'] for t in train])
            for name in ('sc6','sc5','jev'):
                x=np.array([t[name+'_score'] for t in train])
                if np.ptp(x)==0:p=y.mean()
                else:
                    m=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x]),y,rcond=None)[0]
                    p=np.clip(m@[1,row[name+'_score']],0,110)
                assert abs(p-row[name+'_prediction'])<1e-8
            assert row['mean_prediction']==y.mean() and row['median_prediction']==np.median(y)
            checked+=1
        for subset in ('repeats','auto','fixed_0.5','full'):
            rs=[r for r in rows if (r['run_id']=='full' if subset=='full' else r['run_id']!='full' and (subset=='repeats' or r['config']==subset))]
            for name in ('sc6','sc5','jev','mean','median'):
                e=np.array([r[name+'_prediction']-r['truth_shd'] for r in rs]);m=ev['summary'][subset][name]
                assert abs(np.mean(abs(e))-m['mae'])<1e-8 and abs(np.sqrt(np.mean(e*e))-m['rmse'])<1e-8
        summary={name:ev['summary']['repeats'][name]['mae'] for name in ('sc6','jev','sc5','mean','median')}
    print(json.dumps({'gate':origin['gate'],'passed':True,'verified_archive_files':len(package['files']),'verified_original_copies':len(origin['copied_files']),
                      'verified_sample_csvs':20,'verified_graphs':checked,'deepseek_report_matches_original_response':True,'summary':summary,
                      'scope':'offline cached-result verification; no new discovery or API calls; original marginalization audits preserved'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
