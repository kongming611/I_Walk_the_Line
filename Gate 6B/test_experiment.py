import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import api
from core import *

class Contracts(unittest.TestCase):
    def answer(self):
        criteria=read(ROOT/'prompts.json')['criteria']
        return {'model':'jev-1.13.0','answers':{'structural_error':{'type':'score','score':2.,'confidence':1.,'probabilities':{str(i):float(i==2) for i in range(5)},'legend':dict(enumerate(criteria))}}}

    def valid(self):
        r=self.answer();a=r['answers']['structural_error'];a['legend']={str(k):v for k,v in a['legend'].items()};return r

    def test_ols_exact(self):
        m=fit([0,1,2],[3,5,7]);self.assertAlmostEqual(m['intercept'],3);self.assertAlmostEqual(m['slope'],2)
    def test_constant(self):
        m=fit([2,2,2],[1,4,7]);self.assertTrue(m['constant_fallback']);self.assertEqual(float(predict(m,100)),4)
    def test_clip(self):
        self.assertEqual(predict({'intercept':0,'slope':1},[-1,111]).tolist(),[0,110])
    def test_negative_slope_permitted(self):
        self.assertAlmostEqual(fit([0,1,2],[7,5,3])['slope'],-2)
    def test_metrics(self):
        m=metrics([2,4],[3,1]);self.assertEqual(m['mae'],2);self.assertAlmostEqual(m['rmse'],np.sqrt(5));self.assertEqual(m['bias'],-1)
    def test_ties(self):
        self.assertEqual(outcome(1e-8),'tie');self.assertEqual(outcome(-.1),'jev_win');self.assertEqual(outcome(.1),'sc_win')
    def test_group_splits(self):
        rs=[{'run_id':g} for g in RUNS for _ in CONFIGS]
        for group,tr,te in splits(rs):
            self.assertFalse(set(tr)&set(te));self.assertTrue(all(rs[i]['run_id'] not in (group,'full') for i in tr));self.assertEqual(len(te),2)
    def test_heldout_label_invariance(self):
        rs=[{'run_id':g,'x':i,'y':i+1} for i,g in enumerate(RUNS) for _ in CONFIGS]
        group,tr,te=next(splits(rs));m=fit([rs[i]['x'] for i in tr],[rs[i]['y'] for i in tr])
        for i in te: rs[i]['y']=1e9
        self.assertEqual(m,fit([rs[i]['x'] for i in tr],[rs[i]['y'] for i in tr]))
    def test_valid_score(self):
        self.assertEqual(validate_score(self.valid(),read(ROOT/'prompts.json')['criteria'])['score'],2)
    def test_bad_score(self):
        for value in (float('nan'),float('inf'),-1,4.1,True):
            r=self.valid();r['answers']['structural_error']['score']=value
            with self.assertRaises(ValueError): validate_score(r,read(ROOT/'prompts.json')['criteria'])
    def test_bad_probabilities(self):
        r=self.valid();r['answers']['structural_error']['probabilities']['0']=1
        with self.assertRaises(ValueError):validate_score(r,read(ROOT/'prompts.json')['criteria'])
    def test_bad_version(self):
        r=self.valid();r['model']='other'
        with self.assertRaises(ValueError):validate_score(r,read(ROOT/'prompts.json')['criteria'])
    def test_whitelist(self):
        q=read(ROOT/'prompts.json');report=read(OLD/'results/semantic_report.json')['report'];ev=read(OLD/'results/inference/full/evidence.json')
        report['truth_shd']='SECRET_LABEL';ev['kappa']='SECRET_LABEL';ev['variables'][0]['truth_shd']='SECRET_LABEL'
        p=payload(report,ev,['a','b'],np.array([[0,1],[0,0]]),q)
        self.assertNotIn('SECRET_LABEL',p['state']);s=__import__('json').loads(p['state']);self.assertEqual(set(s),{'background','candidate','numerical_evidence'});self.assertEqual(s['candidate']['edges'],[['a','b']])
    def test_cache_no_network_or_credentials(self):
        import hashlib
        p={'model':'jev-1.13.0','state':'test','questions':{'structural_error':read(ROOT/'prompts.json')}};key=hashlib.sha256(canonical(p)).hexdigest()
        with tempfile.TemporaryDirectory() as d, patch.object(api,'RESULTS',Path(d)), patch.object(api,'credential',side_effect=AssertionError('credential called')), patch.object(api.urllib.request,'urlopen',side_effect=AssertionError('network called')):
            raw=Path(d)/'api'/(key+'.raw');raw.parent.mkdir();raw.write_bytes(canonical(self.valid()))
            save(raw.with_suffix('.json'),{'request_sha256':key,'response':self.valid(),'response_sha256':digest(raw),'raw_sha256':digest(raw)})
            self.assertEqual(api.request(p)[1],key)
    def test_budget_blocks_network(self):
        p={'model':'jev-1.13.0','state':'x','questions':{'x':{'type':'score'}}}
        with tempfile.TemporaryDirectory() as d, patch.object(api,'RESULTS',Path(d)), patch.object(api,'credential',return_value='test'), patch.object(api,'old_charge',return_value=2), patch.object(api.urllib.request,'urlopen',side_effect=AssertionError('network called')):
            with self.assertRaisesRegex(RuntimeError,'budget'): api.request(p)
    def test_freeze_blocks_new_request(self):
        p={'model':'jev-1.13.0','state':'x','questions':{'x':{'type':'score'}}}
        with tempfile.TemporaryDirectory() as d, patch.object(api,'RESULTS',Path(d)), patch.object(api,'credential',side_effect=AssertionError('credential called')):
            save(Path(d)/'score_freeze.json',{})
            with self.assertRaisesRegex(RuntimeError,'frozen'):api.request(p)
    def test_forbidden_model(self):
        with self.assertRaises(ValueError):api.request({'model':'deepseek-flash'})

if __name__=='__main__':unittest.main()
