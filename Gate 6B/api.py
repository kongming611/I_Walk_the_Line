"""Jev-only transport with task DPAPI credentials and cumulative durable budget."""
import base64
import hashlib
import os
import subprocess
import time
import urllib.error
import urllib.request
import json
from core import OLD, RESULTS, canonical, digest, read, save

RATE = .042

def credential():
    if os.environ.get('TYPESAFE_API_KEY'):
        return os.environ['TYPESAFE_API_KEY']
    path = OLD / '.credentials.xml'
    script = "$ErrorActionPreference='Stop'; $v=Import-Clixml -LiteralPath '" + str(path).replace("'", "''") + "'; $s=$v['TYPESAFE_API_KEY']; if($null -eq $s){exit 2}; [System.Net.NetworkCredential]::new('', $s).Password"
    code = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    r = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', code], capture_output=True)
    if r.returncode or not r.stdout.strip():
        raise RuntimeError('Cannot decrypt task credential under current Windows user')
    return r.stdout.decode('utf-8').strip()

def ledger():
    p = RESULTS / 'api_ledger.json'
    return read(p) if p.exists() else []

def old_charge():
    return sum(e['budget_charge_usd'] for e in read(OLD / 'results/api_ledger.json'))

def request(payload):
    if payload.get('model') != 'jev-1.13.0' or any(q.get('type') != 'score' for q in payload.get('questions', {}).values()):
        raise ValueError('Only frozen Jev Score requests allowed')
    body = canonical(payload)
    if len(body) > 30000:
        raise RuntimeError('Payload exceeds byte limit; no truncation')
    key = hashlib.sha256(body).hexdigest()
    cache = RESULTS / 'api' / (key + '.json')
    if cache.exists():
        c = read(cache)
        if c['request_sha256'] != key or c['response_sha256'] != hashlib.sha256(canonical(c['response'])).hexdigest() or c['raw_sha256'] != digest(cache.with_suffix('.raw')):
            raise RuntimeError('Corrupt API cache')
        return c['response'], key
    if (RESULTS/'score_freeze.json').exists() or (RESULTS/'evaluation.json').exists():
        raise RuntimeError('Scores frozen: no new paid requests')
    entries = ledger()
    previous = [e for e in entries if e['request_sha256'] == key]
    if any(e['status'] != 'retryable' for e in previous):
        raise RuntimeError('Unresolved or nonretryable request; no duplicate charge')
    secret = credential()
    save(RESULTS/'requests'/(key+'.json'), payload)
    for attempt in range(len(previous), 3):
        reserve = 2*len(body)*RATE/1e6
        if old_charge()+sum(e['budget_charge_usd'] for e in entries)+reserve > 2:
            raise RuntimeError('Cumulative $2 budget exhausted')
        item = {'request_sha256':key,'attempt':attempt+1,'status':'reserved','budget_charge_usd':reserve,'reserved_usd':reserve,'time':time.time()}
        entries.append(item)
        save(RESULTS/'api_ledger.json', entries)
        start = time.perf_counter()
        try:
            req = urllib.request.Request('https://api.typesafe.ai/v1/systemone', body, {'Content-Type':'application/json','Authorization':'Bearer '+secret})
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            if secret.encode() in raw:
                raise RuntimeError('Credential reflection suppressed')
            raw_path = cache.with_suffix('.raw')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            value = json.loads(raw)
            usage = value.get('usage', {})
            charge = usage.get('input_tokens', 2*len(body))*RATE/1e6
            item.update(status='received', runtime_sec=time.perf_counter()-start, usage=usage, peak_price_usage_bound_usd=charge, budget_charge_usd=max(reserve,charge))
            save(RESULTS/'api_ledger.json', entries)
            save(cache, {'request_sha256':key,'response':value,'response_sha256':hashlib.sha256(canonical(value)).hexdigest(),'raw_sha256':digest(raw_path)})
            return value, key
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429,500,502,503,504,529)
            item.update(status='retryable' if retryable else 'nonretryable', error='http_'+str(exc.code))
            save(RESULTS/'api_ledger.json', entries)
            if not retryable:
                raise RuntimeError('API HTTP '+str(exc.code)+'; details suppressed') from None
        except (TimeoutError, urllib.error.URLError) as exc:
            item.update(status='retryable', error=type(exc).__name__)
            save(RESULTS/'api_ledger.json', entries)
        if attempt < 2:
            time.sleep(2**attempt)
    raise RuntimeError('Retry limit exhausted')
