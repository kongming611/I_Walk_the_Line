"""Task-local paid API transport with encrypted credentials and durable budget reservations."""
import hashlib
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from core import canonical, read, save, digest

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RATES = {'deepseek_input': .30, 'deepseek_output': 1.20, 'jev_input': .042}


def credential(name):
    value = os.environ.get(name)
    if value:
        return value
    path = ROOT / '.credentials.xml'
    if not path.exists():
        raise RuntimeError('Missing task credentials; run configure_credentials.ps1 locally.')
    # Only decrypt the requested task credential into this process, never tool output/logs.
    if name not in ('DEEPSEEK_API_KEY', 'TYPESAFE_API_KEY'):
        raise ValueError('Unknown credential name')
    import base64
    script = "$ErrorActionPreference='Stop'; $v=Import-Clixml -LiteralPath '" + str(path).replace("'", "''") + "'; $s=$v['" + name + "']; if($null -eq $s){exit 2}; [System.Net.NetworkCredential]::new('', $s).Password"
    encoded = base64.b64encode(script.encode('utf-16-le')).decode('ascii')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-EncodedCommand', encoded], capture_output=True)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Cannot decrypt task credential under this Windows user.')
    return result.stdout.decode('utf-8').strip()


def ledger():
    p = RESULTS / 'api_ledger.json'
    return read(p) if p.exists() else []


def charge_bound(kind, body):
    # UTF-8 byte count is conservative for input token count; output is API-capped.
    if kind == 'deepseek':
        return (len(body) * 2 * RATES['deepseek_input'] + 4096 * RATES['deepseek_output']) / 1e6
    return len(body) * 2 * RATES['jev_input'] / 1e6


def request(kind, payload):
    body = canonical(payload)
    if len(body) > 30000:
        raise RuntimeError('Request exceeds frozen 30000-byte conservative input limit; no truncation or call.')
    key_hash = hashlib.sha256(body).hexdigest()
    cache = RESULTS / 'api' / f'{kind}_{key_hash}.json'
    if cache.exists():
        saved = read(cache)
        if saved['request_sha256'] != key_hash or saved['response_sha256'] != hashlib.sha256(canonical(saved['response'])).hexdigest():
            raise RuntimeError('API cache integrity failure')
        return saved['response'], key_hash
    if (RESULTS / 'prediction_freeze.json').exists() or (RESULTS / 'evaluation.json').exists():
        raise RuntimeError('Predictions already frozen; no new API calls allowed.')
    name = 'DEEPSEEK_API_KEY' if kind == 'deepseek' else 'TYPESAFE_API_KEY'
    secret = credential(name)
    url = 'https://api.deepseek.com/chat/completions' if kind == 'deepseek' else 'https://api.typesafe.ai/v1/systemone'
    save(RESULTS / 'requests' / f'{kind}_{key_hash}.json', payload)
    entries = ledger()
    previous = [e for e in entries if e['request_sha256'] == key_hash]
    if any(e['status'] in ('reserved', 'received', 'nonretryable') for e in previous):
        raise RuntimeError('Unresolved or nonretryable prior API attempt; manual audit required, no duplicate charge.')
    for attempt in range(len(previous), 3):
        reserve = charge_bound(kind, body)
        if sum(e['budget_charge_usd'] for e in entries) + reserve > 2.:
            raise RuntimeError('Frozen $2 conservative API budget would be exceeded.')
        item = {'kind': kind, 'request_sha256': key_hash, 'attempt': attempt + 1, 'status': 'reserved',
                'reserved_usd': reserve, 'budget_charge_usd': reserve, 'time': time.time()}
        entries.append(item)
        save(RESULTS / 'api_ledger.json', entries)
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, body, {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret})
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            if secret.encode() in raw:
                raise RuntimeError('Credential reflected in response; response suppressed.')
            # Preserve original bytes before parsing; malformed success responses are not paid again.
            raw_path = RESULTS / 'api' / f'{kind}_{key_hash}.raw'
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
            import json
            value = json.loads(raw)
            usage = value.get('usage', {})
            if kind == 'deepseek':
                estimate = (usage.get('prompt_tokens', 2 * len(body)) * RATES['deepseek_input'] + usage.get('completion_tokens', 4096) * RATES['deepseek_output']) / 1e6
            else:
                estimate = usage.get('input_tokens', 2 * len(body)) * RATES['jev_input'] / 1e6
            item.update(status='received', runtime_sec=time.perf_counter() - t0, usage=usage,
                        peak_price_usage_bound_usd=estimate, budget_charge_usd=max(reserve, estimate))
            save(RESULTS / 'api_ledger.json', entries)
            save(cache, {'request_sha256': key_hash, 'response': value, 'response_sha256': hashlib.sha256(canonical(value)).hexdigest(),
                         'raw_response_sha256': digest(raw_path)})
            return value, key_hash
        except urllib.error.HTTPError as exc:
            retryable = exc.code in (429, 500, 502, 503, 504, 529)
            item.update(status='retryable' if retryable else 'nonretryable', error='http_' + str(exc.code))
            save(RESULTS / 'api_ledger.json', entries)
            if not retryable:
                raise RuntimeError('API rejected request: HTTP ' + str(exc.code) + '; details suppressed.') from None
        except (TimeoutError, urllib.error.URLError) as exc:
            item.update(status='retryable', error=type(exc).__name__)
            save(RESULTS / 'api_ledger.json', entries)
        if attempt < 2:
            time.sleep(2 ** attempt)
    raise RuntimeError('API attempts exhausted; no further automatic paid retries.')


def validate_report(response, headers):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('DeepSeek output incomplete')
    import json
    report = json.loads(choice['message']['content'])
    if set(report) != {'research_context', 'variables', 'limitations'}:
        raise ValueError('Unexpected semantic report fields')
    if not isinstance(report['research_context'], str) or not isinstance(report['limitations'], list):
        raise ValueError('Invalid semantic report structure')
    if [v['name'] for v in report['variables']] != headers:
        raise ValueError('Report must retain exact input column order')
    for v in report['variables']:
        if set(v) != {'name', 'possible_meaning', 'measurement_interpretation', 'uncertainty'} or not all(isinstance(s, str) for s in v.values()):
            raise ValueError('Invalid variable interpretation')
    if not all(isinstance(v, str) for v in report['limitations']):
        raise ValueError('Invalid limitations')
    return report


def validate_choice(response):
    if response.get('model') != 'jev-1.13.0':
        raise ValueError('Unexpected Jev version')
    answer = response['answers']['better_graph']
    if answer.get('type') != 'choice' or answer.get('choice') not in ('A', 'B', 'unclear'):
        raise ValueError('Invalid Jev answer')
    probabilities = answer.get('probabilities', {})
    if set(probabilities) != {'A', 'B', 'unclear'}:
        raise ValueError('Missing Jev probabilities')
    import math
    values = list(probabilities.values())
    if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in values) or abs(sum(values) - 1) > .016:
        raise ValueError('Invalid Jev distribution')
    return answer['choice']
