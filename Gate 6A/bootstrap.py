"""Acquire public upstream data; copy pristine author source and packaged dependencies."""
from pathlib import Path
import shutil
import subprocess
import urllib.request
from core import digest, save

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMIT = '9084041f9c1e238250ce281868c4f6d2971d5e0c'


def main():
    source = REPO / 'gate8_self_compatibility/vendor/causal-self-compatibility'
    git = ['git', '-c', 'safe.directory=' + source.as_posix(), '-C', str(source)]
    if subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise RuntimeError('Upstream source commit mismatch')
    if subprocess.check_output(git + ['status', '--porcelain'], text=True).strip():
        raise RuntimeError('Upstream source not pristine')
    target = ROOT / 'vendor/causal-self-compatibility'
    if not target.exists():
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    dependencies = REPO / 'gate8_self_compatibility/vendor/python_deps'
    destination = ROOT / 'vendor/python_deps'
    if not destination.exists():
        shutil.copytree(dependencies, destination, ignore=shutil.ignore_patterns('__pycache__'))
    # Only software is copied; no prior experiment data, prompts, or results.
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    downloads = {}
    for filename in ('cyto_full_data.csv', 'cyto_full_target.csv'):
        url = 'https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/master/cdt/data/resources/' + filename
        p = data / filename
        if not p.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if len(content) < 100:
                raise RuntimeError('Unexpected data response')
            p.write_bytes(content)
        downloads[filename] = {'url': url, 'sha256': digest(p), 'bytes': p.stat().st_size}
    save(ROOT / 'data/provenance.json', {'upstream_commit': COMMIT, 'downloads': downloads,
         'note': 'Reference network downloaded as opaque bytes; parsed only by evaluate. Author source copied from verified clean Git checkout. No old experiment outputs used.'})
    print('Public data and pristine software acquired; reference network not parsed.', flush=True)


if __name__ == '__main__':
    main()
