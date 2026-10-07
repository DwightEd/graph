"""Stream full cache content hashes; excludes mutable evaluation reports and manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def inventory(root):
    inputs = json.loads((root / 'inputs.json').read_text())
    paths = [root / name for name in ('inputs.json', 'execution.json', 'axes.npy',
              'projection.json', 'capture_freeze.json', 'reproject_code.py') if (root / name).exists()]
    paths += sorted((root / 'capture_code').glob('*'))
    for case in inputs['cases']:
        paths += sorted(p for p in (root / case['id']).iterdir() if p.is_file())
    return paths


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


def manifest(root, verify=False):
    paths = inventory(root)
    with ThreadPoolExecutor(max_workers=8) as executor:
        files = {str(p.relative_to(root)): v for p, v in zip(paths, executor.map(file_hash, paths))}
    path = root / 'content_manifest.json'
    if verify:
        previous = json.loads(path.read_text())
        assert previous['files'] == files, 'Cache content or inventory changed'
    else:
        assert not path.exists(), 'Never overwrite established content freeze'
        path.write_text(json.dumps(dict(established=datetime.now(timezone.utc).isoformat(),
            scope='retrospective after original fits; before audit-added control', files=files), indent=2))
    print(json.dumps(dict(root=str(root), verified=verify, files=len(files),
                         bytes=sum(v['bytes'] for v in files.values()))), flush=True)
    return file_hash(path)['sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', type=Path, nargs='+', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for root in args.caches:
        manifest(root, args.verify)
