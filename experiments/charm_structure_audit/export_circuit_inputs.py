"""Export the trained node MLP and saved paired inputs. No model execution."""

import io
import json
from pathlib import Path
import tarfile

import numpy as np


FIELDS = ('error_x', 'normal_x', 'logits', 'baseline_score',
          'tokens', 'text', 'threshold')


def pair_bytes(path):
    """Keep raw node inputs and predictions; leave large audit arrays on disk."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in FIELDS}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def export_circuit_inputs(root, destination):
    """Follow the completed whitebox manifest, including its actual checkpoint."""
    audit = Path(root) / 'audit_whitebox'
    manifest = json.loads((audit / 'manifest.json').read_text(encoding='utf-8'))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')

    with tarfile.open(temporary, 'w:gz') as archive:
        archive.add(manifest['checkpoint'][0], arcname='node_only/checkpoint.pt')
        archive.add(audit / 'manifest.json', arcname='manifest.json')
        archive.add(audit / 'geometry.json', arcname='geometry.json')
        for pair in manifest['pairs']:
            payload = pair_bytes(audit / 'captures' / pair['file'])
            entry = tarfile.TarInfo('inputs/' + pair['file'])
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))

    temporary.replace(destination)
    print(f"Exported {len(manifest['pairs'])} paired inputs and the trained checkpoint")
    print(f'{destination.resolve()} ({destination.stat().st_size / 1024**2:.2f} MiB)')
