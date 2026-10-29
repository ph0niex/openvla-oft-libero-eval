"""Offline integrity checks shared by doctor and evaluation launchers."""
import json
from pathlib import Path

SUITE_REPOS = {suite: f'moojink/openvla-7b-oft-finetuned-{suite.replace("_", "-")}'
               for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}


def check_checkpoint(path):
    path = Path(path)
    errors = []
    required = ['config.json', 'model.safetensors.index.json', 'dataset_statistics.json',
                'preprocessor_config.json', 'tokenizer_config.json', 'tokenizer.json']
    for name in required:
        if not (path / name).is_file():
            errors.append(f'Missing {name}')
    for component in ('action_head', 'proprio_projector'):
        if len(list(path.glob(f'{component}*checkpoint.pt'))) != 1:
            errors.append(f'Expected exactly one {component} checkpoint')
    shards = []
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        try:
            shards = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            if not shards:
                errors.append('Empty model weight map')
            for name in shards:
                if not (path / name).is_file() or (path / name).stat().st_size == 0:
                    errors.append(f'Missing/empty shard {name}')
        except (ValueError, KeyError) as exc:
            errors.append(f'Invalid model index: {exc}')
    manifest = path / 'download-manifest.json'
    provenance = None
    if manifest.is_file():
        try:
            provenance = json.loads(manifest.read_text())
            # Model loader intentionally syncs config/code. Validate immutable weights only here.
            for item in provenance.get('file_metadata', []):
                name = item['name']
                if name.endswith(('.safetensors', '.pt')):
                    f = path / name
                    if not f.is_file() or f.stat().st_size != item['size']:
                        errors.append(f'Weight size mismatch/missing: {name}')
            if not provenance.get('downloaded') or not provenance.get('sha256_verified'):
                errors.append('Download has not passed pinned SHA256 verification')
        except (ValueError, KeyError) as exc:
            errors.append(f'Invalid download manifest: {exc}')
    else:
        errors.append('No pinned download-manifest.json; run checkpoint.py --verify-local')
    return {'path': str(path.resolve()), 'complete': not errors, 'errors': errors,
            'shards': shards, 'repo_id': (provenance or {}).get('repo_id'),
            'revision': (provenance or {}).get('revision')}
