"""Inspect, download via hf, or verify ONE official checkpoint pinned to a Hub commit."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from huggingface_hub import HfApi
from checkpoint_utils import SUITE_REPOS

ROOT = Path(__file__).resolve().parents[1]


def verify_files(target, items):
    errors = []
    for item in items:
        path = target / item['name']
        if not path.is_file() or path.stat().st_size != item['size']:
            errors.append(f'Missing/wrong size: {item["name"]}')
            continue
        if item.get('sha256'):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
                    digest.update(chunk)
            if digest.hexdigest() != item['sha256']:
                errors.append(f'SHA256 mismatch: {item["name"]}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo', choices=list(SUITE_REPOS.values()))
    group.add_argument('--suite', choices=list(SUITE_REPOS))
    parser.add_argument('--revision', help='Default: previously recorded commit, otherwise resolve main once')
    parser.add_argument('--download', action='store_true', help='Download full repo via hf; never all four at once')
    parser.add_argument('--verify-local', action='store_true', help='Hash local files against pinned Hub LFS SHA256')
    args = parser.parse_args()
    repo = args.repo or SUITE_REPOS[args.suite]
    record_path = ROOT / 'manifests' / f'checkpoint-{repo.replace("/", "--")}.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else {}
    revision = args.revision or previous.get('revision', 'main')
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    items = [{'name': f.rfilename, 'size': f.size,
              'sha256': f.lfs.sha256 if f.lfs else None} for f in info.siblings]
    names = [item['name'] for item in items]
    if not any(n.startswith('action_head') for n in names) or 'model.safetensors.index.json' not in names:
        parser.error('Expected OFT action head and sharded model index not found')
    if any(item['size'] is None for item in items):
        parser.error('Unknown file sizes; refusing unbounded download')
    target = ROOT / 'checkpoints' / repo.split('/')[-1]
    report = {'repo_id': repo, 'revision': info.sha, 'known_bytes': sum(i['size'] for i in items),
              'files': names, 'file_metadata': items, 'downloaded': False,
              'sha256_verified': False, 'path': str(target),
              'timestamp': datetime.now().astimezone().isoformat()}
    if args.download:
        # Credit completed files and resumable partial bytes; hf stores them under .cache.
        existing = sum(f.stat().st_size for f in target.rglob('*') if f.is_file()) if target.exists() else 0
        needed = max(0, report['known_bytes'] - existing)
        if shutil.disk_usage(ROOT).free < needed + 5 * 1024**3:
            parser.error(f'Insufficient disk: remaining estimate {needed} bytes + 5 GiB reserve required')
        record_path.write_text(json.dumps(report, indent=2) + '\n')
        subprocess.run([sys.executable, '-c', 'from huggingface_hub.cli.hf import main; main()',
                        'download', repo, '--revision', info.sha,
                        '--local-dir', str(target), '--max-workers', '2'], check=True)
    if args.download or args.verify_local:
        report['verification_errors'] = verify_files(target, items)
        report['downloaded'] = not report['verification_errors']
        report['sha256_verified'] = report['downloaded']
        if report['downloaded']:
            (target / 'download-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    elif (previous.get('downloaded') and previous.get('revision') == info.sha):
        # A metadata-only query must not erase a successful verification record.
        report = previous
    record_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if report.get('verification_errors'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
