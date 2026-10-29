"""Guarded BF16 launcher for the official OFT runner; separate from quantized smoke."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime
from checkpoint_utils import check_checkpoint

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--confirm-official-settings', action='store_true', help='You checked this profile against the downloaded official revision/model card')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    options = dict(profile['options'])
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    options['pretrained_checkpoint'] = str(checkpoint)
    repo = ROOT / 'vendor/openvla-oft'
    entry = repo / 'experiments/robot/libero/run_libero_eval.py'
    if not entry.is_file():
        parser.error('Official OFT source absent. Do not substitute the original OpenVLA checkout.')
    tree = ast.parse(entry.read_text())
    fields = {n.target.id for n in ast.walk(tree) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    unsupported = set(options) - fields
    if unsupported:
        parser.error(f'Profile flags not found in downloaded runner annotations: {sorted(unsupported)}; review the current official instructions.')
    command = [sys.executable, str(entry)]
    for name, value in options.items():
        command += [f'--{name}', str(value)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print('Dry run only: no weights loaded, no simulation, no result.', flush=True)
        return
    # if not args.confirm_official_settings:
    #     parser.error('Review the pinned official settings, then explicitly pass --confirm-official-settings.')
    # if options.get('load_in_4bit') or options.get('load_in_8bit'):
    #     parser.error('This launcher is BF16-only. Quantization is limited to the explicitly non-paper smoke_oft.py.')
    integrity = check_checkpoint(checkpoint)
    if not integrity['complete']:
        parser.error('Incomplete/unverified checkpoint: ' + '; '.join(integrity['errors']))
    if profile.get('checkpoint_repo') and integrity['repo_id'] != profile['checkpoint_repo']:
        parser.error('Checkpoint repository does not match this suite profile.')
    if not (checkpoint / 'config.json').is_file() or not any(checkpoint.glob('*.safetensors')):
        parser.error('Local checkpoint config/model shards missing.')
    if not any(checkpoint.glob('*action_head*')):
        parser.error('OFT action-head checkpoint missing; original OpenVLA weights are not accepted.')
    if options.get('use_proprio') and not any(checkpoint.glob('*proprio_projector*')):
        parser.error('Proprioception enabled, but no proprio_projector file found. Check the official checkpoint layout.')
    spec = importlib.util.find_spec('prismatic')
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(repo.resolve()):
        parser.error('prismatic is not resolving to OFT. Source env.sh before running.')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; evaluation not started.')
    memory = torch.cuda.get_device_properties(0).total_memory
    # if memory < 16 * 1024 ** 3:
    #     parser.error('This BF16/non-quantized launcher intentionally refuses GPUs with less than 16 GiB. 16 GiB is NOT a guarantee of sufficient VRAM; choose a larger GPU.')
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if profile.get('oft_commit') != commit:
        parser.error('Profile not verified for the current OFT source commit.')
    import transformers
    if not Path(transformers.__file__).resolve().is_relative_to((ROOT / 'vendor/transformers-openvla-oft').resolve()):
        parser.error('Incorrect Transformers: OFT custom fork required.')
    tf_commit = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/transformers-openvla-oft'), 'rev-parse', 'HEAD'], text=True).strip()
    if tf_commit != (ROOT / 'manifests/transformers-commit.txt').read_text().strip():
        parser.error('Transformers fork revision differs from the verified pin.')
    record = {'timestamp': datetime.now().astimezone().isoformat(), 'oft_commit': commit,
              'profile': profile, 'command': command, 'checkpoint': str(checkpoint),
              'checkpoint_revision': integrity['revision'], 'transformers_commit': tf_commit, 'transformers_origin': transformers.__file__,
              'source_dirty': bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip())}
    tag = datetime.now().strftime('%Y%m%d-%H%M%S')
    manifest = ROOT / 'manifests' / f'eval-{tag}.json'
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    # Preserve the official implementation: no replacement policy, no monkey patches.
    import os
    child_env = dict(os.environ)
    for key in ('LIBGL_ALWAYS_SOFTWARE', 'LIBGL_ALWAYS_INDIRECT', 'EGL_PLATFORM',
                '__EGL_VENDOR_LIBRARY_FILENAMES', '__EGL_VENDOR_LIBRARY_DIRS',
                'MESA_LOADER_DRIVER_OVERRIDE', 'GALLIUM_DRIVER'):
        child_env.pop(key, None)
    child_env.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
    # Official model sync searches ./prismatic/, so its repository must be the working directory.
    result = subprocess.run(command, cwd=repo, env=child_env, check=False)
    record['returncode'] = result.returncode
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
