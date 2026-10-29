"""Inspect this evaluation workspace; never import the old OpenVLA as OFT."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from checkpoint_utils import check_checkpoint

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'timm',
            'accelerate', 'peft', 'draccus', 'huggingface-hub', 'libero',
            'robosuite', 'mujoco', 'bddl', 'robomimic', 'imageio', 'imageio-ffmpeg',
            'tensorflow', 'tensorflow-datasets', 'diffusers', 'flash-attn', 'bitsandbytes',
            'ninja', 'fastapi', 'uvicorn')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='exit 2 if software/checkpoint checks OR BF16 hardware preflight fail')
    parser.add_argument('--software-only', action='store_true', help='with --strict, ignore hardware capacity (not a BF16 evaluation readiness claim)')
    args = parser.parse_args()
    report = {'timestamp': datetime.now().astimezone().isoformat(), 'python': sys.version, 'executable': sys.executable,
              'base_prefix': sys.base_prefix, 'inherited_packages': True,
              'packages': {}, 'blockers': [], 'hardware_blockers': [], 'warnings': [], 'note': 'Prerequisites only, NOT an evaluation result.'}
    for package in PACKAGES:
        try:
            report['packages'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            report['packages'][package] = None
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False)
    report['nvidia_smi'] = {'returncode': result.returncode, 'output': (result.stdout + result.stderr).strip()}
    import torch
    report['torch_cuda_runtime'] = torch.version.cuda
    report['cuda_available'] = torch.cuda.is_available()
    if report['cuda_available']:
        props = torch.cuda.get_device_properties(0)
        report['gpu'] = {'name': props.name, 'memory_bytes': props.total_memory}
    else:
        report['blockers'].append('CUDA is not accessible to this process.')
        report['hardware_blockers'].append('CUDA device unavailable.')
    repo = ROOT / 'vendor/openvla-oft'
    entry = repo / 'experiments/robot/libero/run_libero_eval.py'
    report['oft_source_present'] = entry.is_file()
    if not entry.is_file():
        report['blockers'].append('Official OpenVLA-OFT source is not downloaded.')
    else:
        report['oft_commit'] = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    spec = importlib.util.find_spec('prismatic')
    report['prismatic_resolves_to'] = spec.origin if spec else None
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(repo.resolve()):
        report['blockers'].append('prismatic resolves to the original OpenVLA, not OFT; evaluation is prohibited.')
    fork = ROOT / 'vendor/transformers-openvla-oft'
    spec = importlib.util.find_spec('transformers')
    report['transformers_resolves_to'] = spec.origin if spec else None
    if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(fork.resolve()):
        report['blockers'].append('Transformers must resolve to the OFT fork; PyPI 4.40.1 is not equivalent.')
    else:
        report['transformers_commit'] = subprocess.check_output(['git', '-C', str(fork), 'rev-parse', 'HEAD'], text=True).strip()
    report['checkpoints'] = [check_checkpoint(p.parent) for p in (ROOT / 'checkpoints').glob('*/config.json')]
    report['complete_oft_checkpoints'] = [p['path'] for p in report['checkpoints'] if p['complete']]
    if not report['complete_oft_checkpoints']:
        report['blockers'].append('No complete pinned and verified OFT checkpoint; see checkpoints[].errors.')
    if report['packages']['diffusers'] is None:
        report['blockers'].append('diffusers is required by the official action_heads import even for L1.')
    for key, file in (('oft_commit', 'oft-commit.txt'), ('transformers_commit', 'transformers-commit.txt')):
        pinned = ROOT / 'manifests' / file
        if pinned.is_file() and report.get(key) != pinned.read_text().strip():
            report['blockers'].append(f'{key} differs from the recorded pinned source revision.')
    report['flash_attention'] = {'installed': report['packages']['flash-attn'] is not None,
                                'required_by_default_eval': False,
                                'note': 'Pinned OFT get_vla leaves attn_implementation commented; CUDA kernel test is in manifests/flash-attn-kernel-test.json.'}
    if report['cuda_available'] and report['gpu']['memory_bytes'] < 16 * 1024**3:
        report['hardware_blockers'].append('GPU has less than the conservative 16 GiB BF16 launch threshold; 8 GiB cannot hold 7B BF16 weights. 16 GiB itself is not a fit guarantee.')
    report['warnings'].append('4-bit smoke is a non-paper diagnostic, not BF16 reproduction. Paper used A100 and averages of three seeds; guide does not name those seeds.')
    check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    report['pip_check'] = {'returncode': check.returncode, 'output': (check.stdout + check.stderr).strip()}
    if check.returncode:
        report['blockers'].append('pip check failed; see pip_check.output')
    report['libero_resources'] = {}
    try:
        from libero.libero import benchmark, get_libero_path
        report['libero_module'] = str(__import__('libero.libero', fromlist=['']).__file__)
        for suite_name in SUITES:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            missing = []
            initial_state_counts = []
            for task_id in range(suite.n_tasks):
                task = suite.get_task(task_id)
                path = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
                if not path.is_file():
                    missing.append(str(path))
                states = suite.get_task_init_states(task_id)
                initial_state_counts.append(len(states))
            if missing or any(count < 1 for count in initial_state_counts):
                report['blockers'].append(f'{suite_name} has missing task definitions or empty initial states.')
            report['libero_resources'][suite_name] = {
                'tasks': suite.n_tasks, 'missing_bddl': missing,
                'initial_states_per_task': initial_state_counts}
        assets = Path(get_libero_path('assets'))
        report['assets_present'] = assets.is_dir()
        if not assets.is_dir():
            report['blockers'].append('LIBERO assets directory is missing.')
    except Exception as exc:
        report['blockers'].append(f'LIBERO resource validation failed: {type(exc).__name__}: {exc}')
    report['software_prerequisites_passed'] = not report['blockers']
    report['bf16_hardware_preflight_passed'] = not report['hardware_blockers']
    report['prerequisites_passed'] = not report['blockers'] and not report['hardware_blockers']
    report['note'] = 'Static prerequisites only; no assertion of sufficient runtime VRAM or paper success rate.'
    (ROOT / 'manifests/doctor.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and (report['blockers'] or (report['hardware_blockers'] and not args.software_only)):
        sys.exit(2)

if __name__ == '__main__':
    main()
