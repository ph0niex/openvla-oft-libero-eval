"""NON-PAPER 4-bit diagnostic: one real observation -> 8 OFT actions -> simulator steps.

Not a full episode, not a success rate, not BF16 reproduction. Uses official helpers unmodified.
"""
from __future__ import annotations
import argparse
import importlib.util
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from checkpoint_utils import check_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--acknowledge-nonpaper-4bit', action='store_true', required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    integrity = check_checkpoint(checkpoint)
    if not integrity['complete']:
        parser.error('; '.join(integrity['errors']))
    if integrity['repo_id'] != 'moojink/openvla-7b-oft-finetuned-libero-spatial':
        parser.error('This bounded smoke uses Spatial task 0 and requires the Spatial checkpoint.')
    for key in ('LIBGL_ALWAYS_SOFTWARE', 'LIBGL_ALWAYS_INDIRECT', 'EGL_PLATFORM',
                '__EGL_VENDOR_LIBRARY_FILENAMES', '__EGL_VENDOR_LIBRARY_DIRS',
                'MESA_LOADER_DRIVER_OVERRIDE', 'GALLIUM_DRIVER'):
        os.environ.pop(key, None)
    os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', HF_HUB_OFFLINE='1', WANDB_MODE='disabled')
    for module, subdir in (('prismatic', 'openvla-oft'), ('transformers', 'transformers-openvla-oft')):
        spec = importlib.util.find_spec(module)
        if not spec or not spec.origin or not Path(spec.origin).resolve().is_relative_to(ROOT / 'vendor' / subdir):
            parser.error(f'Incorrect {module} origin; source env.sh first.')
    # Official check_model_logic_mismatch() walks ./prismatic/; preserve that contract.
    os.chdir(ROOT / 'vendor/openvla-oft')
    import numpy as np
    import torch
    from PIL import Image
    from libero.libero import benchmark
    from experiments.robot.libero.run_libero_eval import (
        GenerateConfig, initialize_model, prepare_observation, process_action, validate_config)
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
    from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere

    report = {'timestamp': datetime.now().astimezone().isoformat(),
              'kind': 'NON_PAPER_4BIT_ONE_ACTION_CHUNK_NOT_A_SUCCESS_RATE',
              'checkpoint': integrity, 'status': 'RUNNING',
              'oft_commit': subprocess.check_output(['git', '-C', str(ROOT/'vendor/openvla-oft'), 'rev-parse', 'HEAD'], text=True).strip()}
    output = ROOT / 'outputs' / 'oft_4bit_smoke.json'
    output.parent.mkdir(exist_ok=True)
    env = None
    start = time.monotonic()
    try:
        cfg = GenerateConfig(pretrained_checkpoint=str(checkpoint), task_suite_name='libero_spatial',
                             load_in_4bit=True, load_in_8bit=False, num_trials_per_task=1, use_wandb=False)
        validate_config(cfg)
        set_seed_everywhere(cfg.seed)
        report['settings'] = vars(cfg).copy()
        report['gpu'] = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
        report['settings'] = vars(cfg).copy()
        report['working_directory'] = str(Path.cwd())
        report['model_logic_sha256'] = {}
        for name in ('modeling_prismatic.py', 'configuration_prismatic.py'):
            source = ROOT / 'vendor/openvla-oft/prismatic/extern/hf' / name
            local = checkpoint / name
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            local_hash = hashlib.sha256(local.read_bytes()).hexdigest()
            report['model_logic_sha256'][name] = {'source': source_hash, 'checkpoint': local_hash}
            if source_hash != local_hash:
                raise RuntimeError(f'Official model logic sync failed: {name}')
        report['model_loaded'] = True
        report['model_tensor_bytes'] = model.get_memory_footprint()
        report['hf_device_map'] = getattr(model, 'hf_device_map', None)
        report['llm_attention_backend'] = model.language_model.config._attn_implementation
        report['is_loaded_in_4bit'] = bool(getattr(model, 'is_loaded_in_4bit', False))
        if not report['is_loaded_in_4bit']:
            raise RuntimeError('Expected actual bitsandbytes 4-bit load')
        suite = benchmark.get_benchmark_dict()['libero_spatial']()
        env, description = get_libero_env(suite.get_task(0), cfg.model_family, resolution=cfg.env_img_res)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(0)[0])
        for _ in range(cfg.num_steps_wait):
            obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, image = prepare_observation(obs, get_image_resize_size(cfg))
        Image.fromarray(image).save(ROOT / 'outputs/oft_4bit_before.png')
        with torch.inference_mode():
            actions = np.asarray(get_action(cfg, model, observation, description,
                processor=processor, action_head=action_head, proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector, use_film=cfg.use_film))
        torch.cuda.synchronize()
        report['actions_shape'] = list(actions.shape)
        report['actions_finite'] = bool(np.isfinite(actions).all())
        if actions.shape != (8, 7) or not report['actions_finite']:
            raise RuntimeError('Expected finite (8,7) action chunk')
        report['actions'] = actions.tolist()
        executed = 0
        for action in actions:
            obs, _, done, _ = env.step(process_action(action.copy(), cfg.model_family).tolist())
            executed += 1
            if done:
                break
        _, image = prepare_observation(obs, get_image_resize_size(cfg))
        Image.fromarray(image).save(ROOT / 'outputs/oft_4bit_after.png')
        report['executed_steps'] = executed
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
        traceback.print_exc()
    finally:
        report['seconds'] = time.monotonic() - start
        report['peak_torch_allocated_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        if env is not None:
            env.close()
        output.write_text(json.dumps(report, indent=2, default=str) + '\n')
        print(json.dumps(report, indent=2, default=str), flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
