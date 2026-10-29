"""Simulator-only smoke test: no policy, no checkpoint, no success-rate claim."""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_spatial', choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'])
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        parser.error('task-id out of range')
    task = suite.get_task(args.task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
        camera_heights=128, camera_widths=128)
    frames = []
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(args.task_id)[0])
        for _ in range(args.steps):
            frames.append(np.asarray(obs['agentview_image'])[::-1, ::-1].copy())
            obs, _, done, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
            if done:
                break
        prefix = ROOT / 'outputs' / f'simulator_{args.suite}_{args.task_id}'
        Image.fromarray(frames[-1]).save(prefix.with_suffix('.png'))
        imageio.mimsave(prefix.with_suffix('.mp4'), frames, fps=10)
        from OpenGL import GL  # Import after robosuite has initialized its EGL backend.

        # The active context reports the actual renderer, not just requested env vars.
        def gl_string(key):
            value = GL.glGetString(key)
            return value.decode('utf-8', errors='replace') if value else None

        report = {'timestamp': datetime.now().astimezone().isoformat(),
                  'rendering': {'mujoco_gl': os.environ.get('MUJOCO_GL'),
                                'vendor': gl_string(GL.GL_VENDOR),
                                'renderer': gl_string(GL.GL_RENDERER)},
                  'kind': 'simulator_only_NOT_policy_evaluation', 'suite': args.suite,
                  'task_id': args.task_id, 'task': task.language, 'frames': len(frames),
                  'reset_render_step': 'PASS'}
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()

if __name__ == '__main__':
    main()
