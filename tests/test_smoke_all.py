"""Regression tests for shell renderer selection; no GPU or model required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = {
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'LIBGL_ALWAYS_INDIRECT': '1',
    'EGL_PLATFORM': 'surfaceless',
    '__EGL_VENDOR_LIBRARY_FILENAMES': '/nonexistent/50_mesa.json',
    '__EGL_VENDOR_LIBRARY_DIRS': '/nonexistent',
    'MESA_LOADER_DRIVER_OVERRIDE': 'swrast',
    'GALLIUM_DRIVER': 'llvmpipe',
}


class SmokeAllTests(unittest.TestCase):
    def run_fixture(self, fail_suite=''):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / 'project'
            (project / 'scripts').mkdir(parents=True)
            (project / 'bin').mkdir()
            shutil.copy2(ROOT / 'scripts/smoke_all.sh', project / 'scripts/smoke_all.sh')
            (project / 'env.sh').write_text('# Stub: no activation needed for this shell test.\n')
            fake_python = project / 'bin/python'
            fake_python.write_text('''#!/usr/bin/env bash
set -euo pipefail
for variable in LIBGL_ALWAYS_SOFTWARE LIBGL_ALWAYS_INDIRECT EGL_PLATFORM __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS MESA_LOADER_DRIVER_OVERRIDE GALLIUM_DRIVER; do
  [[ ! -v "$variable" ]] || { echo "Uncleared override: $variable" >&2; exit 20; }
done
[[ "$MUJOCO_GL" == egl && "$PYOPENGL_PLATFORM" == egl ]]
[[ "$CUDA_VISIBLE_DEVICES" == 0 && "$MUJOCO_EGL_DEVICE_ID" == 0 ]]
[[ "$PWD" == "${0%/bin/python}" ]]
echo "called: $*"
if [[ "$*" == *"${FAIL_SUITE:-__never__}"* && -n "${FAIL_SUITE:-}" ]]; then exit 9; fi
''')
            fake_python.chmod(0o755)
            env = {**os.environ, **OVERRIDES,
                   'PATH': str(project / 'bin') + ':' + os.environ['PATH'],
                   'MUJOCO_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'osmesa',
                   'CUDA_VISIBLE_DEVICES': '0', 'MUJOCO_EGL_DEVICE_ID': '0',
                   'FAIL_SUITE': fail_suite}
            result = subprocess.run(['bash', str(project / 'scripts/smoke_all.sh')],
                                    cwd=temp, env=env, capture_output=True, text=True)
            # The supplied parent environment is not modified by the child script.
            self.assertEqual(env['LIBGL_ALWAYS_SOFTWARE'], '1')
            logs = {p.name: p.read_text() for p in (project / 'logs').glob('*.log')}
            return result, logs

    def test_clears_conflicts_preserves_device_selection_and_runs_four_suites(self):
        result, logs = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count('called:'), 4)
        self.assertEqual(len(logs), 4)

    def test_pipeline_propagates_failure_and_stops(self):
        result, logs = self.run_fixture(fail_suite='libero_object')
        self.assertEqual(result.returncode, 9, result.stdout + result.stderr)
        self.assertEqual(len(logs), 2)
        self.assertNotIn('simulator-libero_goal.log', logs)


if __name__ == '__main__':
    unittest.main()
