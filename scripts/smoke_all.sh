#!/usr/bin/env bash
# GPU EGL smoke tests. Renderer overrides apply only inside this subprocess.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"

# robosuite uses EGL_PLATFORM_DEVICE_EXT, not a surfaceless Mesa display.
# The old forced-software settings conflict with this path on this GPU host.
unset LIBGL_ALWAYS_SOFTWARE LIBGL_ALWAYS_INDIRECT EGL_PLATFORM
unset __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS
unset MESA_LOADER_DRIVER_OVERRIDE GALLIUM_DRIVER
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# Let GLVND enumerate installed vendors/devices instead of forcing Mesa.
# Preserve CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID device selection.
cd "$ROOT"
mkdir -p logs outputs
printf '[smoke] Using GPU EGL; forced Mesa/software overrides cleared for this subprocess.\n'
for suite in libero_spatial libero_object libero_goal libero_10; do
  python scripts/smoke_libero.py --suite "$suite" --task-id 0 --steps 5 2>&1 | tee "logs/simulator-$suite.log"
done
