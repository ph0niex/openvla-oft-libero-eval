#!/usr/bin/env bash
# Download source only. Does not install dependencies or weights.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/vendor/openvla-oft"
if [[ -e "$REPO" ]]; then
  echo "Refusing to overwrite $REPO. Inspect its existing checkout manually." >&2
  exit 2
fi
git clone https://github.com/moojink/openvla-oft.git "$REPO"
if [[ $# -gt 0 ]]; then git -C "$REPO" checkout --detach "$1"; fi
if [[ ! -f "$REPO/experiments/robot/libero/run_libero_eval.py" ]]; then
  echo 'Expected evaluation entrypoint missing; inspect the downloaded revision.' >&2
  exit 2
fi
git -C "$REPO" rev-parse HEAD > "$ROOT/manifests/oft-commit.txt"
git -C "$REPO" remote -v > "$ROOT/manifests/oft-remotes.txt"
printf '\nSource fetched, NOT installed. Read its README and LIBERO instructions first.\n'
printf 'Commit: '; cat "$ROOT/manifests/oft-commit.txt"
printf 'Do not install training data or run training scripts.\n'
