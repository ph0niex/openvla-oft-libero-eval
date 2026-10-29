#!/usr/bin/env bash
# The supported `hf` CLI from the evaluation environment's pinned Hub version.
# Avoid unintentionally using the globally installed, independently upgraded CLI.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"
exec python -c 'from huggingface_hub.cli.hf import main; main()' "$@"
