# bash: source /home/lyx/workspace/openvla-oft-libero-eval/env.sh
export OFT_EVAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$OFT_EVAL_ROOT/.venv/bin/activate"
export LIBERO_CONFIG_PATH="$OFT_EVAL_ROOT/.libero"
# OFT must precede the inherited original OpenVLA editable installation.
export PYTHONPATH="$OFT_EVAL_ROOT/vendor/openvla-oft:$OFT_EVAL_ROOT/vendor/LIBERO"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2
export TF_FORCE_GPU_ALLOW_GROWTH=true
export HF_HOME="$OFT_EVAL_ROOT/.cache/huggingface"
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
