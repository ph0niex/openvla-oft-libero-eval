# OpenVLA-OFT / LIBERO evaluation workspace

完整安装清单、版本来源、未安装项目及实验边界见 **[配置.md](配置.md)**。

**2026-10-29：官方 OFT 与专用 Transformers 分支已安装，依赖已补齐，CUDA / NVIDIA EGL 四套仿真 smoke 已通过。Spatial checkpoint 已完整下载并通过尺寸/SHA256校验，真实4-bit模型已通过8×7动作生成及8步仿真测试。软件doctor通过，最终证据见`manifests/final-verification-20261029.json`。本机8GB显存不支持默认7B BF16论文评测；4-bit smoke不能作为论文结果。**

```bash
cd /home/lyx/workspace/openvla-oft-libero-eval
source env.sh
python -m pip check
python scripts/doctor.py
bash scripts/smoke_all.sh
python -m unittest discover -s tests -v
```

- 项目 venv 继承 `/home/lyx/miniforge3/envs/openvla`；不要删除该基础环境。
- 正式 BF16 profile：`profiles/*.l1.official.json`。不要使用历史离线 template。
- 权重按 suite 下载：`python scripts/checkpoint.py --suite libero_object --download`，需磁盘和可用网络。
- `doctor.py --strict --software-only` 检查软件 + 至少一套已验证 checkpoint；`--strict` 另检查 BF16 显存，因此在8GB机器上预期非零退出。
- 明确非论文的本机量化测试：`python scripts/smoke_oft.py --checkpoint checkpoints/openvla-7b-oft-finetuned-libero-spatial --acknowledge-nonpaper-4bit`。
- 未下载训练数据、未运行训练、未重装驱动。

## Git 仓库与上传范围

此仓库包含本项目的脚本、profiles、配置文档及环境/版本记录。`vendor/` 下的三个上游项目通过 Git submodule 固定到本机使用的 commit；请递归克隆：

```bash
git clone --recurse-submodules https://github.com/ph0niex/openvla-oft-libero-eval.git
cd openvla-oft-libero-eval
# 已经普通克隆时执行：
git submodule update --init --recursive
```

未上传 `.venv/`、checkpoint 权重、下载缓存、`wheelhouse/`、训练数据、运行日志、输出图像及 rollout 视频。这些内容仍保留在原机器上。此仓库不是可直接复制使用的虚拟环境；安装方式和基础环境要求见 `配置.md`。

`.libero/config.yaml` 含本机绝对路径，不纳入版本控制。在新目录中可以用下面的命令重新生成（随后再 `source env.sh`）：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path.cwd().resolve()
libero = root / "vendor/LIBERO/libero/libero"
paths = {
    "benchmark_root": libero,
    "bddl_files": libero / "bddl_files",
    "init_states": libero / "init_files",
    "assets": libero / "assets",
    "datasets": root / "data",
}
(root / ".libero").mkdir(exist_ok=True)
(root / ".libero/config.yaml").write_text(
    "".join(f"{key}: {json.dumps(str(value))}\n" for key, value in paths.items())
)
PY
```

**当前工作区快照说明：** `scripts/run_eval.py` 中的人工确认、禁止量化和最小显存检查已被本地注释；`profiles/libero_spatial.l1.LocalRun.json` 使用 4-bit、seed 26，不能与 `*.official.json` 的非量化设置等同。上文及 `配置.md` 中的软件安装验收记录是历史状态，不代表所有后续本地运行都遵守原 BF16 启动限制。
