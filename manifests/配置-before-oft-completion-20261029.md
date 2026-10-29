# OpenVLA-OFT / LIBERO 仅评测环境配置记录

- 初次配置：2026-10-28（Asia/Shanghai）；最近修正：2026-10-29
- 项目目录：`/home/lyx/workspace/openvla-oft-libero-eval`
- 目标：使用作者已经训练好的 **OpenVLA-OFT** checkpoint 做 LIBERO evaluation；不训练、不微调。
- **当前状态：完成了本地依赖复用、LIBERO 安装及仿真验证；尚未完成可以加载 OFT checkpoint 的端到端评测环境。**

> 最新结论（2026-10-29）：CUDA 和 NVIDIA GPU EGL 渲染均已验证可用；已修复旧 `smoke_all.sh` 强制 Mesa 软件渲染导致的 EGL 初始化失败，四个 suite 的 task 0 均通过。OFT 源码和 checkpoint 仍未准备，不能据此宣称“论文结果已复现”。网络失败属于 2026-10-28 的历史检测，本次只修复渲染，没有重新验证网络。

## 一、为什么这样配置

1. **只准备评测链路**：LIBERO 的任务定义、初始状态、仿真资产，以及后续的 OFT 模型推理。没有下载训练演示数据，没有运行训练。
2. **保留原有项目和 Conda 环境**：没有升级、卸载或改动原有 `openvla` / `lerobot` 环境中的软件。
3. **使用轻量 venv 覆盖层**：复用本机已有的大体积 PyTorch、TensorFlow 等依赖，避免无网络时重复安装和大量占用磁盘。它是项目专属安装位置，**但不是完全独立环境**。
4. **原版 OpenVLA 不等于 OFT**：原有 `openvla-7b-finetuned-libero-spatial` 等 checkpoint 没有被当作 OFT checkpoint 使用。项目启动器会检查 OFT 源码位置和 action head，缺少时拒绝启动。
5. **不把 4-bit 测试当作论文评测**：本项目没有为了在小显存显卡上跑通而静默改成量化推理，也没有修改官方模型或评测器。

### 环境复用关系（重要）

```text
项目 .venv（本次新建）
  ├── 本地安装：项目 vendor/LIBERO
  └── --system-site-packages 继承：
      /home/lyx/miniforge3/envs/openvla
          ├── PyTorch / Transformers / MuJoCo 等
          └── 旧版 OpenVLA 的 editable 安装（不是 OFT）
```

- 基础解释器：`/home/lyx/miniforge3/envs/openvla/bin/python`。
- 项目解释器：`.venv/bin/python`，Python **3.10.22**。
- **不要删除或随意升级被复用的 `openvla` 环境。** 它发生变化会影响本项目。
- 以后若迁移到其他机器，应按已核实的 OFT 源码依赖新建独立环境；不能只复制这个 venv。
- `env.sh` 为将来的 OFT 源码配置了优先搜索路径。但在 OFT 尚未下载时，手工 `import prismatic` 仍可能找到旧 OpenVLA。**不要绕过 `scripts/run_eval.py` 的来源检查。**

## 二、机器与访问情况：实际检测结果（GPU 于 2026-10-29 更新）

| 项目 | 结果 |
|---|---|
| 操作系统 | Ubuntu 22.04 系列，Linux 6.8.0-138-generic，x86_64 |
| 物理内存 | 约 31 GiB |
| GPU 型号 | `/proc/driver/nvidia/gpus/.../information` 显示 NVIDIA GeForce RTX 4060 Laptop GPU |
| 当前 CUDA 可用性 | 2026-10-29：`torch.cuda.is_available()` 为 **True** |
| `nvidia-smi` | 2026-10-29：成功，Driver 595.99.02，显示显存 8188 MiB |
| GPU 设备文件 | 2026-10-29：可见 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 等 |
| NVIDIA 内核模块 | 能从 `/proc` / `lsmod` 读到已加载模块；没有改动驱动 |
| CUDA runtime | 已有 PyTorch wheel 的 CUDA runtime 为 **12.1** |
| 完整 CUDA Toolkit | 未发现 `nvcc`；本次没有安装 |
| 网络 | 2026-10-28 的检测为代理不可连接、直连解析失败；2026-10-29 本次未重新验证 |
| 可用磁盘 | 开始时 `/home` 约 40 GiB；项目新增约 0.8 GiB，未下载模型权重 |

**诊断边界：** 2026-10-28 的受限会话无法访问 GPU，不应当成当前状态。2026-10-29 重新检测后，CUDA 和 GPU EGL 均可用，无需为本次错误重装驱动。没有改代理、系统网络或 `.bashrc`。

**显存规划：** 2026-10-29 的 `nvidia-smi` 显示总显存 8188 MiB；PyTorch 报告可见设备内存 8160673792 字节。仅按 7B 参数 × BF16 的 2 字节计算，参数就约 14 GB，尚未计入其他开销。因此不要预期这张卡能直接完成不量化的标准 7B 评测。可在具有更充足显存的 GPU 上完成正式评测；项目启动器保守拒绝低于 16 GiB 的 GPU，**这不是说 16 GiB 一定足够，也不是论文的官方最低配置**。

## 三、本次实际新增了什么

| 内容 | 状态与说明 |
|---|---|
| 项目文件夹 | 已创建，与旧的 `openvla-lab` 分开 |
| `.venv` | 已创建；`venv --system-site-packages`，不是新建 Conda 环境 |
| LIBERO 源码副本 | 已从本机已有 Git 仓库克隆到 `vendor/LIBERO`；不是本次联网下载 |
| LIBERO Python 包 | 已在项目 venv 内执行本地 editable 安装：`pip install --no-index --no-deps --no-build-isolation -e vendor/LIBERO` |
| LIBERO 版本 | 包版本 `0.1.0`；Git commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| 项目专属 LIBERO 配置 | `.libero/config.yaml`，指向本项目的 BDDL、初始状态和 assets；未改 `~/.libero` |
| LIBERO 资源 | 本地副本包含评测使用的 BDDL、初始状态和仿真资产 |
| 环境入口 | `env.sh`，仅影响 source 它的当前 shell |
| 检查与测试脚本 | `doctor.py`、仿真 smoke test、后续源码/权重准备工具、带防误用检查的启动器 |
| 记录 | `manifests/` 中的版本、commit、依赖及检查报告，`logs/` 中的实际输出 |

`vendor/LIBERO` 的 origin 已指向官方仓库地址，但**本次没有联网确认它是否为 OFT 作者推荐的具体 LIBERO revision**。后续必须与 OFT 官方评测文档对齐，不能只凭本机可运行就认定版本完全一致。

## 四、已可用但不是本次重新安装的依赖

以下版本来自实际 `importlib.metadata` 检查，均复用已有 `openvla` 环境：

| 依赖 | 当前版本 | 说明 |
|---|---|---|
| Python | 3.10.22 | 基础解释器 |
| torch | 2.2.0+cu121 | CUDA 12.1 wheel；2026-10-29 已验证 CUDA 可用 |
| torchvision | 0.17.0+cu121 | 已有 |
| transformers | 4.40.1 | 已有；还没有完成 OFT 导入测试 |
| tokenizers | 0.19.1 | 已有 |
| numpy | 1.26.4 | 不盲目升级到 NumPy 2 |
| timm | 0.9.10 | 已有 |
| accelerate | 0.30.1 | 已有 |
| peft | 0.11.1 | 已有；不代表会运行训练 |
| draccus | 0.8.0 | 已有 |
| huggingface-hub | 0.36.2 | 可用 API；联网状态沿用历史记录，2026-10-29 本次未重测 |
| robosuite | 1.4.1 | 已通过本次仿真测试 |
| mujoco | 2.3.7 | 已通过本次仿真测试 |
| bddl | 3.6.0 | 已有，并通过当前任务测试；仍需与作者版本核对 |
| imageio | 2.38.0 | 用于输出验证视频 |
| imageio-ffmpeg | 0.6.0 | 用于输出验证视频 |
| tensorflow | 2.15.0 | 已有；没有重新安装，是否被 OFT 图像预处理路径使用需看实际代码 |
| tensorflow-datasets | 4.9.3 | 已有；没有因此下载训练数据 |
| bitsandbytes | 0.43.3 | 已有，但本项目没有启用量化 |

- `pip check` 实测为 **No broken requirements found**。
- 这只说明**当前已安装包声明的依赖**没有冲突，**不表示尚未安装的 OFT 已经满足运行要求**。
- `manifests/pip-freeze-inherited.txt` 是当前可见包的审计快照，含旧 OpenVLA 的 editable 路径；**不要直接把它当作独立机器的 OFT 安装锁文件**。
- `manifests/pip-local.json` / `pip-freeze-local.txt` 记录项目覆盖层的本地安装。
- `previous-openvla-constraints-reference.txt` 来自旧项目，**只是原版 OpenVLA 的历史参考，不是 OFT 官方 requirements**。

## 五、什么没有安装/下载

| 内容 | 本次状态 | 原因/后续处理 |
|---|---|---|
| **OpenVLA-OFT 源码与 Python 包** | **未下载、未安装** | 无法连接官方仓库；旧 OpenVLA 不可替代 |
| **OFT 预训练/微调 checkpoint** | **未下载** | 无法连接 Hugging Face，且应先核实作者给出的具体仓库和 revision |
| **OFT action head / proprio projector 权重** | **未下载** | 不应只下载 7B 主干，漏掉配套模块 |
| `diffusers` | **未安装** | 本地没有；具体需求和版本待核实所用 OFT revision；不能因为选 L1 就假设模块一定不会被 import |
| `flash-attn` | **未安装** | 无本地可用包，且未发现 `nvcc`；先核实官方加载器是否强制 FlashAttention，再按匹配版本准备 |
| `robomimic` | **未安装** | 当前已验证的仿真路径不需要；后续 OFT 是否需要以实际导入为准 |
| 完整 CUDA Toolkit / `nvcc` | **未安装** | 不为普通 wheel 推理盲目安装；如需要编译扩展再按兼容矩阵处理 |
| 新的 NVIDIA 驱动 | **未安装** | 没有修改系统驱动或内核模块 |
| LIBERO 演示训练集、RLDS 数据 | **未下载** | 目标只是 checkpoint evaluation，不是训练 |
| 原版 OpenVLA 基础 checkpoint 的新副本 | **未下载** | 已有缓存没有冒充 OFT，也没有重复复制 |
| 新的训练工具链/分布式组件 | **未新增** | 不运行训练；旧环境本来已有的相关依赖仍保留 |

**注意 FlashAttention：** 没有验证 OFT 的实际加载器，不能宣称它对本次任务一定可省略。若官方路径写死了 `flash_attention_2`，缺包时就必须先解决，不能静默换实现后仍称严格复现。

## 六、实际验证到了哪一步

### 已通过

1. 项目 Python 解释器启动、依赖版本枚举及 `pip check`。
2. 四个 suite 的资源检查：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，各 10 个任务，共 40 个；全部 BDDL 文件存在，全部任务均可读取 50 个初始状态。
3. **每个 suite 的 task 0**：创建环境、reset、装载初始状态、离屏渲染、执行 5 次零位移动作、保存 PNG 和 MP4。
4. 2026-10-29：修正脚本后使用 NVIDIA GPU EGL；实际 OpenGL vendor 为 `NVIDIA Corporation`，renderer 为 `NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2`。2026-10-28 的软件渲染测试仅代表当时受限会话，不代表当前推荐配置。
5. `doctor.py --strict` 正确以退出码 2 报告尚未就绪。
6. OFT 源码缺失时，评测启动器正确拒绝运行，没有误用旧 OpenVLA。

### 未验证 / 未完成

- 不是对全部 40 个任务逐个进行了仿真 rollout；本次仿真测试为每个 suite 的第一个任务。
- 未下载或导入官方 OFT，不知道本机版本组合能否满足所选 revision 的完整导入路径。
- 未加载模型，未测试 action head / proprio projector，也未执行模型预测。
- NVIDIA EGL GPU 渲染现已验证；模型推理的显存占用和推理速度仍未验证。
- 未进行正式成功率评测、种子重复、统计汇总或与论文表格比较。
- `profiles/*.template.json` 是**离线待核对模板**，不是从本次联网核实的论文实验配置。50 次 trial、seed 7、L1、2 路图像、proprio、8 步 action chunk 等字段均须逐项对照作者文档，不能据此声称复现了论文协议。

验证产物：`outputs/simulator_<suite>_0.{png,mp4,json}`；原始输出见 `logs/simulator-<suite>.log`。

## 七、现在可以直接执行的命令

```bash
cd /home/lyx/workspace/openvla-oft-libero-eval
source env.sh

# 查看真实状态；加 --strict 在未就绪时返回非零退出码
python scripts/doctor.py
python scripts/doctor.py --strict

# 已验证可用：四个 suite 各跑一个仅仿真的短测试
bash scripts/smoke_all.sh
```

`smoke_all.sh` 现已清理冲突的软件渲染变量，并设置 `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；让系统 GLVND 枚举渲染设备，不再强制 Mesa。所有变量操作仅影响脚本子进程，不改系统 EGL 配置。它测试的是仿真器，不是 OFT 模型。

## 八、确认网络并准备 OFT 后，如何继续（GPU 已验证可用）

### 1. 先在你自己的普通终端确认访问能力

```bash
nvidia-smi
curl -I --max-time 20 https://github.com
curl -I --max-time 20 https://huggingface.co
```

不要因为 2026-10-28 受限会话的历史失败就直接重装驱动；2026-10-29 已验证 GPU 和 EGL 正常。先确认普通终端是否能看到 GPU，以及执行评测的会话是否能访问 NVIDIA 设备。若本机显存不足，正式评测应转到更大显存的 GPU；这里的 venv 不能直接搬到远程机器使用。

### 2. 获取并固定官方 OFT 源码

待核实的作者仓库入口：

```text
https://github.com/moojink/openvla-oft
https://github.com/Lifelong-Robot-Learning/LIBERO
```

```bash
cd /home/lyx/workspace/openvla-oft-libero-eval
source env.sh
bash scripts/fetch_oft_source.sh
# 已知作者推荐 commit 时：bash scripts/fetch_oft_source.sh <commit>
```

此脚本只获取源码、检查预期入口并记录 commit，**不会安装所有依赖，更不会下载模型**。如果目录已存在，它会拒绝覆盖。网络中断后留下的不完整目录需要先人工检查。

必须阅读下载下来的 README、LIBERO 专用评测说明、依赖声明和模型卡，确认：

- OFT 推荐的 Python / PyTorch / Transformers / timm 版本，以及 LIBERO / robosuite revision；
- L1 与 diffusion checkpoint 的区别；图像数、proprio、FiLM、action chunk、center crop；
- 正式评测 suite、trial 数、种子数和统计方式；
- FlashAttention 是否被实际加载路径强制使用；
- checkpoint 除主干外还需要哪些文件。

在确认版本兼容并补齐必要依赖之后，可将 OFT 安装到覆盖层：

```bash
# 注意：--no-deps 是为了避免顺带安装训练依赖，并不代表依赖已经满足。
python -m pip install --no-deps --no-build-isolation -e vendor/openvla-oft
python -m pip check
python scripts/doctor.py --strict
```

没有为尚未读取的 OFT revision 编造一份“保证可用”的依赖锁文件。需要按实际源代码补齐 `diffusers`、可能必需的 `flash-attn` 等，完成官方评测入口的导入/帮助输出测试后，再保存最终冻结版本。不要直接 `pip install -U` 全部依赖。

### 3. 获取一个 suite 对应的完整 checkpoint

本次没有联网验证具体的 HF repo ID，因此没有在脚本中写死一个未经核实的模型地址。请从**所选 OFT revision 的官方 README/模型卡**复制对应 suite 的准确 repo ID，不要填旧的 `openvla/openvla-7b-finetuned-libero-spatial`。

```bash
# 先替换占位符；下面第一条命令只查询元数据，不下载权重。
python scripts/checkpoint.py --repo '作者给出的准确OFT仓库ID'

# 确认仓库、文件列表、大小、磁盘余量后，再下载。
python scripts/checkpoint.py --repo '作者给出的准确OFT仓库ID' --download
```

工具会将查询到的 Hub commit SHA 用于实际下载，保存下载清单，并保留整个模型仓库的配套文件。它检查 action head 和 safetensors，但**文件名检查不能证明 checkpoint 身份或其对应的 suite，仍需核实模型卡**。只准备当前要评测的 suite，避免一次下载多份 7B 权重耗尽磁盘。

### 4. 核实 profile 后再运行官方评测器

先复制模板为正式配置并对照作者设置修改。现有模板不是已确认的论文参数：

```bash
cp profiles/libero_spatial.l1.template.json profiles/libero_spatial.verified.json
# 编辑新文件，核对 options 中所有字段、checkpoint 和 suite 映射。

python scripts/run_eval.py \
  --profile profiles/libero_spatial.verified.json \
  --checkpoint checkpoints/实际下载目录名 \
  --dry-run

# 只有完成核对后才使用下面的确认参数。
python scripts/run_eval.py \
  --profile profiles/libero_spatial.verified.json \
  --checkpoint checkpoints/实际下载目录名 \
  --confirm-official-settings
```

- 启动器调用的预期入口为 `vendor/openvla-oft/experiments/robot/libero/run_libero_eval.py`。若官方 revision 更改了目录/参数，应先按官方文档调整，不能换成旧 OpenVLA 脚本。
- 启动前检查 profile 字段、OFT 的导入来源、本地 checkpoint、action head、proprio projector（启用时）、CUDA 和显存门槛。
- `--dry-run` 只用于检查/打印命令；不加载模型，也不证明依赖与 checkpoint 已经可运行。
- 启动时记录 OFT commit、工作区是否有改动、配置和命令；官方输出保留在其原有路径。建议自行将终端输出用 `tee` 保存。
- 对其他 suite 重复以上过程，使用各自配套的 checkpoint，不能未经作者说明就用一个 spatial checkpoint 评测所有 suite。

## 九、项目结构与使用边界

```text
openvla-oft-libero-eval/
├── 配置.md                         本文
├── env.sh                          项目环境入口
├── .venv/                          继承已有环境的覆盖层
├── .libero/config.yaml             本项目专属路径配置
├── vendor/LIBERO/                  已有本地源码克隆，已安装
├── vendor/openvla-oft/             当前不存在；待联网获取
├── profiles/*.l1.template.json     未核实的配置模板
├── scripts/
│   ├── doctor.py                   前置条件与资源检查
│   ├── smoke_libero.py             仅仿真 smoke test
│   ├── smoke_all.sh                四个 suite 的 GPU EGL 测试
│   ├── fetch_oft_source.sh         获取并记录 OFT 源码
│   ├── checkpoint.py               查询/下载一个明确指定的模型仓库
│   └── run_eval.py                 带前置检查的官方入口启动器
├── checkpoints/                    当前没有模型权重
├── data/                           当前为空，不含训练演示数据
├── manifests/                      已安装版本、commit、诊断快照
├── logs/                           本次真实安装与测试日志
├── tests/test_smoke_all.py          渲染变量和失败传播的回归测试
└── outputs/                        仅仿真验证图片、视频和 JSON
```

**最终结论：** 这次实际完成的是“保留旧环境的项目化离线准备 + LIBERO 仿真链路验证”，不是已经完成的 OpenVLA-OFT evaluation。剩余关键项是官方源码和协议核实、缺失依赖、完整 OFT checkpoint，以及能运行标准非量化推理的 GPU。没有把这些未完成项写成已安装或已通过。


## 十、2026-10-29：EGL 报错原因和修复记录

### 根因

旧脚本同时设置了：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
EGL_PLATFORM=surfaceless
LIBGL_ALWAYS_SOFTWARE=1
```

这些设置适用于前一次受限会话中验证过的路径，但并不适合当前有可见 GPU 的会话。检查本地 robosuite 1.4.1 源码可见，它会通过 `eglQueryDevicesEXT` 枚举设备，并调用 `eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, ...)`。当前环境中，强制 Mesa 软件渲染与其显式设备选择相冲突，日志先出现：

```text
Not allowed to force software rendering when API explicitly selects a hardware device.
```

随后 EGL display 初始化失败。`Cannot initialize a EGL device display` 后半句只是库的通用提示，不代表已经证实 NVIDIA 不支持相应扩展。后面的 `_context`、`con` 属性错误是对象未完成初始化时析构导致的次生错误。

### 修改内容

- `scripts/smoke_all.sh`：移除强制 Mesa、surfaceless、软件渲染的设置；清理继承的冲突变量，显式使用 EGL；保留用户的设备选择变量。仍保留 `set -euo pipefail`，不会把失败伪装成通过。
- `scripts/smoke_libero.py`：在每次结果 JSON 中记录时间和实际 OpenGL vendor / renderer，方便区分 GPU 渲染与软件渲染。
- `tests/test_smoke_all.py`：验证冲突变量已被清理、设备选择保留、四个 suite 被执行，以及 Python 失败时管道返回非零并停止。
- 未安装/升级/卸载任何 Python 包、系统库或驱动，没有修改第三方依赖源码。

### 验证记录

- 用旧变量在本机重现失败，退出码 1，日志：`logs/egl-software-failure-20261029.log`。
- 去掉冲突变量后，相同任务通过，日志：`logs/egl-gpu-probe-20261029.log`。
- 修复后的 `bash scripts/smoke_all.sh`：四个 suite 各 task 0、5 步全部通过，实际 renderer 均为 NVIDIA RTX 4060 Laptop。
- 两项无 GPU 依赖的 shell 回归测试通过：`python -m unittest discover -s tests -v`。

`No private macro file found` 和 Gym 的维护状态提示仍可能打印。它们不是此次 EGL 失败的原因；不要为消除提示而盲目升级旧版仿真依赖。

`doctor.py --strict` 仍应返回非零，因为 OFT 源码、checkpoint 等尚未就绪；仿真测试通过不会自动消除这些独立阻塞项。
