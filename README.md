# 原 LNO + 可选多阶段训练

本项目基于原 LNO 开源仓库 [torch-local-neural-operators](https://github.com/PPhub-hy/torch-local-neural-operators) 中的 `Train_Validation` 代码开发，保留基线网络、数据集处理、单阶段训练、测试和 Legendre 滤波器，并增加可选的多阶段残差学习功能。

This project builds on the `Train_Validation` code from the original [LNO repository](https://github.com/PPhub-hy/torch-local-neural-operators), retaining the baseline network, data processing, single-stage training, evaluation and Legendre filters while adding optional multi-stage residual learning.

在此基础上主要增加或修正了：

1. `multistage.py`：多阶段残差训练逻辑。
2. `main.py --stages 1/2/3`：选择是否启用多阶段。
3. GPU 优先的设备选择，以及供新运行使用的随机性控制和版本化安全检查点（非已发布模型的历史训练设置）。
4. 数据与测试辅助函数的多帧通道排列、旧 PyTorch 检查点及评估指标兼容处理。
5. 缓存复用前检查数据划分，避免先评估后训练时误用空训练缓存。

原目录没有被修改。

## 目录

```text
MR-LNO/
├─ main.py                  # 训练与评估入口，支持 --stages
├─ multistage.py            # 多阶段插件
├─ pretrained.py            # 已提供模型的安全推理与运行测试入口
├─ requirements.txt         # 环境依赖
├─ PlotComNS.m
├─ models/
│  ├─ Re100Ma2_t3_s3_inference.pt        # 推荐使用的安全推理包
│  ├─ Re100Ma2_t3_s3_stage1.pp           # 原始第一阶段模型
│  ├─ Re100Ma2_t3_s3_stage2.pp           # 原始第二阶段残差模型
│  ├─ Re100Ma2_t3_s3_stage3.pp           # 原始第三阶段残差模型
│  └─ Re100Ma2_t3_s3_multistage_meta.pt  # 原始阶段元数据
├─ Data/
│  └─ DatasetNS.py          # 原 LNO 数据集
├─ lib/
│  ├─ networkNS.py          # 原 LNO 网络
│  ├─ train.py              # 原单阶段训练
│  ├─ test.py               # 原测试
│  ├─ utils.py
│  └─ legendres/*.mat       # 原 LNO 必需滤波器
└─ tests/
   ├─ test_multistage.py    # 训练、检查点与设备兼容测试
   ├─ test_cache_splits.py  # 数据划分与缓存复用测试
   ├─ test_pretrained.py    # 预训练推理与文件输入输出测试
   └─ test_training_config.py # 与作者训练脚本的参数一致性测试
```

`models/` 已提供三个阶段的原始模型、元数据和安全推理包。不包含原始训练/测试数据、`.idea`、缓存、日志、实验输出和频响分析脚本。

## 环境安装

建议使用 Python 3.12，并在独立环境中安装锁定版本：

```powershell
python -m pip install -r requirements.txt
```

当前验证环境为 Python 3.12.13、PyTorch 2.12.0、NumPy 2.5.1、SciPy 1.18.0；GPU 环境使用 CUDA 12.6。

## 使用方法

先进入下载或克隆后的仓库目录（以下假设目录名为 `MR-LNO`）：

```powershell
cd MR-LNO
```

以下命令中的 `<data-root>` 是占位符，请将其替换为自己电脑上的数据根目录（即包含 `ComNS128Re100Ma2` 文件夹的目录），不要原样输入尖括号。路径可以是绝对路径，也可以是相对于当前工作目录的路径；本项目不要求特定盘符。

Replace `<data-root>` with your own data root containing the `ComNS128Re100Ma2` folder. It may be an absolute path or a path relative to your current working directory. Do not type the angle brackets literally. No particular drive letter is required.

原始单阶段 LNO：

```powershell
python main.py -n baseline --stages 1 --data-dir "<data-root>"
```

两阶段：

```powershell
python main.py -n stage2 --stages 2 --data-dir "<data-root>"
```

三阶段：

```powershell
python main.py -n stage3 --stages 3 --data-dir "<data-root>"
```

`--stages` 默认是 `1`，因此不写这个参数时就是原始 LNO：

```powershell
python main.py -n baseline --data-dir "<data-root>"
```

## 接口行为

| 参数 | 训练代码 | 模型文件 |
|---|---|---|
| `--stages 1` | 原 `lib/train.py` | `models/<name>_model.pp` |
| `--stages 2` | `multistage.py` | `models/<name>_multistage.pt` |
| `--stages 3` | `multistage.py` | `models/<name>_multistage.pt` |

无论选择哪种模式，训练结束后都会继续调用原来的 `lib/test.py`，生成 `outputs` 和 `MSE_t` 结果。

新生成的模型使用版本 2 检查点：只保存 `state_dict`、网络参数、完整实验配置和运行环境版本，可通过 `weights_only=True` 安全加载。版本 1 多阶段模型仍可直接读取。

早期单阶段 `torch.save(model)` 文件属于 Python pickle。只有确认文件可信时才允许加载：

```powershell
python main.py -n old_baseline --eval-only --allow-legacy-pickle --data-dir "<data-root>"
```

加载可信旧模型时，程序还会补齐旧版 PyTorch `GELU` 缺失的 `approximate="none"` 属性，使原 LNO 整模型检查点可以在新版 PyTorch 中继续前向运行。

`--eval-only` 会跳过训练和保存，直接加载已有检查点，因此不会覆盖旧模型。版本 1 多阶段模型也通过同一方式评估：

```powershell
python main.py -n old_stage3 --stages 3 --eval-only --data-dir "<data-root>"
```

## 多阶段原理

```text
Stage 1: f1(x) ≈ y
Stage 2: f2(x) ≈ y - f1(x)
Stage 3: f3(x) ≈ y - f1(x) - f2(x)
最终输出: F(x) = f1(x) + f2(x) + f3(x)
```

训练 Stage 2/3 时，已经训练完成的阶段会被冻结。自回归滚动到下一时刻时，使用当前所有阶段的总输出，而不是只使用残差网络输出。

默认训练参数已按作者提供的 `main_NS_multistage.py` 及其调用的 `lib/train_multistage.py` 对齐：

```text
每个 Stage
└─ 10 rounds
   └─ 每轮 10 epochs（每阶段共 100 epochs）
      └─ 每个 epoch 500 iterations
```

每个阶段使用独立 Adam；每轮结束执行原来的 StepLR（`gamma=0.7`）。

## 已发布模型的训练配置

作者确认已发布模型由 `main_NS_multistage.py` 训练。下面列出该脚本的参数，当前仓库 `main.py` 的对应默认值已同步；但 `main.py` 是整理后的实现，并非历史训练脚本的逐字副本。

```python
learning_rate = 0.001
weight_decay = 1e-4
batch_size = 8
print_frequency = 25
rounds = 10
epochs = 10
recurrent = 10

Re = 100
Ma = 2
t_interval = 3

N = 12
K = 2
M = 6
num_blocks = 4
```

训练集编号为 **41–210（170 个样本）**，测试集编号为 **1–40（40 个样本）**。输入历史长度为 1，训练 rollout 长度为 10；每个 epoch 500 次迭代。每阶段使用独立 Adam，梯度裁剪阈值为 5.0，每轮结束执行 StepLR（`step_size=1, gamma=0.7`）。网络使用 `norm_factors=[0.5,0.5,5,10]` 和 `if_ln=True`。

原脚本 `--train_stages` 默认值为 3；本整理版保留 `--stages` 默认值 1 以支持单阶段基线。运行对应的三阶段配置必须显式传入 `--stages 3`：

```bash
python main.py -n Re100Ma2_t3_s3_newrun --stages 3 --data-dir "<data-root>"
```

The author identifies `main_NS_multistage.py` as the training source for the published models. Matching defaults are batch size 8, 10 rounds × 10 epochs per stage, 500 iterations per epoch, rollout length 10 and raw-frame stride 3. Training samples are 41–210; test samples are 1–40. Use `--stages 3` explicitly in this refactored entry point. Configuration alignment does not imply bitwise reproduction of the historical weights.

## 数据目录

```text
<data-root>/
└─ ComNS128Re100Ma2/
   ├─ ComNS128Re100Ma2_1.mat
   ├─ ComNS128Re100Ma2_2.mat
   └─ ...
```

`main.py` 会自动处理 `--data-dir` 末尾有没有 `/`。

## 计算设备

`--device` 默认是 `auto`：有可用 CUDA GPU 时优先使用 GPU，否则自动回退到 CPU。

```powershell
# 自动选择（默认）
python main.py -n stage2 --stages 2 --device auto --data-dir "<data-root>"

# 明确指定设备
python main.py -n stage2 --stages 2 --device cuda:0 --data-dir "<data-root>"
python main.py -n stage2 --stages 2 --device cpu --data-dir "<data-root>"
```

CPU 路径可用于功能验证，但完整 LNO 训练计算量较大，实际训练仍建议使用 GPU。

## 新运行的随机性控制（不代表历史模型的训练设置）

作者提供的原训练脚本没有显式固定 Python、NumPy、PyTorch/CUDA 随机种子，也没有启用确定性算法。不能把下面的默认种子 `0`、确定性开关或当前验证环境追溯为已发布模型的历史训练条件，不能保证重新训练得到逐位相同的权重。

本仓库整理版为**今后的新运行**增加了 `configure_reproducibility`：设置 Python、NumPy、PyTorch 和 CUDA 的种子；启用 `torch.use_deterministic_algorithms(True)`；关闭 cuDNN benchmark、启用 cuDNN deterministic，并配置 cuBLAS 工作区。默认种子为 `0`，也可自行指定：

```powershell
python main.py -n stage2 --stages 2 --seed 2026 --device auto --data-dir "<data-root>"
```

这些控制旨在减少同一代码、数据、缓存状态和软硬件环境下的运行差异，不保证跨设备、跨版本完全一致，也不是已经完成全程训练复现的证明。代码内设置的 `PYTHONHASHSEED` 不会追溯改变当前解释器启动时的哈希种子；若需控制它，应在启动 Python 前设置环境变量。

可使用下面的开关关闭确定性算法要求；它仍然设置随机种子，因此也不等于原训练脚本未固定种子的行为：

```powershell
python main.py -n stage2 --stages 2 --seed 2026 --no-deterministic --data-dir "<data-root>"
```

整理版新训练生成的版本 2 检查点会记录种子、确定性开关、数据划分、网络配置、优化器与学习率调度器、梯度裁剪、初始化系数、实际运行设备与 GPU 信息、Python/PyTorch/NumPy/SciPy/CUDA 版本，以及核心源代码和当前 Legendre 滤波器资产的 SHA-256 指纹。加载时核对相应配置。**这些新记录机制不适用于历史 `.pp` 文件；推理包转换也不会补造未知的历史种子或环境。**

Randomness controls were added to the refactored code for new runs. The supplied original training script does not explicitly fix random seeds or enable deterministic algorithms. The published weights must not be described as having been trained with seed 0 or the current validation environment. Matching seeds and deterministic settings do not guarantee identical results across software/hardware environments or reproduce unknown historical random states.

命令行入口及 `load_trained_model` 会执行上述完整语义校验；`load_single_stage`、`load_multistage` 是供工具代码使用的低层权重读取函数，其中多阶段读取可通过 `expected_stages` 强制检查阶段数。

## 误差定义

为保持与原 LNO 历史结果可比较，同时明确提供真正的均方误差，评估会生成两份日志：

- `MSE_t/<name>_MSE.log`：原 LNO 历史误差口径，即速度误差模长与 `rho/T` 绝对误差。
- `MSE_t/<name>_true_MSE.log`：真正的均方误差：

```text
UV  = mean((u_pred-u_true)^2 + (v_pred-v_true)^2)
rho = mean((rho_pred-rho_true)^2)
T   = mean((T_pred-T_true)^2)
```

## 测试

插件测试不需要 CUDA 或真实数据：

```powershell
python -m unittest discover -s tests -v
```

测试会覆盖多阶段逻辑、多帧输入通道排列以及 CPU/GPU 设备迁移。

## 当前验证范围与限制

- 自动化测试覆盖缓存复用、训练/加载接口及预训练推理输入转换。此次发布未重新运行完整训练实验。
- 多阶段训练保留作者原训练器的行为：旧阶段通过 `torch.no_grad()` 计算，同时冻结旧参数并切断经过旧阶段的输入梯度；此次配置对齐没有改变这一行为。
- 当前网络仍按四通道、单帧输入使用；辅助函数的多帧通道修复不代表网络已支持 `in_length > 1`。
- 当前 `rho/T` 评估采用对数变量的逆变换；将 `if_ln` 改为 `False` 时，需同步调整评估转换逻辑。
- 仓库包含运行所需的 Legendre 滤波器及下列作者提供的模型文件，不包含训练/测试原始数据或实验输出。数据缓存采用 pickle，仅应加载可信来源的缓存。
- 兼容的旧缓存继续复用；数据划分不兼容时另建带训练集标识的缓存目录，保留旧缓存。首次新建需要原始数据和额外磁盘空间。

## 作者提供的模型文件

`models/` 中提供以下原始文件，未进行格式转换或重新训练：

- `Re100Ma2_t3_s3_stage1.pp`
- `Re100Ma2_t3_s3_stage2.pp`
- `Re100Ma2_t3_s3_stage3.pp`
- `Re100Ma2_t3_s3_multistage_meta.pt`

这四个文件保留原貌。另提供 `models/Re100Ma2_t3_s3_inference.pt`，将同一组权重、网络配置及滤波器导出为可用 `weights_only=True` 加载的推理包。三个阶段的参数逐一一致；在固定非零合成输入上，各阶段和累计输出与原模型在 CPU 上完全一致。这不是重新训练，也不是物理精度复现。

旧 `.pp` 文件使用整模型 pickle，不建议测试者直接加载；使用下方安全推理入口，无需启用 `--allow-legacy-pickle`。推理包与 `main.py` 的训练检查点格式不同，请勿混用入口。

### 下载后直接测试 / Quick start

在克隆的仓库根目录安装依赖后，执行：

```bash
git clone https://github.com/YixingZhu01/MR-LNO.git
cd MR-LNO
python -m pip install -r requirements.txt
python pretrained.py --smoke-test --device cpu
```

This runs the supplied pretrained weights on a synthetic, nonzero input. No dataset or retraining is required. A successful run prints `PASS` and output shape `(1, 1, 4, 128, 128)`. This checks execution, **not scientific accuracy**.

上述命令无需数据集或重新训练，成功后打印 `PASS` 和输出形状。它只验证加载与推理能运行，不验证论文精度。支持 CUDA 的环境也可运行：

```bash
python pretrained.py --smoke-test --device cuda:0 --stages 3 --steps 2
```

### 使用自己的输入 / Custom input

准备 `.npz` 文件，键名为 `input`，数组形状为 `(batch, 4, 128, 128)`，通道依次为 `u, v, rho, T`。必须采用与原数据一致的无量纲量和空间排列；`rho/T` 为正的物理变量，**不要提前取对数或乘网络归一化系数**。本入口自动完成对数变换，模型内部应用归一化系数。

```bash
python pretrained.py --input initial_state.npz --output outputs/prediction.npz --device cpu --stages 3 --steps 10
```

Input: NPZ key `input`, shape `(B,4,128,128)`, channels `u,v,rho,T` in the original nondimensional units. Supply positive physical density and temperature, not their logarithms. Output: NPZ key `prediction`, shape `(steps,B,4,128,128)`, in physical variables (density/temperature exponentiated). The initial frame is excluded. Existing output files are not overwritten.

`--stages 1/2/3` 分别使用第 1 阶段、前 2 阶段之和、全部 3 阶段之和；不是单独使用第 2 或第 3 残差网络。每一步以当前累计预测作为下一步输入。输出不包含初始帧，且不会覆盖已有输出文件。

这组预训练模型用于 Re100Ma2，原始数据帧间隔为 `t_interval=3`；当前 `main.py` 的默认间隔已与作者训练脚本对齐为 3。

These pretrained models use Re100Ma2 with a raw-frame stride of 3, matching the corrected training defaults. Raw datasets and reference trajectories are not included; quantitative validation requires matching reference data and preprocessing. Inference compatibility has been verified; full training reproduction has not been validated.

原始测试数据与参考轨迹仍未提供；验证误差或论文结果需要相应真值数据。推理包内记录了四个原始模型文件的 SHA-256，便于追踪来源。
