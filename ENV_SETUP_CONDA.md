# Conda 环境配置与使用说明

本文档说明如何为本项目配置 Python/Conda 环境，并介绍 `environment.yml` 和 `requirements.txt` 的使用方法。

当前项目推荐环境：

- Python 3.10
- PyTorch 2.1.2
- TorchVision 0.16.2
- CUDA 11.8
- Windows + Conda 环境优先

如果只是快速复现当前环境，优先使用 `environment.yml`。如果已经有自己的 Conda 环境，只想补齐 Python 包，可以使用 `requirements.txt`。

## 1. 准备 Conda

如果已经安装 Anaconda 或 Miniconda，可以跳过本节。

推荐安装 Miniconda：

```text
https://docs.conda.io/en/latest/miniconda.html
```

Windows 用户建议使用 Anaconda Prompt 或 PowerShell，并确认 `python` 来自 Conda 环境：

```powershell
where python
python --version
```

如果 `python --version` 打开的是 Microsoft Store，或者命令不可用，请在 Windows 设置中搜索 `App execution aliases`，关闭 `python.exe` 和 `python3.exe` 的别名。

## 2. 方式一：使用 environment.yml 创建完整环境

这是推荐方式。`environment.yml` 记录了当前已经配置好的 Conda 环境，包括 PyTorch、CUDA 运行时和 pip 依赖。

在项目根目录运行：

```bash
conda env create -f environment.yml
```

创建完成后激活环境：

```bash
conda activate omnifood
```

如果本机已经存在 `omnifood` 环境，可以更新环境：

```bash
conda env update -n omnifood -f environment.yml --prune
```

注意：当前 `environment.yml` 是从 Windows Conda 环境导出的，末尾可能包含本机路径形式的 `prefix`。如果在其他电脑上创建环境时遇到路径相关问题，可以删除 `environment.yml` 最后一行的 `prefix: ...` 后再执行创建命令。

## 3. 方式二：使用 requirements.txt 安装 pip 依赖

如果你希望手动创建 Conda 环境，再用 pip 安装依赖，可以使用本方式。

先创建并激活环境：

```bash
conda create -n omnifood python=3.10 -y
conda activate omnifood
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含 PyTorch CUDA 11.8 的额外索引：

```text
--extra-index-url https://download.pytorch.org/whl/cu118
```

因此会安装：

- `torch==2.1.2`
- `torchvision==0.16.2`
- `torchaudio==2.1.2`
- `numpy==1.24.4`
- `scipy==1.10.1`
- `pandas==2.0.3`
- `timm==0.9.12`
- `transformers==4.36.2`
- `opencv-python`
- `open3d==0.18.0`
- `pytorch-wavelets`
- 以及训练、测试和可视化所需的其他工具包

如果没有 NVIDIA GPU，建议不要直接使用当前 `requirements.txt` 安装 CUDA 版 PyTorch，而是先按 PyTorch 官方说明安装 CPU 版 PyTorch，再安装其他依赖。

## 4. 验证环境

激活环境后，在项目根目录运行：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

GPU 环境正常时，通常会看到：

```text
2.1.2
True
11.8
```

继续检查主要依赖：

```bash
python -c "import torch, torchvision, timm, transformers, cv2, open3d, pytorch_wavelets; print('deps ok')"
```

如果该命令输出 `deps ok`，说明训练和推理所需的主要依赖已经可用。

## 5. Depth Anything V2 配置

项目的 `run.py` 会使用 Depth Anything V2 生成深度图：

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

当前项目已经支持从以下目录自动导入 Depth Anything V2：

```text
external/Depth-Anything-V2/
```

因此只要源码位于该目录，运行 `run.py` 时一般不需要再手动配置 `PYTHONPATH`。

如果目录不存在，可以在项目根目录执行：

```bash
mkdir external
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2
pip install -r external/Depth-Anything-V2/requirements.txt
```

验证导入：

```bash
python -c "import sys, os; sys.path.insert(0, os.path.abspath('external/Depth-Anything-V2')); from depth_anything_v2.dpt import DepthAnythingV2; print('depth anything ok')"
```

如果输出 `depth anything ok`，说明 Depth Anything V2 源码可以被 Python 找到。

## 6. 放置权重文件

项目默认从 `pth/` 目录读取预训练权重：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
pth/depth_anything_v2_vitl.pth
```

其中：

- `swin_base_patch4_window12_384_22k.pth` 用于 Swin 分支
- `convnext_small_22k_1k_384.pth` 用于 ConvNeXt 分支
- `depth_anything_v2_vitl.pth` 用于 `run.py` 生成深度图

`depth_anything_v2_vitl.pth` 文件需要在 https://github.com/DepthAnything/Depth-Anything-V2 找到 Pre-trained Models，下载其中的 Depth-Anything-V2-Large。

也可以在运行时手动指定权重路径：

```bash
python train_nutrition.py --swin_ckpt /path/to/swin.pth --convnext_ckpt /path/to/convnext.pth
python run.py --img-path /path/to/images --ckpt /path/to/depth_anything_v2_vitl.pth
```

## 7. 放置数据集

默认数据集目录：

```text
data/nutrition5k_dataset/
```

默认会读取：

```text
data/nutrition5k_dataset/imagery/
data/nutrition5k_dataset/imagery/txt-file/rgbd_train_processed.txt
data/nutrition5k_dataset/imagery/txt-file/rgbd_test_processed1.txt
data/nutrition5k_dataset/imagery/txt-file/rgb_in_overhead_train_processed.txt
data/nutrition5k_dataset/imagery/txt-file/rgb_in_overhead_test_processed1.txt
```

如果数据集放在其他位置，可以运行时指定：

```bash
python train_nutrition.py --data_root /path/to/nutrition5k_dataset
python test.py --data_root /path/to/nutrition5k_dataset --ckpt ./saved/train/ckpt_best.pth
```

### 7.1 OmniFood8K 本地目录

当前工作区使用开源 OmniFood8K 数据集目录：

```text
data/0-OminiFood8k/
  train_new333.txt
  test_new333.txt
  8036/
    <sample_id>/
      camera_4.jpg
      rgb-d.png
```

`train_nutrition.py` 和 `test.py` 默认使用 `--data_root_8k ./data/0-OminiFood8k`。`--dataset nutrition8K` 分支会在该目录下自动识别 `1-data/` 或 `8036/`，当前公开数据集使用的是 `8036/`。

训练或测试 OmniFood8K 前，每个样本目录都需要同时存在：

```text
camera_4.jpg
rgb-d.png
```

`camera_4.jpg` 来自公开数据集。`rgb-d.png` 是用 `run.py` 和 `pth/depth_anything_v2_vitl.pth` 生成的预测深度图，需要放在对应样本目录中。

一键批量生成缺失的 `rgb-d.png`：

```powershell
.\generate_8k_depth.bat
```

或者直接运行 Python 脚本：

```bash
python generate_8k_depth.py --data-root ./data/0-OminiFood8k --ckpt ./pth/depth_anything_v2_vitl.pth
```

脚本会自动识别 `8036/` 或 `1-data/`，跳过已经存在的 `rgb-d.png`。如果需要重算，添加 `--overwrite`。

本地 OmniFood8K 可以这样运行：

```bash
python train_nutrition.py --dataset nutrition8K
python test.py --dataset nutrition8K --ckpt ./saved/train/ckpt_best.pth
```

## 8. 快速检查代码

在项目根目录运行：

```bash
python -m py_compile train_nutrition.py test.py run.py utils/utils_data222.py
```

如果没有输出错误，说明这些入口脚本至少可以通过 Python 语法检查。

Windows 上如果训练时 DataLoader 卡住或启动很慢，可以先把数据加载的 `num_workers` 临时调小，例如改成 `0`、`4` 或 `8`，确认流程正常后再逐步调高。

## 9. 运行示例

生成深度图：

```bash
python run.py --img-path /path/to/rgb_images --encoder vitl --outdir ./vis_depth
```

指定 Depth Anything V2 权重：

```bash
python run.py --img-path /path/to/rgb_images --encoder vitl --ckpt ./pth/depth_anything_v2_vitl.pth --outdir ./vis_depth
```

训练：

```bash
python train_nutrition.py --dataset nutrition_rgb_pre_d --b 8 --epoch 150 --log ./logs/train
```

使用当前 OmniFood8K 数据集训练：

```bash
python train_nutrition.py --dataset nutrition8K --b 8 --epoch 150 --log ./logs/train_8k
```

测试：

```bash
python test.py --dataset nutrition_rgb_pre_d --b 8 --ckpt ./saved/train/ckpt_best.pth
```

## 10. 常见问题

### ModuleNotFoundError: depth_anything_v2

确认 Depth Anything V2 源码目录存在：

```text
external/Depth-Anything-V2/depth_anything_v2/dpt.py
```

当前 `run.py` 已经会自动把 `external/Depth-Anything-V2` 加入导入路径。如果仍然报错，检查是否在项目根目录运行，或者确认源码目录名称是否一致。

### ModuleNotFoundError: pytorch_wavelets

执行：

```bash
pip install pytorch-wavelets
```

注意 pip 包名是 `pytorch-wavelets`，导入名是 `pytorch_wavelets`。

### torch.cuda.is_available() 是 False

先检查 NVIDIA 驱动：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不可用，需要先安装或更新 NVIDIA 驱动。Conda 或 pip 安装的 CUDA 版 PyTorch 通常不要求单独安装完整 CUDA Toolkit，但系统驱动必须可用。

### WinError 1114 或 c10.dll 加载失败

如果导入 `torch` 时出现类似错误：

```text
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败
Error loading ... torch\lib\c10.dll
```

通常是 PyTorch、CUDA 运行时、显卡驱动或 Conda 环境冲突导致。建议优先使用 `environment.yml` 重新创建干净环境，并确认激活环境后再运行脚本。
