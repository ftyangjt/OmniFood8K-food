# Conda 环境配置说明

本文档说明如何为本项目配置 Python/Conda 环境。当前项目推荐使用 Windows + Conda。

## 推荐环境

```text
Python      3.10
PyTorch     2.1.2
TorchVision 0.16.2
CUDA        11.8
环境名       omnifood
```

## 使用 environment.yml

在项目根目录运行：

```powershell
conda env create -f environment.yml
conda activate omnifood
```

如果本机已经存在 `omnifood` 环境，可以更新：

```powershell
conda env update -n omnifood -f environment.yml --prune
```

如果 `environment.yml` 末尾带有其他机器的 `prefix: ...`，在其他电脑上创建环境失败时，可以删除这一行后再执行。

## 手动创建环境

如果完整 Conda 环境创建较慢，可以使用：

```powershell
conda create -n omnifood python=3.10 -y
conda activate omnifood
python -m pip install -r requirements.txt
```

`requirements.txt` 已包含 PyTorch CUDA 11.8 的 wheel 源：

```text
--extra-index-url https://download.pytorch.org/whl/cu118
```

## 验证环境

```powershell
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import torch, torchvision, timm, transformers, cv2, open3d, pytorch_wavelets; print('deps ok')"
```

GPU 环境正常时通常看到：

```text
Python 3.10.x
2.1.2+cu118
True
11.8
deps ok
```

## Depth Anything V2

以下脚本会使用 Depth Anything V2：

```text
scripts/run.py
scripts/generate_8k_depth.py
scripts/infer_nutrition.py
```

源码目录应位于：

```text
external/Depth-Anything-V2/
```

验证导入：

```powershell
python -c "import sys, os; sys.path.insert(0, os.path.abspath('external/Depth-Anything-V2')); from depth_anything_v2.dpt import DepthAnythingV2; print('depth anything ok')"
```

## 权重文件

预训练权重统一放在：

```text
pth/
```

常用文件：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
pth/depth_anything_v2_vitl.pth
```

训练得到的最优权重建议放在：

```text
trained_weights/
```

例如：

```text
trained_weights/omnifood8k/ckpt_best.pth
```

## Windows 常见问题

### DataLoader 启动报错或卡住

Windows 下多进程 DataLoader 容易触发 spawn 递归问题。当前训练和测试脚本默认：

```text
--num_workers 0
```

如果确认流程稳定，可以手动尝试：

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --num_workers 4
```

### CUDA 不可用

先检查 NVIDIA 驱动：

```powershell
nvidia-smi
```

再检查 PyTorch：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### xFormers not available

Depth Anything V2 可能提示：

```text
xFormers not available
```

这是可选加速库缺失提示，通常不影响基本运行。
