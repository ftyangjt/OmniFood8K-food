# Conda 环境配置教程

本文档说明如何使用 Conda 为本项目配置可训练、可测试的 Python 环境。推荐优先使用 GPU 环境；CPU 环境可以用于检查代码和小规模推理，但不适合完整训练。

## 1. 安装 Conda

如果已经安装 Anaconda 或 Miniconda，可以跳过本步。

推荐安装 Miniconda：

```text
https://docs.conda.io/en/latest/miniconda.html
```

Windows 用户注意：如果命令行里的 `python --version` 打开的是 Microsoft Store 或直接失败，请在系统设置里关闭 Python 的 App execution aliases，或者始终在 `conda activate omnifood` 后使用 Python。

## 2. 创建项目环境

建议使用 Python 3.10。不要优先选择 Python 3.12，因为 `open3d`、`pytorch_wavelets`、旧版 `timm` 等依赖更容易出现兼容性问题。

```bash
conda create -n omnifood python=3.10 -y
conda activate omnifood
```

确认当前 Python 来自 Conda 环境：

```bash
python --version
where python
```

Linux/macOS 使用：

```bash
which python
```

## 3. 安装 PyTorch

推荐使用 PyTorch 2.1.2、TorchVision 0.16.2、CUDA 11.8：

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

这组版本来自 PyTorch 官方历史版本安装说明：

```text
https://pytorch.org/get-started/previous-versions/
```

如果你的机器没有 NVIDIA GPU，可以安装 CPU 版本：

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch -y
```

验证 PyTorch：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

GPU 环境下，`torch.cuda.is_available()` 应该输出 `True`。

## 4. 安装项目依赖

先安装常规科学计算和图像处理依赖：

```bash
conda install numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 pillow matplotlib scikit-learn tqdm -y
```

再用 pip 安装深度学习相关依赖：

```bash
pip install timm==0.9.12 transformers==4.36.2
pip install opencv-python imageio open3d==0.18.0
pip install tensorboardX ptflops thop pytorch-wavelets seaborn
```

验证主要依赖：

```bash
python -c "import torch, torchvision, timm, transformers, cv2, open3d, pytorch_wavelets; print('deps ok')"
```

## 5. 配置 Depth Anything V2

只有在需要运行 `run.py` 生成预测深度图时，才需要配置 Depth Anything V2。

项目代码中使用：

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

可以把 Depth Anything V2 源码放到项目的 `external/` 目录：

```bash
mkdir external
git clone https://github.com/DepthAnything/Depth-Anything-V2 external/Depth-Anything-V2
pip install -r external/Depth-Anything-V2/requirements.txt
```

Windows PowerShell 临时加入 `PYTHONPATH`：

```powershell
$env:PYTHONPATH="$PWD\external\Depth-Anything-V2;$env:PYTHONPATH"
```

Linux/macOS 临时加入 `PYTHONPATH`：

```bash
export PYTHONPATH="$PWD/external/Depth-Anything-V2:$PYTHONPATH"
```

验证：

```bash
python -c "from depth_anything_v2.dpt import DepthAnythingV2; print('depth anything ok')"
```

## 6. 放置权重文件

项目默认从 `pth/` 目录读取权重：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
pth/depth_anything_v2_vitl.pth
```

前两个是训练脚本使用的 Swin 和 ConvNeXt 预训练权重。第三个是 `run.py` 生成深度图时使用的 Depth Anything V2 权重。

也可以在运行时手动指定：

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

如果数据放在其他位置，运行时指定：

```bash
python train_nutrition.py --data_root /path/to/nutrition5k_dataset
python test.py --data_root /path/to/nutrition5k_dataset --ckpt ./saved/train/ckpt_best.pth
```

## 8. 快速检查

在项目根目录运行：

```bash
python -m py_compile train_nutrition.py test.py run.py utils/utils_data222.py
```

Windows 如果遇到 DataLoader 卡住或启动很慢，建议先把 `utils/utils_data222.py` 里的：

```python
num_workers=32
```

临时改成：

```python
num_workers=0
```

确认环境没问题后再逐步调到 `4`、`8` 或更高。

## 9. 运行示例

生成深度图：

```bash
python run.py --img-path /path/to/rgb_images --encoder vitl --outdir ./vis_depth
```

训练：

```bash
python train_nutrition.py --dataset nutrition_rgb_pre_d --b 8 --epoch 150 --log ./logs/train
```

测试：

```bash
python test.py --dataset nutrition_rgb_pre_d --b 8 --ckpt ./saved/train/ckpt_best.pth
```

## 10. 常见问题

### `python --version` 没有正常输出

Windows 上常见原因是系统调用了 Microsoft Store 的 Python alias。解决方式：

1. 打开 Windows 设置。
2. 搜索 App execution aliases。
3. 关闭 `python.exe` 和 `python3.exe` 的别名。
4. 重新打开终端并执行 `conda activate omnifood`。

### `torch.cuda.is_available()` 是 `False`

检查 NVIDIA 驱动是否可用：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不可用，需要先安装或更新 NVIDIA 驱动。Conda 安装的 `pytorch-cuda=11.8` 不要求你额外安装完整 CUDA Toolkit，但需要系统驱动支持对应运行时。

### `ModuleNotFoundError: depth_anything_v2`

说明 Depth Anything V2 源码没有被 Python 找到。按第 5 步配置 `PYTHONPATH`，或把 `depth_anything_v2` 包放到项目根目录。

### `ModuleNotFoundError: pytorch_wavelets`

执行：

```bash
pip install pytorch-wavelets
```

注意 pip 包名是 `pytorch-wavelets`，导入名是 `pytorch_wavelets`。

