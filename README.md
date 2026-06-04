# OmniFood8K-food

这是 OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion 的代码整理版本。项目用于从单张食物图像估计 5 个营养指标：热量、质量、脂肪、碳水化合物和蛋白质。

项目主页：[OmniFood8K-food](https://yudongjian.github.io/OmniFood8K-food/)

## 代码结构

```text
.
├── model/                 # Swin、ConvNeXt、融合网络等模型定义
├── modules/               # adapter、预测头、辅助融合模块
├── utils/                 # DataLoader、日志、工具函数
├── source/                # 论文图和展示材料
├── pth/                   # 本地预训练权重和大模型文件，不提交到 Git
├── data/                  # 本地数据集目录，不提交到 Git
├── run.py                 # 使用 Depth Anything V2 生成预测深度图
├── train_nutrition.py     # 训练入口
├── test.py                # 测试入口
└── mydataset.py           # 数据集定义
```

## 环境准备

建议使用 Python 3.8 或更高版本，并安装 PyTorch、torchvision、timm、transformers、opencv-python、open3d、tensorboardX、tqdm、ptflops、thop、pytorch_wavelets 等依赖。

推荐使用 Conda 配置环境，详细步骤见 [ENV_SETUP_CONDA.md](./ENV_SETUP_CONDA.md)。

示例：

```bash
pip install torch torchvision timm transformers opencv-python open3d tensorboardX tqdm ptflops thop pytorch_wavelets
```

如果需要运行 `run.py` 生成深度图，还需要准备 Depth Anything V2 的源码包，使下面的导入可用：

```python
from depth_anything_v2.dpt import DepthAnythingV2
```

## 权重文件

预训练权重统一放在项目根目录的 `pth/` 下。当前训练脚本默认读取：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
```

如果要用 Depth Anything V2 生成深度图，默认读取：

```text
pth/depth_anything_v2_vitl.pth
```

也可以通过参数指定其他位置：

```bash
python train_nutrition.py \
  --swin_ckpt /path/to/swin_base_patch4_window12_384_22k.pth \
  --convnext_ckpt /path/to/convnext_small_22k_1k_384.pth

python run.py \
  --img-path /path/to/images \
  --ckpt /path/to/depth_anything_v2_vitl.pth
```

## 数据集目录

默认数据集目录为：

```text
data/nutrition5k_dataset/
```

默认脚本会在其中查找：

```text
data/nutrition5k_dataset/imagery/
data/nutrition5k_dataset/imagery/txt-file/rgbd_train_processed.txt
data/nutrition5k_dataset/imagery/txt-file/rgbd_test_processed1.txt
data/nutrition5k_dataset/imagery/txt-file/rgb_in_overhead_train_processed.txt
data/nutrition5k_dataset/imagery/txt-file/rgb_in_overhead_test_processed1.txt
```

也可以指定数据路径：

```bash
python train_nutrition.py --data_root /path/to/nutrition5k_dataset
python test.py --data_root /path/to/nutrition5k_dataset --ckpt ./saved/train/ckpt_best.pth
```

如果使用其他数据集分支：

```bash
python train_nutrition.py --dataset nutrition8K --data_root_8k /path/to/nutrition8k
python train_nutrition.py --dataset 11w --data_root_11w /path/to/syn-data
```

## 生成深度图

```bash
python run.py \
  --img-path /path/to/rgb_images \
  --encoder vitl \
  --outdir ./vis_depth
```

如果只保存预测深度图，不拼接原图：

```bash
python run.py --img-path /path/to/rgb_images --pred-only --grayscale
```

## 训练

```bash
python train_nutrition.py \
  --dataset nutrition_rgb_pre_d \
  --b 8 \
  --epoch 150 \
  --log ./logs/train
```

训练日志默认写入 `logs/train/train_log.txt`，最优 checkpoint 默认保存到：

```text
saved/train/ckpt_best.pth
```

## 测试

```bash
python test.py \
  --dataset nutrition_rgb_pre_d \
  --b 8 \
  --ckpt ./saved/train/ckpt_best.pth
```

测试脚本会输出 5 个营养指标的 MAE、PMAE，以及 Mean MAE、Mean PMAE、Sum PMAE。

## 说明

- `pth/`、`data/`、`logs/`、`saved/` 等目录已加入 `.gitignore`，适合保存本地大文件和运行结果。
- 默认路径都改为项目内相对路径；如果数据或权重放在其他位置，请使用命令行参数覆盖。
- 当前代码默认使用 RGB 图像和预测深度图两路输入；点云字段虽然在数据集中读取，但主训练和测试流程没有使用。
