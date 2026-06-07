# OmniFood8K-food

本仓库是 **OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion** 的本地可运行整理版本。

项目用于从食物图像估计 5 个营养指标：

```text
Calories, Mass, Fat, Carbohydrate, Protein
```

项目主页：

```text
https://yudongjian.github.io/OmniFood8K-food/
```

## 快速开始

激活 Conda 环境：

```powershell
conda activate omnifood
```

生成 OmniFood8K 缺失的预测深度图：

```powershell
python scripts\generate_8k_depth.py --data-root .\data\0-OminiFood8k --encoder vitl --ckpt .\pth\depth_anything_v2_vitl.pth
```

训练 OmniFood8K：

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 2 --epoch 150 --log .\logs\omnifood8k --save_dir .\trained_weights
```

最优权重保存到：

```text
trained_weights/omnifood8k/ckpt_best.pth
```

测试训练后的权重：

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 8 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

单张图片推理：

```powershell
python scripts\infer_nutrition.py --img-path .\fqcd.jpg --ckpt .\trained_weights\omnifood8k\ckpt_best.pth --depth-ckpt .\pth\depth_anything_v2_vitl.pth --save-depth
```

入口脚本集中在 `scripts/` 目录。请统一使用 `python scripts\...` 的命令形式。

## 仓库结构

```text
.
|-- scripts/                  # 训练、测试、推理、深度图生成脚本
|-- model/                    # Swin、ConvNeXt、融合网络等模型定义
|-- modules/                  # Adapter、融合模块、预测头
|-- utils/                    # DataLoader 和工具函数
|-- docs/                     # 环境、数据集、使用说明
|-- external/Depth-Anything-V2/
|-- source/                   # 论文图和展示材料
|-- data/                     # 本地数据集，不提交 Git
|-- pth/                      # 下载的预训练权重，不提交 Git
|-- trained_weights/          # 本地训练输出，.pth 不提交 Git
|-- logs/                     # 训练日志，不提交 Git
`-- outputs/                  # 推理输出，不提交 Git
```

## 文档

- [Conda 环境配置](docs/ENV_SETUP_CONDA.md)
- [代码使用指南](docs/代码使用指南.md)
- [数据集目录结构](docs/DATASET_STRUCTURE.md)
- [项目思路及流程](docs/原项目思路及流程.md)
- [原始 README 中文整理](docs/Original_README.md)

## 必要权重

下载或准备以下权重，并放在 `pth/`：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
pth/depth_anything_v2_vitl.pth
```

训练得到的营养估计模型建议放在：

```text
trained_weights/
```

## 本地目录约定

以下目录或文件默认不提交 Git：

```text
data/
pth/
logs/
outputs/
trained_weights/*.pth
```

仓库保留 `trained_weights/.gitkeep`，用于保留目录本身。实际 `.pth` 权重文件只保留在本地。

## 注意事项

- Windows 下 `scripts/train_nutrition.py` 和 `scripts/test.py` 默认使用 `--num_workers 0`，避免 DataLoader 多进程递归启动问题。
- `nutrition8K` 每个样本目录都需要同时存在 `camera_4.jpg` 和 `rgb-d.png`。
- `scripts/test.py` 用于测试数据集划分；自定义图片请使用 `scripts/infer_nutrition.py`。
