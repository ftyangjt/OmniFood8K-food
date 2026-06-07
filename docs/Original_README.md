# 原始 README 中文整理

本文档根据原始项目 README 整理，用于保留作者发布时提供的关键信息。

## 项目名称

```text
OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion
```

项目主页：

```text
https://yudongjian.github.io/OmniFood8K-food/
```

## 论文信息

该工作被 CVPR 2026 接收。

作者：

```text
Dongjian Yu, Weiqing Min, Qian Jiang, Xing Lin, Xin Jin, Shuqiang Jiang
```

单位：

```text
Yunnan University
Key Laboratory of Intelligent Information Processing,
Institute of Computing Technology,
Chinese Academy of Sciences
```

## 使用前准备 1：下载预训练权重

原始 README 要求先下载 Swin Transformer 和 ConvNeXt 的预训练权重。

下载链接：

```text
https://drive.google.com/drive/folders/1i-AExbFDi4cLy_OPYUmGm_q5f8EITpjJ?usp=drive_link
```

下载后放入项目根目录的：

```text
pth/
```

当前本地代码默认使用：

```text
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
```

## 使用前准备 2：生成预测深度图

原始 README 说明需要生成预测深度图，并提到项目中的 `run.py` 可作为参考。

Depth Anything V2 项目地址：

```text
https://github.com/DepthAnything/Depth-Anything-V2
```

当前本地批量生成 OmniFood8K 深度图可以运行：

```powershell
python scripts\generate_8k_depth.py --data-root .\data\0-OminiFood8k --encoder vitl --ckpt .\pth\depth_anything_v2_vitl.pth
```

## 原始训练命令

原始 README 给出的训练命令为：

```bash
train_nutrition.py --b 8 --log ./logs/test3
```

当前本地建议显式指定数据集和权重保存目录：

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 2 --epoch 150 --log .\logs\omnifood8k --save_dir .\trained_weights
```

## 原始测试命令

原始 README 给出的测试命令为：

```bash
python test.py --ckpt ./saved/logs/test2/ckpt_best.pth
```

当前本地建议使用：

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 8 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

## 原始训练后权重

原始 README 提供了训练后权重下载链接：

```text
https://drive.google.com/file/d/1aeRV_Ag2m4YlYlWEYVpZHcVRlDoaxLEW/view?usp=sharing
```

注意：该权重是否对应当前本地的 `nutrition8K` 数据划分和当前生成的 `rgb-d.png`，需要进一步向作者确认。如果直接测试结果明显偏离论文，优先检查：

```text
1. checkpoint 对应的数据集
2. 深度图生成方式
3. train/test 划分文件
4. 标注字段顺序
```

## 作者联系方式

原始 README 给出的联系方式：

```text
yudongjian@stu.ynu.edu.cn
```
