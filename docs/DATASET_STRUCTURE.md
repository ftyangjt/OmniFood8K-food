# 数据集目录结构说明

本文档说明当前代码期望的数据集目录结构、标注文件格式和常用命令。

## OmniFood8K

使用参数：

```text
--dataset nutrition8K
```

默认根目录：

```text
data/0-OminiFood8k/
```

期望结构：

```text
data/0-OminiFood8k/
  train_new333.txt
  test_new333.txt
  8036/
    1/
      camera_4.jpg
      rgb-d.png
    2/
      camera_4.jpg
      rgb-d.png
    ...
```

代码也兼容 `1-data/` 目录。如果数据根目录下存在 `1-data/`，会优先使用它；否则使用 `8036/`。

相关逻辑位于：

```text
utils/utils_data222.py
```

## OmniFood8K 标注文件格式

`train_new333.txt` 和 `test_new333.txt` 每一行按空格切分，字段顺序为：

```text
sample_id mass calories protein fat carb
```

示例：

```text
2 106.8 854.492423076923 10.7779826259947 15.2960192307692 6.19553315649867
```

代码中的字段映射：

```text
图片路径    8036/<sample_id>/camera_4.jpg
深度图路径  8036/<sample_id>/rgb-d.png
mass        第 2 列
calories    第 3 列
protein     第 4 列
fat         第 5 列
carb        第 6 列
```

## 生成 rgb-d.png

训练或测试 `nutrition8K` 前，每个样本目录都需要：

```text
camera_4.jpg
rgb-d.png
```

批量生成命令：

```powershell
python scripts\generate_8k_depth.py --data-root .\data\0-OminiFood8k --encoder vitl --ckpt .\pth\depth_anything_v2_vitl.pth
```

脚本会读取：

```text
train_new333.txt
test_new333.txt
```

然后把深度图写入：

```text
8036/<sample_id>/rgb-d.png
```

## Nutrition5K 风格数据

使用参数：

```text
--dataset nutrition_rgb_pre_d
--dataset nutrition_rgbd
```

默认根目录：

```text
data/nutrition5k_dataset/
```

期望结构：

```text
data/nutrition5k_dataset/
  imagery/
    txt-file/
      rgbd_train_processed.txt
      rgbd_test_processed1.txt
      rgb_in_overhead_train_processed.txt
      rgb_in_overhead_test_processed1.txt
    ...
```

Nutrition5K 风格文本行按如下顺序解析：

```text
image_path label calories mass fat carb protein
```

对于 `nutrition_rgb_pre_d`，代码会把深度图路径中的：

```text
depth_color.png
```

替换为：

```text
rgb-d.png
```

所以数据目录中需要存在对应的预测深度图。

## Synthetic 11w 数据

使用参数：

```text
--dataset 11w
```

默认根目录：

```text
data/syn-data/
```

期望结构：

```text
data/syn-data/
  train2.txt
  test2.txt
  ...
```

文本行按如下顺序解析：

```text
image_id mass calories protein fat carb
```

## 常用命令

训练 OmniFood8K：

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 2 --epoch 150 --log .\logs\omnifood8k --save_dir .\trained_weights
```

测试 OmniFood8K：

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 8 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

自定义图片推理：

```powershell
python scripts\infer_nutrition.py --img-path .\your_food.jpg --ckpt .\trained_weights\omnifood8k\ckpt_best.pth --depth-ckpt .\pth\depth_anything_v2_vitl.pth --save-depth
```
