# 推理阶段消融实验

这个文件夹用于做 inference-time ablation，也就是不重新训练模型，只在测试集推理时改变 Depth 输入，观察当前模型对深度图的依赖程度。

## 最短运行命令

推荐先进入项目根目录：

```powershell
cd C:\Users\13786\Desktop\Deep_Learning\OmniFood8K-food
conda activate omnifood
python run_ablation.py
```

默认会使用：

```text
trained_weights/omnifood8k/ckpt_best.pth
```

如果显存不够，把 batch size 调小：

```powershell
python run_ablation.py --b 2
```

如果你当前站在上一级目录 `C:\Users\13786\Desktop\Deep_Learning`，则运行：

```powershell
python .\OmniFood8K-food\run_ablation.py
```

## 使用其他权重

如果你重新训练后的权重在别的位置，再手动指定：

```powershell
python run_ablation.py --ckpt .\saved\train\ckpt_best.pth
```

## 默认比较内容

脚本默认比较 6 组：

| 模式 | 含义 |
|---|---|
| `rgb_depth` | 正常输入，RGB + 原始 Depth |
| `blank_depth` | RGB + 全 0 空白 Depth |
| `mean_depth` | RGB + 每张图的平均 Depth |
| `random_depth` | RGB + 随机噪声 Depth |
| `shuffled_depth` | RGB + batch 内打乱后的 Depth |
| `blur_depth` | RGB + 模糊后的 Depth |

其中 `rgb_depth` 是基准组，其他都是推理阶段消融组。

## 输出文件

运行后会生成：

```text
experiments/inference_ablation/outputs/ablation_summary.csv
experiments/inference_ablation/outputs/ablation_report.md
```

`ablation_summary.csv` 适合后续画图或放进表格。

`ablation_report.md` 是已经整理好的 Markdown 报告，可以直接复制到实验报告里。

## 指标说明

| 指标 | 含义 |
|---|---|
| MAE | 平均绝对误差，表示预测值实际错了多少 |
| PMAE mean | 每个样本百分比绝对误差的平均值 |
| PMAE std | 每个样本百分比绝对误差的标准差，表示稳定性 |
| PMAE median | 百分比误差中位数，减少极端样本影响 |
| PMAE max | 最大百分比误差，观察最差情况 |

PMAE 计算方式：

```text
PMAE = abs(prediction - label) / label
```

报告里一般写成百分比，例如 `18.5%`。

## 如何写结论

如果结果显示：

```text
rgb_depth PMAE 最低
blank_depth / random_depth PMAE 明显升高
```

可以说明：

> 在不重新训练模型的条件下，破坏深度图会导致 PMAE 上升，说明当前模型对深度信息具有依赖性。深度图能够为食物体积、质量和热量估计提供额外空间信息。

如果 `PMAE std` 也明显变大，可以补充：

> 消融深度图后，PMAE 标准差增大，说明模型在不同样本上的误差波动更大，预测稳定性下降。

## 注意事项

这个实验是推理阶段消融，不是严格的重新训练消融。

它能说明：

```text
当前训练好的 RGB-D 模型对 Depth 输入是否敏感。
```

它不能完全证明：

```text
重新训练 RGB-only 模型一定比 RGB-D 模型差。
```

如果要做严格消融，需要分别重新训练：

```text
RGB only
RGB + Depth
RGB + Depth + Food Mask
RGB + Depth + Calories Consistency
```

然后在同一个测试集上比较 MAE / PMAE。
