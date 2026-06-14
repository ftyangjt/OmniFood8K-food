我看了当前训练与模型代码，整体架构大概是：

RGB 分支：自定义 `SwinTransformer`  
深度/伪深度分支：`DepthAdapterV4` + `ConvNeXt-Small`  
融合：低层用 FFT 高频/低频融合，高层直接拼接  
预测：5 个独立 `FeatureFusionNetwork222_Mask` 头分别预测 calories / mass / fat / carb / protein  
主要代码在 [scripts/train_nutrition.py](f:/work_new/Deeplearning_big/OmniFood8K-food/scripts/train_nutrition.py)、[model/dual_swin_convnext.py](f:/work_new/Deeplearning_big/OmniFood8K-food/model/dual_swin_convnext.py)、[modules/fusion.py](f:/work_new/Deeplearning_big/OmniFood8K-food/modules/fusion.py)。

**最优先改的点**

1. **测试阶段预测头没有切到 `eval()`**
   
   `test()` 里设置了 `net/net2/net_cat/adapter.eval()`，但没有设置 `pre_net1` 到 `pre_net5` 的 `eval()`。这些头里面有 `Dropout`，所以测试结果会带随机性，也会影响 best checkpoint 判断。这个是非常值得优先修的点。

2. **`squeeze()` 可能在 batch size = 1 时出错**
   
   现在是：
   
   ```python
   outputs[0] = pre_net1(...).squeeze()
   ```
   
   如果最后一个 batch 只有 1 张图，会把 `[1, 1]` 压成标量，后面按 batch 索引或算 loss 可能不稳定。建议改成：
   
   ```python
   .squeeze(-1)
   ```

3. **动态任务权重里 fat 写错了**
   
   当前更新 KPI 时用了：
   
   ```python
   [calories_loss, mass_loss, mass_loss, carb_loss, protein_loss]
   ```
   
   第三个应该是 `fat_loss`。这会导致脂肪任务的权重调度不正确。

4. **当前 alignment loss 语义可能不对**
   
   `inter_modal_alignment_loss(o1)` 假设 `o1` 的前半通道是 RGB、后半通道是 Depth。但 `o1` 已经经过 `MultiFreqFusion` 和卷积混合，通道不再严格代表 RGB/Depth。这个 loss 可能不是“模态对齐”，甚至可能干扰训练。更合理的是在融合前对 `r1` 和 `d1` 做 projection 后再对齐。

**模型结构改进**

1. **把 5 个完全独立预测头改成共享 trunk + 多任务 heads**
   
   现在 5 个 `FeatureFusionNetwork222_Mask` 是重复结构，参数多，而且 5 个营养值之间没有显式关系。建议改成一个共享融合特征提取器，再接 5 个轻量 head。这样更稳，也更容易让 calories、mass、fat、carb、protein 共享食物体积、类别、纹理等信息。

2. **加入营养一致性约束**
   
   热量和三大营养素有物理关系：
   
   ```text
   calories ≈ 9 * fat + 4 * carb + 4 * protein
   ```
   
   可以加一个 consistency loss，或者让模型预测 mass/fat/carb/protein，再派生一个 calories 辅助约束。这个对营养估计任务很有价值。

3. **输出加非负约束**
   
   当前最后是线性层，可能输出负数。营养值天然非负，建议用 `Softplus` 或者预测 `log(1 + y)`，推理时再反变换。通常会让小目标如 fat/protein 更稳定。

4. **深度分支需要更强的尺度信息**
   
   如果 `inputs_rgbd` 是 Depth Anything 生成的伪深度，它主要是相对深度，不一定包含真实尺度。营养估计尤其依赖份量/体积，建议引入：
   
   - 食物/盘子分割 mask
   - 盘子直径或相机尺度归一化
   - metric depth 或 HHA 表示
   - 基于 mask 的深度体积特征

5. **FFT 融合可以改成可学习频域融合**
   
   现在 `MultiFreqFusion` 用固定 `freq_threshold=0.1` 切高低频，比较硬。可以尝试：
   
   - 可学习频域 mask
   - 每通道不同频率权重
   - wavelet/DWT 多尺度融合
   - RGB-depth cross attention 替代简单相加

**训练策略改进**

1. 使用 `AdamW`、warmup + cosine，当前只有 cosine，没有 warmup。
2. batch size 默认是 2，建议用梯度累积扩大有效 batch。
3. 加 AMP 混合精度，能省显存，也能支持更大 batch。
4. 先冻结 backbone 训练融合头，再逐步解冻 Swin/ConvNeXt，通常比一开始全量微调稳。
5. 当前 loss 是 batch 内 PMAE 形式，小 batch 下波动会很大。可以尝试 `SmoothL1Loss`、log-space loss、或者 dataset-level target normalization。

**数据与评估改进**

1. `nutrition_rgb_pre_d` 训练增强基本没开，建议适度加入 resize crop、水平翻转、轻量颜色扰动，但颜色增强别太重，因为食物颜色和类别相关。
2. 确认 train/test 是否按 dish 级别划分，避免同一菜品或近似图片泄漏。
3. 加 RGB-only、Depth-only、No-FFT、No-alignment 的 ablation，这能快速判断复杂融合模块到底有没有贡献。
4. 记录每个任务的 MAE、PMAE、RMSE、R² 和最差样本可视化，比只看总 PMAE 更容易定位问题。

我的建议优先级是：先修 `eval()`、`squeeze(-1)`、fat 权重 bug 和 alignment loss 位置；然后做“共享 trunk + 多任务 heads + 非负输出 + calories 一致性约束”。这些改动成本不高，但很可能比继续堆复杂模块更直接地提高稳定性和指标。