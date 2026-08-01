# OmniFood8K RGB-D Nutrition Estimation

This repository estimates five nutrition values from a single food image:

```text
Calories (kcal), Mass (g), Fat (g), Carbohydrate (g), Protein (g)
```

The checked-in inference pipeline uses RGB-D input. It generates a pseudo-depth image with Depth Anything V2, then evaluates the nutrition model.

## Model And Checkpoint Compatibility

The included nutrition checkpoint was trained with the legacy architecture below:

```text
RGB branch: Swin-Base (384 input, window size 12)
Depth branch: DepthAdapterV4 + ConvNeXt-Small
Fusion: FusionNet_3Branch_UNet_FFT
Prediction heads: five FeatureFusionNetwork222_Mask heads (pre_net1 ... pre_net5)
```

Do not use a Swin-Tiny, ConvNeXt-Tiny, or shared `nutrition_head` checkpoint with this code. Those model definitions have incompatible parameter shapes.

## Requirements

Use the provided Conda environment:

```powershell
conda env create -f environment.yml
conda activate omnifood
```

The validated environment uses Python 3.10, PyTorch 2.1.2 with CUDA 11.8, torchvision 0.16.2, OpenCV, timm, and Open3D.

Place local model files at:

```text
pth/depth_anything_v2_vitl.pth
pth/swin_base_patch4_window12_384_22k.pth
pth/convnext_small_22k_1k_384.pth
trained_weights/omnifood8k/ckpt_best.pth
```

## Dataset Layout

```text
data/0-OminiFood8k/
|-- train_new333.txt
|-- test_new333.txt
`-- 8036/                         # 1-data is also supported
    `-- sample_id/
        |-- camera_4.jpg
        `-- rgb-d.png
```

Each annotation line has this field order:

```text
sample_id mass_g calories_kj protein_g fat_g carb_g
```

The code converts calories from kJ to kcal.

Generate any missing depth images before training or testing:

```powershell
python scripts\generate_8k_depth.py --data-root .\data\0-OminiFood8k --encoder vitl --ckpt .\pth\depth_anything_v2_vitl.pth
```

## Run

Train:

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 2 --epoch 150 --log .\logs\omnifood8k --save_dir .\trained_weights
```

Test:

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 2 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

Single-image inference:

```powershell
python scripts\infer_nutrition.py --img-path .\food\fqcd.jpg --ckpt .\trained_weights\omnifood8k\ckpt_best.pth --depth-ckpt .\pth\depth_anything_v2_vitl.pth --save-depth
```

Predictions are written to `outputs/infer_nutrition/predictions.csv`.

## Web Demo

```powershell
conda activate omnifood
python demo_server.py
```

Open `http://127.0.0.1:8000/demo_vue.html`. Do not open `demo_vue.html` via `file://` because it calls local API endpoints.

Optional nutrition advice uses values from a local `.env` file:

```text
AI_API_KEY=replace_with_your_key
AI_BASE_URL=https://api.deepseek.com/v1
AI_MODEL=deepseek-chat
AI_API_STYLE=chat
```

Never commit `.env` or an API key.

## Repository Notes

Large local artifacts are intentionally ignored by Git:

```text
data/
pth/
trained_weights/
logs/
outputs/
.env
```

The legacy-checkpoint branch removes obsolete Swin-Tiny/ConvNeXt-Tiny shared-head conversion and ablation entry points. They could not load the included legacy checkpoint and would make the documented workflow misleading.
