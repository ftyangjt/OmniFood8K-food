# OmniFood8K RGB-D Nutrition Estimation

This repository implements a computer vision course project for single-image food nutrition estimation. Given a food RGB image, the system predicts five nutrition values:

```text
Calories (kcal), Mass (g), Fat (g), Carbohydrate (g), Protein (g)
```

The current pipeline uses RGB-D input:

```text
camera_4.jpg + rgb-d.png -> RGB branch + depth branch -> feature fusion -> shared nutrition head
```

The project also provides a web demo, single-image inference, testing scripts, and an inference-time ablation experiment.

## Project Structure

```text
.
|-- scripts/                         # training, testing, inference scripts
|-- model/                           # Swin, ConvNeXt, and fusion network definitions
|-- modules/                         # depth adapter and shared nutrition head
|-- utils/                           # dataset and dataloader utilities
|-- experiments/inference_ablation/   # inference-time ablation experiment
|-- docs/                            # project notes and report-related documents
|-- external/Depth-Anything-V2/       # Depth Anything V2 dependency
|-- demo_server.py                   # backend server for web demo
|-- demo_vue.html                    # frontend GUI
|-- run_ablation.py                  # short entry for ablation experiment
|-- environment.yml                  # conda environment file
|-- requirements.txt                 # Python dependencies for common CUDA setup
|-- requirements-cu128.txt           # Python dependencies for RTX 50-series / CUDA 12.8
`-- .env.example                     # example LLM API configuration
```

Large local files are not recommended for code submission:

```text
data/
pth/
trained_weights/
logs/
outputs/
.env
```

## Installation Steps

### 1. Create Conda Environment

Recommended:

```powershell
conda env create -f environment.yml
conda activate omnifood
```

If the environment file does not work on your machine, create a clean Python environment and install dependencies manually:

```powershell
conda create -n omnifood python=3.10
conda activate omnifood
pip install -r requirements.txt
```

For RTX 50-series GPUs or CUDA 12.8 environments:

```powershell
pip install -r requirements-cu128.txt
```

### 2. Prepare External Depth Model

The project uses Depth Anything V2 to generate pseudo-depth maps for custom images and GUI inference.

Place the checkpoint here:

```text
pth/depth_anything_v2_vitl.pth
```

### 3. Prepare Nutrition Model Checkpoint

Place the trained nutrition estimation checkpoint here:

```text
trained_weights/omnifood8k/ckpt_best.pth
```

If you train a new model, update the `--ckpt` argument in testing or inference commands.

### 4. Prepare Dataset

The expected OmniFood8K dataset layout is:

```text
data/0-OminiFood8k/
|-- train_new333.txt
|-- test_new333.txt
|-- 1-data/
    |-- sample_id/
        |-- camera_4.jpg
        |-- rgb-d.png
```

Each line in `train_new333.txt` and `test_new333.txt` follows:

```text
sample_id mass_g calories_kcal protein_g fat_g carb_g
```

The current code uses `camera_4.jpg` as RGB input and `rgb-d.png` as depth input.

## Dependencies

Main dependencies include:

```text
Python 3.10
PyTorch
torchvision
numpy
pandas
opencv-python
Pillow
timm
transformers
open3d
tqdm
matplotlib
pytorch-wavelets
PyWavelets
```

See:

```text
environment.yml
requirements.txt
requirements-cu128.txt
```

## Run Instructions

### 1. Generate Missing Depth Maps

If some OmniFood8K samples do not have `rgb-d.png`, generate them first:

```powershell
python scripts\generate_8k_depth.py --data-root .\data\0-OminiFood8k --encoder vitl --ckpt .\pth\depth_anything_v2_vitl.pth
```

### 2. Train the Nutrition Model

```powershell
python scripts\train_nutrition.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 6 --epoch 150 --log .\logs\omnifood8k --save_dir .\trained_weights
```

The best checkpoint is saved to:

```text
trained_weights/omnifood8k/ckpt_best.pth
```

### 3. Test on the Test Split

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 8 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

This evaluates the model on `test_new333.txt`.

### 4. Single-Image Inference

```powershell
python scripts\infer_nutrition.py --img-path .\fqcd.jpg --ckpt .\trained_weights\omnifood8k\ckpt_best.pth --depth-ckpt .\pth\depth_anything_v2_vitl.pth --save-depth
```

Output predictions are saved to:

```text
outputs/infer_nutrition/predictions.csv
```

### 5. Web GUI Demo

Start the backend:

```powershell
python demo_server.py
```

Open the frontend in a browser:

```text
http://127.0.0.1:8000/demo_vue.html
```

Important: do not open `demo_vue.html` directly with `file://`. The page must be served by `demo_server.py`, otherwise API calls such as `/api/nutrition/predict` may fail.

Optional LLM advice can be configured through `.env`:

```text
AI_API_KEY=your_api_key
AI_BASE_URL=https://api.deepseek.com/v1
AI_MODEL=deepseek-chat
AI_API_STYLE=chat
```

Do not submit `.env`; submit only `.env.example`.

### 6. Inference-Time Ablation Experiment

Run from the project root:

```powershell
python run_ablation.py
```

If GPU memory is limited:

```powershell
python run_ablation.py --b 2
```

The ablation experiment compares:

```text
rgb_depth      : RGB + original depth
blank_depth    : RGB + zero depth
mean_depth     : RGB + mean depth
random_depth   : RGB + random depth
shuffled_depth : RGB + mismatched depth from another sample in the batch
blur_depth     : RGB + blurred depth
```

Outputs:

```text
experiments/inference_ablation/outputs/ablation_summary.csv
experiments/inference_ablation/outputs/ablation_report.md
```

## Test Cases and Corresponding Results

### Test Case 1: Standard RGB-D Testing

Command:

```powershell
python scripts\test.py --dataset nutrition8K --data_root_8k .\data\0-OminiFood8k --b 8 --ckpt .\trained_weights\omnifood8k\ckpt_best.pth
```

Input:

```text
RGB image: camera_4.jpg
Depth image: rgb-d.png
Labels: test_new333.txt
```

Metrics:

```text
MAE and PMAE for Calories, Mass, Fat, Carb, and Protein.
```

The exact numerical results depend on the checkpoint used. Save the terminal output or copy it into the final report.

### Test Case 2: Single-Image Inference

Command:

```powershell
python scripts\infer_nutrition.py --img-path .\fqcd.jpg --ckpt .\trained_weights\omnifood8k\ckpt_best.pth --depth-ckpt .\pth\depth_anything_v2_vitl.pth --save-depth
```

Expected output:

```text
Calories (kcal)
Mass (g)
Fat (g)
Carb (g)
Protein (g)
```

CSV result:

```text
outputs/infer_nutrition/predictions.csv
```

### Test Case 3: Inference-Time Depth Ablation

Command:

```powershell
python run_ablation.py
```

Observed overall PMAE comparison:

| Mode           | PMAE Mean | PMAE Std |
| -------------- | --------: | -------: |
| RGB + Depth    |    38.82% |   47.08% |
| Blank Depth    |    85.61% |   91.23% |
| Mean Depth     |    89.69% |  105.27% |
| Random Depth   |    76.88% |  103.08% |
| Shuffled Depth |    43.97% |   56.10% |
| Blur Depth     |    92.43% |  217.83% |

Interpretation:

```text
The normal RGB-D input obtains the lowest PMAE. When depth information is removed or corrupted, PMAE increases and the PMAE standard deviation also becomes larger. This suggests that the trained model depends on depth information and that depth contributes to prediction stability.
```

Note:

```text
This is an inference-time ablation study. It evaluates the sensitivity of the trained RGB-D model to corrupted depth input. A strict architectural ablation would require retraining RGB-only and RGB-D models separately.
```

## Model Summary

Current model components:

```text
RGB branch: Swin Transformer
Depth branch: DepthAdapterV4 + ConvNeXt-Tiny
Fusion module: FusionNet_3Branch_UNet_FFT
Prediction head: SharedNutritionHead
Output heads: Calories, Mass, Fat, Carb, Protein
```

Training also includes a nutrition consistency term:

```text
calories ≈ 9 * fat + 4 * carb + 4 * protein
```

This term is used as an auxiliary loss and does not modify the ground-truth labels.

## Known Limitations

```text
1. The model mainly uses camera_4.jpg and rgb-d.png, so multi-view food images are not fully utilized.
2. Food mask segmentation is not used during training or testing by default.
3. The inference-time ablation is not equivalent to retraining separate RGB-only and RGB-D models.
4. Predictions depend on the quality of the generated or precomputed depth map.
5. Large files such as datasets and checkpoints are excluded from the code submission package.
```

## Suggested Submission Package

Do not directly compress the full local workspace. Exclude large and private files:

```text
data/
pth/
trained_weights/
logs/
outputs/
.env
.git/
__pycache__/
```

Recommended submission contents:

```text
README.md or NEW_README.md
environment.yml
requirements.txt
requirements-cu128.txt
scripts/
model/
modules/
utils/
experiments/
demo_server.py
demo_vue.html
run_ablation.py
.env.example
docs/final_report.pdf
docs/presentation.pptx
```

If model weights are required by the instructor, submit them separately or provide a download link.