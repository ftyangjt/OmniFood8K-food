import argparse
import csv
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import dual_swin_convnext
from model.convnext1 import convnext_tiny
from model.myswinb import SwinTransformer
from modules.adapter import DepthAdapterV4
from modules.fusion import SharedNutritionHead


DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, 'external', 'Depth-Anything-V2')
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else project_path(path)


def load_strict_state_dict(model, state_dict, name):
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f'{name} checkpoint does not exactly match the current inference model. '
            'Use a checkpoint trained with the current Swin-T/ConvNeXt-Tiny/shared-head code.'
        ) from exc


def build_depth_model(encoder, ckpt_path, device):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.to(device).eval()


def build_nutrition_model(ckpt_path, device):
    net = SwinTransformer().to(device)
    net2 = convnext_tiny(pretrained=False, in_22k=False).to(device)
    net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_FFT().to(device)
    adapter = DepthAdapterV4(in_ch=3, base_ch=32).to(device)
    nutrition_head = SharedNutritionHead(dropout=0.1).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    required = ['net', 'net2', 'adapter', 'net_cat', 'nutrition_head']
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f'Checkpoint is not a trained nutrition model. Missing keys: {missing}')

    load_strict_state_dict(net, ckpt['net'], 'net')
    load_strict_state_dict(net2, ckpt['net2'], 'net2')
    load_strict_state_dict(adapter, ckpt['adapter'], 'adapter')
    load_strict_state_dict(net_cat, ckpt['net_cat'], 'net_cat')
    load_strict_state_dict(nutrition_head, ckpt['nutrition_head'], 'nutrition_head')

    modules = [net, net2, net_cat, adapter, nutrition_head]
    for module in modules:
        module.eval()
    return modules


def make_depth_image(depth_model, raw_image, input_size, grayscale=True):
    depth = depth_model.infer_image(raw_image, input_size)
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max <= depth_min:
        depth = np.zeros(depth.shape, dtype=np.uint8)
    else:
        depth = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
    if grayscale:
        return np.repeat(depth[..., np.newaxis], 3, axis=-1)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def segment_food_grabcut(image_bgr, margin_ratio=0.08, iterations=5):
    height, width = image_bgr.shape[:2]
    margin_x = max(1, int(width * margin_ratio))
    margin_y = max(1, int(height * margin_ratio))
    rect = (
        margin_x,
        margin_y,
        max(1, width - 2 * margin_x),
        max(1, height - 2 * margin_y),
    )
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        food_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        food_mask = np.full((height, width), 255, dtype=np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel)
    if cv2.countNonZero(food_mask) == 0:
        food_mask = np.full((height, width), 255, dtype=np.uint8)
    return food_mask


def apply_food_mask(image_bgr, mask):
    masked = image_bgr.copy()
    masked[mask == 0] = 0
    return masked


def preprocess_bgr(image_bgr):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return transform(image_rgb).unsqueeze(0)


def predict_batch(raw_images, depth_images, nutrition_modules, device):
    net, net2, net_cat, adapter, nutrition_head = nutrition_modules
    inputs = torch.cat([preprocess_bgr(image) for image in raw_images], dim=0).to(device)
    inputs_depth = torch.cat([preprocess_bgr(image) for image in depth_images], dim=0).to(device)

    with torch.no_grad():
        r0, r1, r2, r3, r4 = net(inputs)
        d1, d2, d3, d4 = net2(adapter(inputs_depth))
        o1, o2, o3, o4 = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])
        outputs = nutrition_head(o1, o2, o3, o4).cpu().tolist()
    return outputs


def predict(raw_image, depth_image, nutrition_modules, device):
    return predict_batch([raw_image], [depth_image], nutrition_modules, device)[0]


def main():
    parser = argparse.ArgumentParser(description='Predict nutrition values for custom food images')
    parser.add_argument('--img-path', type=str, required=True, help='input image or image directory')
    parser.add_argument('--ckpt', type=str, required=True, help='trained nutrition checkpoint, e.g. ./saved/omnifood8k/ckpt_best.pth')
    parser.add_argument('--depth-ckpt', type=str, default=None,
                        help='Depth Anything V2 checkpoint. Defaults to ./pth/depth_anything_v2_{encoder}.pth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./outputs/infer_nutrition')
    parser.add_argument('--save-depth', action='store_true', help='save generated depth maps beside prediction csv')
    parser.add_argument('--use-food-mask', action='store_true',
                        help='segment the food region with GrabCut and mask background before nutrition inference')
    parser.add_argument('--save-mask', action='store_true', help='save generated food masks')
    parser.add_argument('--mask-margin', type=float, default=0.08,
                        help='GrabCut rectangle margin ratio used by --use-food-mask')
    parser.add_argument('--batch-size', type=int, default=8, help='nutrition model batch size for image-directory inference')
    args = parser.parse_args()

    img_path = resolve_path(args.img_path)
    ckpt_path = resolve_path(args.ckpt)
    depth_ckpt_path = resolve_path(args.depth_ckpt) if args.depth_ckpt else project_path('pth', f'depth_anything_v2_{args.encoder}.pth')
    outdir = resolve_path(args.outdir)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Nutrition checkpoint not found: {ckpt_path}')
    if not os.path.exists(depth_ckpt_path):
        raise FileNotFoundError(f'Depth checkpoint not found: {depth_ckpt_path}')

    if os.path.isdir(img_path):
        image_files = [
            os.path.join(img_path, name)
            for name in sorted(os.listdir(img_path))
            if os.path.splitext(name.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]
    else:
        image_files = [img_path]

    os.makedirs(outdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_model = build_depth_model(args.encoder, depth_ckpt_path, device)
    nutrition_modules = build_nutrition_model(ckpt_path, device)

    csv_rows = []
    batch_raw_images = []
    batch_depth_images = []
    batch_records = []

    def flush_batch():
        if not batch_raw_images:
            return

        batch_values = predict_batch(batch_raw_images, batch_depth_images, nutrition_modules, device)
        for record, values in zip(batch_records, batch_values):
            values = [max(0.0, float(value)) for value in values]
            image_file = record['image_path']

            print(f'\n{image_file}')
            print(f'Calories (kcal): {values[0]:.4f}')
            print(f'Mass    : {values[1]:.4f}')
            print(f'Fat     : {values[2]:.4f}')
            print(f'Carb    : {values[3]:.4f}')
            print(f'Protein : {values[4]:.4f}')

            csv_rows.append({
                'image': os.path.basename(image_file),
                'image_path': image_file,
                'depth_path': record['depth_path'],
                'mask_path': record['mask_path'],
                'calories_kcal': f'{values[0]:.6f}',
                'mass_g': f'{values[1]:.6f}',
                'fat_g': f'{values[2]:.6f}',
                'carb_g': f'{values[3]:.6f}',
                'protein_g': f'{values[4]:.6f}',
            })

        batch_raw_images.clear()
        batch_depth_images.clear()
        batch_records.clear()

    for image_file in image_files:
        raw_image = cv2.imread(image_file)
        if raw_image is None:
            print(f'Skip unreadable image: {image_file}')
            continue

        mask_path = ''
        inference_image = raw_image
        if args.use_food_mask:
            food_mask = segment_food_grabcut(raw_image, margin_ratio=args.mask_margin)
            inference_image = apply_food_mask(raw_image, food_mask)
            if args.save_mask:
                stem = os.path.splitext(os.path.basename(image_file))[0]
                mask_path = os.path.join(outdir, f'{stem}_food-mask.png')
                cv2.imwrite(mask_path, food_mask)

        depth_image = make_depth_image(depth_model, raw_image, args.input_size, grayscale=True)
        if args.use_food_mask:
            depth_image = apply_food_mask(depth_image, food_mask)

        depth_path = ''
        if args.save_depth:
            stem = os.path.splitext(os.path.basename(image_file))[0]
            depth_path = os.path.join(outdir, f'{stem}_rgb-d.png')
            cv2.imwrite(depth_path, depth_image)

        batch_raw_images.append(inference_image)
        batch_depth_images.append(depth_image)
        batch_records.append({'image_path': image_file, 'depth_path': depth_path, 'mask_path': mask_path})
        if len(batch_raw_images) >= args.batch_size:
            flush_batch()

    flush_batch()

    csv_path = os.path.join(outdir, 'predictions.csv')
    fieldnames = ['image', 'image_path', 'depth_path', 'mask_path', 'calories_kcal', 'mass_g', 'fat_g', 'carb_g', 'protein_g']
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'\nSaved predictions: {csv_path}')


if __name__ == '__main__':
    main()
