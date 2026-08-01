import argparse
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
from model.convnext1 import convnext_small
from model.myswinb import SwinTransformer
from modules.adapter import DepthAdapterV4
from modules.fusion import FeatureFusionNetwork222_Mask


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
    net2 = convnext_small(pretrained=False, in_22k=False).to(device)
    net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_FFT().to(device)
    adapter = DepthAdapterV4(in_ch=3, base_ch=32).to(device)

    heads = [
        FeatureFusionNetwork222_Mask(dropout=0.1).to(device),
        FeatureFusionNetwork222_Mask(dropout=0.1).to(device),
        FeatureFusionNetwork222_Mask(dropout=0.1).to(device),
        FeatureFusionNetwork222_Mask(dropout=0.05).to(device),
        FeatureFusionNetwork222_Mask(dropout=0.1).to(device),
    ]

    ckpt = torch.load(ckpt_path, map_location=device)
    required = ['net', 'net2', 'adapter', 'net_cat', 'pre_net1', 'pre_net2', 'pre_net3', 'pre_net4', 'pre_net5']
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f'Checkpoint is not a trained nutrition model. Missing keys: {missing}')

    net.load_state_dict(ckpt['net'], strict=False)
    net2.load_state_dict(ckpt['net2'], strict=False)
    adapter.load_state_dict(ckpt['adapter'], strict=False)
    net_cat.load_state_dict(ckpt['net_cat'], strict=False)
    for head, key in zip(heads, ['pre_net1', 'pre_net2', 'pre_net3', 'pre_net4', 'pre_net5']):
        head.load_state_dict(ckpt[key], strict=False)

    modules = [net, net2, net_cat, adapter, *heads]
    for module in modules:
        module.eval()
    return net, net2, net_cat, adapter, heads


def make_depth_image(depth_model, raw_image, input_size, grayscale=True):
    depth = depth_model.infer_image(raw_image, input_size)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    if grayscale:
        return np.repeat(depth[..., np.newaxis], 3, axis=-1)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    return (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


def preprocess_bgr(image_bgr):
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return transform(image_rgb).unsqueeze(0)


def predict(raw_image, depth_image, nutrition_modules, device):
    net, net2, net_cat, adapter, heads = nutrition_modules
    inputs = preprocess_bgr(raw_image).to(device)
    inputs_depth = preprocess_bgr(depth_image).to(device)

    with torch.no_grad():
        r0, r1, r2, r3, r4 = net(inputs)
        d1, d2, d3, d4 = net2(adapter(inputs_depth))
        o1, o2, o3, o4 = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])
        outputs = [head(o1, o2, o3, o4).squeeze().item() for head in heads]
    return outputs


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
            for name in os.listdir(img_path)
            if os.path.splitext(name.lower())[1] in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ]
    else:
        image_files = [img_path]

    os.makedirs(outdir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    depth_model = build_depth_model(args.encoder, depth_ckpt_path, device)
    nutrition_modules = build_nutrition_model(ckpt_path, device)

    rows = ['image,calories,mass,fat,carb,protein']
    for image_file in image_files:
        raw_image = cv2.imread(image_file)
        if raw_image is None:
            print(f'Skip unreadable image: {image_file}')
            continue

        depth_image = make_depth_image(depth_model, raw_image, args.input_size, grayscale=True)
        values = predict(raw_image, depth_image, nutrition_modules, device)
        values = [max(0.0, value) for value in values]

        print(f'\n{image_file}')
        print(f'Calories: {values[0]:.4f}')
        print(f'Mass    : {values[1]:.4f}')
        print(f'Fat     : {values[2]:.4f}')
        print(f'Carb    : {values[3]:.4f}')
        print(f'Protein : {values[4]:.4f}')

        rows.append(f'{os.path.basename(image_file)},{values[0]:.6f},{values[1]:.6f},{values[2]:.6f},{values[3]:.6f},{values[4]:.6f}')

        if args.save_depth:
            stem = os.path.splitext(os.path.basename(image_file))[0]
            cv2.imwrite(os.path.join(outdir, f'{stem}_rgb-d.png'), depth_image)

    csv_path = os.path.join(outdir, 'predictions.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')
    print(f'\nSaved predictions: {csv_path}')


if __name__ == '__main__':
    main()
