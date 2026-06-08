import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEPTH_ANYTHING_ROOT = os.path.join(PROJECT_ROOT, "external", "Depth-Anything-V2")
if DEPTH_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, DEPTH_ANYTHING_ROOT)


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_path(path):
    return path if os.path.isabs(path) else project_path(path)


def find_image_root(data_root):
    for dirname in ("1-data", "8036"):
        candidate = os.path.join(data_root, dirname)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"OmniFood8K image directory not found under {data_root}. "
        "Expected one of: 1-data, 8036."
    )


def read_split_ids(data_root):
    ids = []
    for filename in ("train_new333.txt", "test_new333.txt"):
        split_path = os.path.join(data_root, filename)
        if not os.path.exists(split_path):
            return None
        with open(split_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if parts:
                    ids.append(parts[0])
    return sorted(set(ids), key=lambda value: int(value) if value.isdigit() else value)


def has_valid_image(path):
    if not os.path.exists(path):
        return False
    try:
        import cv2

        return cv2.imread(path) is not None
    except Exception:
        return False


def collect_samples(data_root, image_root, overwrite, all_dirs):
    jobs = []
    names = None if all_dirs else read_split_ids(data_root)
    if names is None:
        names = sorted(os.listdir(image_root), key=lambda value: int(value) if value.isdigit() else value)

    for name in names:
        sample_dir = os.path.join(image_root, name)
        if not os.path.isdir(sample_dir):
            continue

        image_path = os.path.join(sample_dir, "camera_4.jpg")
        output_path = os.path.join(sample_dir, "rgb-d.png")
        if not os.path.exists(image_path):
            continue
        if not overwrite and has_valid_image(output_path):
            continue
        jobs.append((image_path, output_path))
    return jobs


def normalize_depth(depth):
    import numpy as np

    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)
    depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
    return depth.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Generate rgb-d.png for OmniFood8K samples.")
    parser.add_argument("--data-root", type=str, default="./data/0-OminiFood8k",
                        help="OmniFood8K root containing train_new333.txt, test_new333.txt, and 8036/ or 1-data/.")
    parser.add_argument("--ckpt", type=str, default="./pth/depth_anything_v2_vitl.pth",
                        help="Depth Anything V2 checkpoint path.")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518,
                        help="Depth Anything input size. The official default is 518.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate rgb-d.png even if it already exists.")
    parser.add_argument("--all-dirs", action="store_true",
                        help="Generate for every directory under the image root instead of train/test split ids.")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N missing files.")
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    ckpt_path = resolve_path(args.ckpt)
    image_root = find_image_root(data_root)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {ckpt_path}")

    jobs = collect_samples(data_root, image_root, args.overwrite, args.all_dirs)
    total_jobs = len(jobs)
    if args.limit is not None:
        jobs = jobs[:args.limit]

    print(f"Data root: {data_root}")
    print(f"Image root: {image_root}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Missing files found: {total_jobs}")
    print(f"Files to generate in this run: {len(jobs)}")

    if not jobs:
        if total_jobs:
            print("Nothing generated because --limit selected zero files.")
        else:
            print("Nothing to do. All rgb-d.png files already exist.")
        return

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm
    from depth_anything_v2.dpt import DepthAnythingV2

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    with torch.no_grad():
        for image_path, output_path in tqdm(jobs, desc="Generating depth"):
            raw_image = cv2.imread(image_path)
            if raw_image is None:
                print(f"Skip unreadable image: {image_path}")
                continue

            depth = model.infer_image(raw_image, args.input_size)
            depth = normalize_depth(depth)
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

            if not cv2.imwrite(output_path, depth):
                raise RuntimeError(f"Failed to write depth image: {output_path}")


if __name__ == "__main__":
    main()
