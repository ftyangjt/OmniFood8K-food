import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.convnext1 import model_urls


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def download_convnext_tiny():
    output_path = project_path("pth", "convnext_tiny_1k_224_ema.pth")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        print(f"ConvNeXt-Tiny checkpoint already exists: {output_path}")
        return

    checkpoint = torch.hub.load_state_dict_from_url(
        url=model_urls["convnext_tiny_1k"],
        map_location="cpu",
        check_hash=True,
    )
    torch.save(checkpoint, output_path)
    print(f"Saved ConvNeXt-Tiny checkpoint to: {output_path}")


def main():
    download_convnext_tiny()
    print("Swin-Tiny checkpoint is not downloaded here because checkpoint formats vary by source.")
    print("Pass it with --swin_ckpt when you have a compatible file.")


if __name__ == "__main__":
    main()
