import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.fusion import FeatureFusionNetwork222_Mask, SharedNutritionHead


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_path(path):
    return path if os.path.isabs(path) else project_path(path)


def build_legacy_heads(checkpoint):
    legacy_keys = ["pre_net1", "pre_net2", "pre_net3", "pre_net4", "pre_net5"]
    missing = [key for key in legacy_keys if key not in checkpoint]
    if missing:
        raise KeyError(f"Legacy checkpoint is missing keys: {missing}")

    dropouts = [0.1, 0.1, 0.1, 0.05, 0.1]
    heads = [FeatureFusionNetwork222_Mask(dropout=dropout) for dropout in dropouts]
    for head, key in zip(heads, legacy_keys):
        head.load_state_dict(checkpoint[key], strict=True)
        head.eval()
    return heads


def convert_checkpoint(input_path, output_path):
    checkpoint = torch.load(input_path, map_location="cpu")
    legacy_heads = build_legacy_heads(checkpoint)

    nutrition_head = SharedNutritionHead(dropout=0.1)
    nutrition_head.initialize_from_legacy_heads(legacy_heads)

    converted = dict(checkpoint)
    for key in ["pre_net1", "pre_net2", "pre_net3", "pre_net4", "pre_net5"]:
        converted.pop(key, None)
    converted["nutrition_head"] = nutrition_head.state_dict()
    converted["head_init"] = {
        "source": "legacy_pre_net_average",
        "legacy_checkpoint": os.path.abspath(input_path),
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(converted, output_path)
    print(f"Converted checkpoint saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert legacy five-head checkpoints to the shared nutrition head.")
    parser.add_argument("--input", required=True, help="legacy checkpoint with pre_net1...pre_net5")
    parser.add_argument("--output", required=True, help="output checkpoint with nutrition_head")
    args = parser.parse_args()

    convert_checkpoint(resolve_path(args.input), resolve_path(args.output))


if __name__ == "__main__":
    main()
