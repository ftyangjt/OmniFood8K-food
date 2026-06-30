import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import dual_swin_convnext
from model.convnext1 import convnext_tiny
from model.myswinb import SwinTransformer
from modules.adapter import DepthAdapterV4
from modules.fusion import SharedNutritionHead
from utils.utils_data222 import get_DataLoader


TARGETS = [
    ("calories", "Calories", "kcal"),
    ("mass", "Mass", "g"),
    ("fat", "Fat", "g"),
    ("carb", "Carb", "g"),
    ("protein", "Protein", "g"),
]


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    candidate = os.path.abspath(path)
    if os.path.exists(candidate):
        return candidate
    return project_path(path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_strict_state_dict(model, state_dict, name):
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{name} checkpoint does not exactly match the current model. "
            "Use a checkpoint trained with the current Swin-T/ConvNeXt-Tiny/shared-head code."
        ) from exc


def build_model(ckpt_path, device):
    net = SwinTransformer().to(device)
    net2 = convnext_tiny(pretrained=False, in_22k=False).to(device)
    net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_FFT().to(device)
    adapter = DepthAdapterV4(in_ch=3, base_ch=32).to(device)
    nutrition_head = SharedNutritionHead(dropout=0.1).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    required = ["net", "net2", "adapter", "net_cat", "nutrition_head"]
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint is not a shared-head nutrition model. Missing keys: {missing}")

    load_strict_state_dict(net, ckpt["net"], "net")
    load_strict_state_dict(net2, ckpt["net2"], "net2")
    load_strict_state_dict(adapter, ckpt["adapter"], "adapter")
    load_strict_state_dict(net_cat, ckpt["net_cat"], "net_cat")
    load_strict_state_dict(nutrition_head, ckpt["nutrition_head"], "nutrition_head")

    modules = [net, net2, net_cat, adapter, nutrition_head]
    for module in modules:
        module.eval()
    return modules


def forward_once(inputs, depth_inputs, modules):
    net, net2, net_cat, adapter, nutrition_head = modules
    _, r1, r2, r3, r4 = net(inputs)
    d1, d2, d3, d4 = net2(adapter(depth_inputs))
    o1, o2, o3, o4 = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])
    return nutrition_head(o1, o2, o3, o4)


def ablate_depth(depth_inputs, mode, blur_kernel):
    if mode == "rgb_depth":
        return depth_inputs
    if mode == "blank_depth":
        return torch.zeros_like(depth_inputs)
    if mode == "mean_depth":
        return depth_inputs.mean(dim=(2, 3), keepdim=True).expand_as(depth_inputs)
    if mode == "random_depth":
        return torch.randn_like(depth_inputs)
    if mode == "shuffled_depth":
        if depth_inputs.shape[0] <= 1:
            return depth_inputs.flip(dims=[0])
        return depth_inputs[torch.randperm(depth_inputs.shape[0], device=depth_inputs.device)]
    if mode == "blur_depth":
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        return F.avg_pool2d(depth_inputs, kernel_size=kernel, stride=1, padding=kernel // 2)
    raise ValueError(f"Unknown ablation mode: {mode}")


def tensor_to_rows(tensor):
    return tensor.detach().cpu().numpy().astype(float)


def summarize_mode(mode, predictions, labels, eps):
    abs_errors = np.abs(predictions - labels)
    safe_labels = np.maximum(np.abs(labels), eps)
    pmae_samples = abs_errors / safe_labels

    rows = []
    for index, (key, display_name, unit) in enumerate(TARGETS):
        target_pmae = pmae_samples[:, index]
        rows.append({
            "mode": mode,
            "target": key,
            "target_name": display_name,
            "unit": unit,
            "mae": float(abs_errors[:, index].mean()),
            "pmae_mean": float(target_pmae.mean()),
            "pmae_std": float(target_pmae.std(ddof=0)),
            "pmae_median": float(np.median(target_pmae)),
            "pmae_max": float(target_pmae.max()),
            "num_samples": int(labels.shape[0]),
        })

    rows.append({
        "mode": mode,
        "target": "average",
        "target_name": "Average",
        "unit": "-",
        "mae": float(abs_errors.mean()),
        "pmae_mean": float(pmae_samples.mean()),
        "pmae_std": float(pmae_samples.std(ddof=0)),
        "pmae_median": float(np.median(pmae_samples)),
        "pmae_max": float(pmae_samples.max()),
        "num_samples": int(labels.shape[0]),
    })
    return rows


def format_percent(value):
    return f"{value * 100:.2f}%"


def write_summary_csv(rows, out_path):
    fieldnames = [
        "mode",
        "target",
        "target_name",
        "unit",
        "mae",
        "pmae_mean",
        "pmae_std",
        "pmae_median",
        "pmae_max",
        "num_samples",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, args, out_path):
    by_target = {}
    for row in rows:
        by_target.setdefault(row["target"], []).append(row)

    lines = [
        "# 推理阶段消融实验结果",
        "",
        "本实验不重新训练模型，只在测试集推理时改变深度图输入，用于观察当前模型对 Depth 分支的依赖程度。",
        "",
        "## 实验配置",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Dataset: `{args.dataset}`",
        f"- Data root: `{args.data_root_8k}`",
        f"- Batch size: `{args.b}`",
        f"- Modes: `{', '.join(args.modes)}`",
        "",
        "## 指标说明",
        "",
        "- MAE: 平均绝对误差，表示实际错了多少。",
        "- PMAE mean: 每个样本百分比绝对误差的平均值。",
        "- PMAE std: 每个样本百分比绝对误差的标准差，表示误差波动和稳定性。",
        "- PMAE median: 百分比误差中位数，用于减弱极端样本影响。",
        "",
        "## 总体对比",
        "",
        "| Mode | Avg MAE | Avg PMAE mean | Avg PMAE std | Avg PMAE median |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in by_target.get("average", []):
        lines.append(
            f"| `{row['mode']}` | {row['mae']:.4f} | {format_percent(row['pmae_mean'])} | "
            f"{format_percent(row['pmae_std'])} | {format_percent(row['pmae_median'])} |"
        )

    lines.extend([
        "",
        "## 分营养项对比",
        "",
        "| Mode | Target | MAE | PMAE mean | PMAE std | PMAE median |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for row in rows:
        if row["target"] == "average":
            continue
        mae_text = f"{row['mae']:.4f} {row['unit']}"
        lines.append(
            f"| `{row['mode']}` | {row['target_name']} | {mae_text} | "
            f"{format_percent(row['pmae_mean'])} | {format_percent(row['pmae_std'])} | "
            f"{format_percent(row['pmae_median'])} |"
        )

    lines.extend([
        "",
        "## 如何解读",
        "",
        "- 如果 `blank_depth`、`random_depth` 或 `shuffled_depth` 的 PMAE 明显高于 `rgb_depth`，说明当前模型确实依赖深度信息。",
        "- 如果某个消融模式的 PMAE std 明显变大，说明模型在部分样本上变得更不稳定。",
        "- 该实验属于推理阶段消融，只能说明模型对输入变化的敏感性；严格证明某个模块是否提升性能，需要重新训练对应模型。",
        "",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-time ablation for OmniFood8K nutrition model")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./trained_weights/omnifood8k/ckpt_best.pth",
        help="trained nutrition checkpoint",
    )
    parser.add_argument("--dataset", choices=["nutrition8K", "nutrition_rgb_pre_d"], default="nutrition8K")
    parser.add_argument("--data_root_8k", type=str, default="./data/0-OminiFood8k")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset")
    parser.add_argument("--data_root_11w", type=str, default="./data/syn-data")
    parser.add_argument("--b", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=1e-6, help="epsilon for percentage errors")
    parser.add_argument("--blur-kernel", type=int, default=31)
    parser.add_argument("--outdir", type=str, default="./experiments/inference_ablation/outputs")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["rgb_depth", "blank_depth", "mean_depth", "random_depth", "shuffled_depth", "blur_depth"],
        choices=["rgb_depth", "blank_depth", "mean_depth", "random_depth", "shuffled_depth", "blur_depth"],
        help="ablation modes to evaluate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.ckpt = resolve_path(args.ckpt)
    args.data_root = resolve_path(args.data_root)
    args.data_root_8k = resolve_path(args.data_root_8k)
    args.data_root_11w = resolve_path(args.data_root_11w)
    args.outdir = resolve_path(args.outdir)

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")

    modules = build_model(args.ckpt, device)
    _, testloader = get_DataLoader(args)

    predictions_by_mode = {mode: [] for mode in args.modes}
    labels_all = []

    with torch.no_grad():
        iterator = tqdm(testloader, desc="Inference ablation", dynamic_ncols=True)
        for batch in iterator:
            rgb_inputs = batch[0].to(device)
            labels = torch.stack([
                batch[2].float(),
                batch[3].float(),
                batch[4].float(),
                batch[5].float(),
                batch[6].float(),
            ], dim=1).to(device)
            depth_inputs = batch[7].to(device)

            labels_all.append(tensor_to_rows(labels))
            for mode in args.modes:
                ablated_depth = ablate_depth(depth_inputs, mode, args.blur_kernel)
                outputs = forward_once(rgb_inputs, ablated_depth, modules)
                predictions_by_mode[mode].append(tensor_to_rows(outputs))

    labels_np = np.concatenate(labels_all, axis=0)
    summary_rows = []
    for mode in args.modes:
        predictions_np = np.concatenate(predictions_by_mode[mode], axis=0)
        summary_rows.extend(summarize_mode(mode, predictions_np, labels_np, args.eps))

    csv_path = os.path.join(args.outdir, "ablation_summary.csv")
    md_path = os.path.join(args.outdir, "ablation_report.md")
    write_summary_csv(summary_rows, csv_path)
    write_markdown(summary_rows, args, md_path)

    print(f"Saved CSV summary: {csv_path}")
    print(f"Saved Markdown report: {md_path}")

    average_rows = [row for row in summary_rows if row["target"] == "average"]
    print("\nOverall PMAE comparison:")
    for row in average_rows:
        print(
            f"{row['mode']:>14s} | PMAE mean {format_percent(row['pmae_mean'])} "
            f"| PMAE std {format_percent(row['pmae_std'])}"
        )


if __name__ == "__main__":
    main()
