import argparse
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from infer_nutrition import (  # noqa: E402
    build_depth_model,
    build_nutrition_model,
    make_depth_image,
    predict,
    project_path,
    resolve_path,
)


DEFAULT_CKPT = r"trained_weights\omnifood8k\ckpt_best.pth"
DEFAULT_ENCODER = "vitl"
DEFAULT_INPUT_SIZE = 518
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class NutritionGUI:
    def __init__(self, root, args):
        self.root = root
        self.root.title("OmniFood8K 营养估计")
        self.root.geometry("900x620")
        self.root.minsize(780, 540)

        self.args = args
        self.image_path = None
        self.preview_image = None
        self.depth_preview_image = None
        self.models = None
        self.model_key = None

        self.ckpt_var = tk.StringVar(value=args.ckpt)
        self.depth_ckpt_var = tk.StringVar(value=args.depth_ckpt)
        self.encoder_var = tk.StringVar(value=args.encoder)
        self.input_size_var = tk.IntVar(value=args.input_size)
        self.status_var = tk.StringVar(value="请选择一张食物图片。")

        self.result_vars = {
            "calories": tk.StringVar(value="-"),
            "mass": tk.StringVar(value="-"),
            "fat": tk.StringVar(value="-"),
            "carb": tk.StringVar(value="-"),
            "protein": tk.StringVar(value="-"),
        }

        self._build_layout()

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.BOTH, expand=True)
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)
        top.rowconfigure(0, weight=1)

        image_panel = ttk.LabelFrame(top, text="图片")
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        image_panel.rowconfigure(0, weight=1)
        image_panel.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(image_panel, text="未选择图片", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        image_buttons = ttk.Frame(image_panel)
        image_buttons.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(image_buttons, text="选择图片", command=self.choose_image).pack(side=tk.LEFT)
        self.run_button = ttk.Button(image_buttons, text="开始估计", command=self.start_prediction)
        self.run_button.pack(side=tk.LEFT, padx=8)

        result_panel = ttk.LabelFrame(top, text="营养素含量")
        result_panel.grid(row=0, column=1, sticky="nsew")
        result_panel.columnconfigure(1, weight=1)

        rows = [
            ("热量", "calories", "kJ"),
            ("质量", "mass", "g"),
            ("脂肪", "fat", "g"),
            ("碳水化合物", "carb", "g"),
            ("蛋白质", "protein", "g"),
        ]
        for index, (label, key, unit) in enumerate(rows):
            ttk.Label(result_panel, text=f"{label}:").grid(row=index, column=0, sticky="w", padx=12, pady=8)
            ttk.Label(result_panel, textvariable=self.result_vars[key], font=("", 12, "bold")).grid(
                row=index,
                column=1,
                sticky="e",
                padx=8,
                pady=8,
            )
            ttk.Label(result_panel, text=unit).grid(row=index, column=2, sticky="w", padx=(0, 12), pady=8)

        depth_frame = ttk.LabelFrame(result_panel, text="生成深度图")
        depth_frame.grid(row=len(rows), column=0, columnspan=3, sticky="nsew", padx=12, pady=(12, 8))
        result_panel.rowconfigure(len(rows), weight=1)
        depth_frame.rowconfigure(0, weight=1)
        depth_frame.columnconfigure(0, weight=1)
        self.depth_label = ttk.Label(depth_frame, text="预测后显示", anchor=tk.CENTER)
        self.depth_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        settings = ttk.LabelFrame(outer, text="模型设置")
        settings.pack(fill=tk.X, pady=(12, 8))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="营养估计权重").grid(row=0, column=0, sticky="w", padx=8, pady=5)
        ttk.Entry(settings, textvariable=self.ckpt_var).grid(row=0, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(settings, text="浏览", command=self.choose_ckpt).grid(row=0, column=2, padx=8, pady=5)

        ttk.Label(settings, text="深度估计权重").grid(row=1, column=0, sticky="w", padx=8, pady=5)
        ttk.Entry(settings, textvariable=self.depth_ckpt_var).grid(row=1, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(settings, text="浏览", command=self.choose_depth_ckpt).grid(row=1, column=2, padx=8, pady=5)

        small = ttk.Frame(settings)
        small.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=5)
        ttk.Label(small, text="Encoder").pack(side=tk.LEFT)
        ttk.Combobox(
            small,
            textvariable=self.encoder_var,
            values=["vits", "vitb", "vitl", "vitg"],
            width=8,
            state="readonly",
        ).pack(side=tk.LEFT, padx=(8, 18))
        ttk.Label(small, text="输入尺寸").pack(side=tk.LEFT)
        ttk.Spinbox(small, from_=256, to=1024, increment=14, textvariable=self.input_size_var, width=8).pack(
            side=tk.LEFT,
            padx=8,
        )

        ttk.Label(outer, textvariable=self.status_var).pack(fill=tk.X)

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择食物图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        if os.path.splitext(path.lower())[1] not in IMAGE_EXTS:
            messagebox.showerror("文件格式不支持", "请选择 jpg、jpeg、png、bmp 或 webp 图片。")
            return
        self.image_path = path
        self.status_var.set(f"已选择: {path}")
        self._show_image(path, self.image_label, "preview_image", max_size=(520, 360))

    def choose_ckpt(self):
        path = filedialog.askopenfilename(title="选择营养估计权重", filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def choose_depth_ckpt(self):
        path = filedialog.askopenfilename(title="选择深度估计权重", filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")])
        if path:
            self.depth_ckpt_var.set(path)

    def start_prediction(self):
        if not self.image_path:
            messagebox.showinfo("请选择图片", "请先选择一张食物图片。")
            return
        self.run_button.configure(state=tk.DISABLED)
        self.status_var.set("正在加载模型并估计营养素，第一次运行会稍慢。")
        thread = threading.Thread(target=self._predict_worker, daemon=True)
        thread.start()

    def _predict_worker(self):
        try:
            result = self._predict()
        except Exception as exc:
            self.root.after(0, self._prediction_failed, str(exc))
            return
        self.root.after(0, self._prediction_done, result)

    def _predict(self):
        encoder = self.encoder_var.get()
        ckpt_path = resolve_path(self.ckpt_var.get())
        depth_ckpt_path = resolve_path(self.depth_ckpt_var.get())
        input_size = int(self.input_size_var.get())

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"营养估计权重不存在: {ckpt_path}")
        if not os.path.exists(depth_ckpt_path):
            raise FileNotFoundError(f"深度估计权重不存在: {depth_ckpt_path}")

        model_key = (ckpt_path, depth_ckpt_path, encoder)
        if self.models is None or self.model_key != model_key:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_model = build_depth_model(encoder, depth_ckpt_path, device)
            nutrition_modules = build_nutrition_model(ckpt_path, device)
            self.models = depth_model, nutrition_modules, device
            self.model_key = model_key

        depth_model, nutrition_modules, device = self.models
        raw_image = cv2.imread(self.image_path)
        if raw_image is None:
            raise ValueError(f"图片无法读取: {self.image_path}")

        depth_image = make_depth_image(depth_model, raw_image, input_size, grayscale=True)
        values = predict(raw_image, depth_image, nutrition_modules, device)
        values = [max(0.0, float(value)) for value in values]
        depth_rgb = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        return values, depth_rgb

    def _prediction_done(self, result):
        values, depth_rgb = result
        keys = ["calories", "mass", "fat", "carb", "protein"]
        for key, value in zip(keys, values):
            self.result_vars[key].set(f"{value:.2f}")

        self._show_array(depth_rgb, self.depth_label, "depth_preview_image", max_size=(300, 210))
        self.status_var.set("估计完成。")
        self.run_button.configure(state=tk.NORMAL)

    def _prediction_failed(self, message):
        self.status_var.set("估计失败。")
        self.run_button.configure(state=tk.NORMAL)
        messagebox.showerror("估计失败", message)

    def _show_image(self, path, label, attr_name, max_size):
        image = Image.open(path).convert("RGB")
        self._set_photo(image, label, attr_name, max_size)

    def _show_array(self, image_rgb, label, attr_name, max_size):
        image = Image.fromarray(image_rgb).convert("RGB")
        self._set_photo(image, label, attr_name, max_size)

    def _set_photo(self, image, label, attr_name, max_size):
        image.thumbnail(max_size)
        photo = ImageTk.PhotoImage(image)
        setattr(self, attr_name, photo)
        label.configure(image=photo, text="")


def main():
    parser = argparse.ArgumentParser(description="Launch a simple desktop GUI for nutrition estimation.")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="trained nutrition checkpoint")
    parser.add_argument("--depth-ckpt", type=str, default=None, help="Depth Anything V2 checkpoint")
    parser.add_argument("--encoder", type=str, default=DEFAULT_ENCODER, choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    args = parser.parse_args()

    if args.depth_ckpt is None:
        args.depth_ckpt = project_path("pth", f"depth_anything_v2_{args.encoder}.pth")

    root = tk.Tk()
    NutritionGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
