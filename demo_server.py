import json
import os
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

from scripts.infer_nutrition import (
    build_depth_model,
    build_nutrition_model,
    make_depth_image,
    predict,
    project_path,
    resolve_path,
)


HOST = "127.0.0.1"
PORT = 8000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "aaaaaaaaaaaaaaaa")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://tokenflux.dev/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")

DEFAULT_NUTRITION_CKPT = r"trained_weights\omnifood8k\ckpt_best.pth"
DEFAULT_ENCODER = "vitl"
DEFAULT_INPUT_SIZE = 518

MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({
                "ok": True,
                "endpoints": {
                    "advice": "/api/advice",
                    "nutrition_predict": "/api/nutrition/predict",
                    "depth_preview": "/api/depth/preview"
                }
            })
            return

        super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/advice":
                payload = self.read_json_body()
                advice = generate_advice(payload)
                self.send_json({"advice": advice})
                return

            if self.path == "/api/nutrition/predict":
                payload = self.read_multipart_body()
                result = predict_nutrition(payload)
                self.send_json(result)
                return

            if self.path == "/api/depth/preview":
                self.send_json({
                    "error": "Depth preview endpoint is reserved for the final model version."
                }, status=501)
                return

            self.send_error(404, "Not found")
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=500)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def read_multipart_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        message = BytesParser(policy=default).parsebytes(
            b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
        )

        fields = {}
        files = {}
        if message.is_multipart():
            for part in message.iter_parts():
                disposition = part.get_content_disposition()
                if disposition != "form-data":
                    continue

                name = part.get_param("name", header="content-disposition")
                filename = part.get_filename()
                data = part.get_payload(decode=True) or b""
                if filename:
                    files[name] = {"filename": filename, "data": data}
                elif name:
                    fields[name] = data.decode(part.get_content_charset() or "utf-8", errors="replace")

        image = files.get("image")
        options = parse_json_text(fields.get("options"), {})
        profile = parse_json_text(fields.get("profile"), {})
        mode = fields.get("mode", "normal")

        return {
            "filename": image.get("filename", "") if image else "",
            "image_data": image.get("data", b"") if image else b"",
            "image_size": len(image.get("data", b"")) if image else 0,
            "mode": mode,
            "profile": profile,
            "options": options,
        }

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_json_text(raw, default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def predict_nutrition(payload):
    image_data = payload.get("image_data") or b""
    if not image_data:
        raise ValueError("No image file was uploaded.")

    options = payload.get("options") or {}
    encoder = options.get("encoder") or DEFAULT_ENCODER
    input_size = int(options.get("inputSize") or DEFAULT_INPUT_SIZE)
    ckpt_path = resolve_path(options.get("ckpt") or DEFAULT_NUTRITION_CKPT)
    depth_ckpt = options.get("depthCkpt") or project_path("pth", f"depth_anything_v2_{encoder}.pth")
    depth_ckpt_path = resolve_path(depth_ckpt)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"营养估计权重不存在: {ckpt_path}")
    if not os.path.exists(depth_ckpt_path):
        raise FileNotFoundError(f"深度估计权重不存在: {depth_ckpt_path}")

    image_array = np.frombuffer(image_data, dtype=np.uint8)
    raw_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if raw_image is None:
        raise ValueError(f"图片无法读取: {payload.get('filename') or 'uploaded image'}")

    depth_model, nutrition_modules, device = load_models(ckpt_path, depth_ckpt_path, encoder)
    depth_image = make_depth_image(depth_model, raw_image, input_size, grayscale=True)
    values = predict(raw_image, depth_image, nutrition_modules, device)
    values = [round(max(0.0, float(value)), 4) for value in values]

    return {
        "source": "omnifood8k",
        "model": "OmniFood8K + Depth Anything V2",
        "device": device,
        "nutrition": {
            "calories": values[0],
            "mass": values[1],
            "fat": values[2],
            "carb": values[3],
            "protein": values[4],
        },
        "received": {
            "filename": payload.get("filename"),
            "image_size": payload.get("image_size"),
            "mode": payload.get("mode"),
            "profile": payload.get("profile"),
        }
    }


def load_models(ckpt_path, depth_ckpt_path, encoder):
    cache_key = (ckpt_path, depth_ckpt_path, encoder)
    with MODEL_CACHE_LOCK:
        if cache_key not in MODEL_CACHE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_model = build_depth_model(encoder, depth_ckpt_path, device)
            nutrition_modules = build_nutrition_model(ckpt_path, device)
            MODEL_CACHE[cache_key] = depth_model, nutrition_modules, device
        return MODEL_CACHE[cache_key]


def generate_advice(payload):
    if is_placeholder_key(OPENAI_API_KEY):
        return generate_mock_advice(payload)
    return generate_llm_advice(payload)


def is_placeholder_key(api_key):
    cleaned = (api_key or "").strip()
    return not cleaned or cleaned.startswith("aaaa")


def generate_mock_advice(payload):
    mode = payload.get("mode", "normal")
    nutrition = payload.get("nutrition", {})
    profile = payload.get("profile", {})

    calories = nutrition.get("calories", "--")
    mass = nutrition.get("mass", "--")
    fat = nutrition.get("fat", "--")
    carb = nutrition.get("carb", "--")
    protein = nutrition.get("protein", "--")

    if mode == "weight_loss":
        height = profile.get("height_cm")
        weight = profile.get("weight_kg")
        age = profile.get("age")
        profile_text = (
            f"当前参数：身高 {height} cm，体重 {weight} kg，年龄 {age}。"
            if height and weight and age
            else "身高、体重、年龄不完整，因此先给出通用建议。"
        )
        return (
            "减脂模式建议\n"
            f"{profile_text}\n"
            f"本次估计：热量 {calories} kcal，重量 {mass} g，脂肪 {fat} g，"
            f"碳水 {carb} g，蛋白质 {protein} g。\n"
            "优先保证蛋白质和蔬菜，减少额外油脂、含糖饮料和精制碳水。"
        )

    return (
        "均衡模式建议\n"
        f"本次估计：热量 {calories} kcal，重量 {mass} g，脂肪 {fat} g，"
        f"碳水 {carb} g，蛋白质 {protein} g。\n"
        "可以检查本餐是否包含蔬菜、膳食纤维、钙来源和足够饮水。"
    )


def generate_llm_advice(payload):
    nutrition = payload.get("nutrition", {})
    mode = payload.get("mode", "normal")
    profile = payload.get("profile", {})

    prompt = build_prompt(mode, nutrition, profile)
    request_body = {
        "model": OPENAI_MODEL,
        "input": prompt,
        "reasoning": {
            "effort": OPENAI_REASONING_EFFORT
        },
        "store": False,
    }

    request = Request(
        f"{OPENAI_BASE_URL.rstrip('/')}/responses",
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error {exc.code}: {error_body}") from exc

    return extract_response_text(data)


def build_prompt(mode, nutrition, profile):
    mode_text = "减脂模式" if mode == "weight_loss" else "均衡模式"
    requirements = (
        "你是营养建议助手。请基于食物营养估计结果给出中文建议。"
        "建议要具体、简洁、适合普通用户理解。"
        "不要做医疗诊断，不要承诺治疗效果。"
    )

    if mode == "weight_loss":
        requirements += (
            "当前是减脂模式，请结合身高、体重、年龄，"
            "从热量控制、蛋白质、碳水和脂肪控制角度给出建议。"
        )
    else:
        requirements += (
            "当前是均衡模式，请从营养均衡角度说明还可以补充哪些营养物质，"
            "例如膳食纤维、蔬菜、水、钙来源等。"
        )

    return (
        f"{requirements}\n"
        f"模式：{mode_text}\n"
        f"营养估计 JSON：{json.dumps(nutrition, ensure_ascii=False)}\n"
        f"用户参数 JSON：{json.dumps(profile, ensure_ascii=False)}\n"
        "输出格式：用 3 到 5 条短建议，每条单独一行。"
    )


def extract_response_text(data):
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                texts.append(content.get("text", ""))

    return "\n".join(texts).strip() or "大模型未返回建议文本。"


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Demo server running at http://{HOST}:{PORT}/demo_vue.html")
    print(f"Reserved nutrition API: http://{HOST}:{PORT}/api/nutrition/predict")
    print(f"LLM endpoint: {OPENAI_BASE_URL.rstrip('/')}/responses")
    print(f"LLM model: {OPENAI_MODEL}")
    server.serve_forever()
