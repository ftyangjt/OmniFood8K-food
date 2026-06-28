import json
import os
import threading
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HOST = "127.0.0.1"
PORT = 8000

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_env(filename=".env"):
    env_path = os.path.join(PROJECT_DIR, filename)
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def first_env(*names, default=""):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return default


load_local_env()

AI_API_KEY = first_env("AI_API_KEY", "OPENAI_API_KEY")
AI_BASE_URL = first_env("AI_BASE_URL", "OPENAI_BASE_URL", default="https://tokenflux.dev/v1")
AI_MODEL = first_env("AI_MODEL", "OPENAI_MODEL", default="gpt-5.5")
AI_API_STYLE = first_env("AI_API_STYLE", "OPENAI_API_STYLE", default="chat").lower()
AI_REASONING_EFFORT = first_env("AI_REASONING_EFFORT", "OPENAI_REASONING_EFFORT", default="xhigh")
AI_TIMEOUT_SECONDS = int(first_env("AI_TIMEOUT_SECONDS", default="60"))

DEFAULT_NUTRITION_CKPT = r"trained_weights\omnifood8k\ckpt_best.pth"
DEFAULT_ENCODER = "vitl"
DEFAULT_INPUT_SIZE = 518

MODEL_CACHE = {}
MODEL_CACHE_LOCK = threading.Lock()


def project_path(*parts):
    return os.path.join(PROJECT_DIR, *parts)


def resolve_path(path):
    if path is None:
        return None
    return path if os.path.isabs(path) else project_path(path)


def import_inference_dependencies():
    try:
        import cv2
        import numpy as np
        import torch
        from scripts.infer_nutrition import (
            build_depth_model,
            build_nutrition_model,
            make_depth_image,
            predict,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            "模型推理依赖缺失："
            f"{missing}。请先激活项目环境 `conda activate omnifood`，"
            "或安装 requirements.txt/requirements-cu128.txt 后再启动服务。"
        ) from exc

    return cv2, np, torch, build_depth_model, build_nutrition_model, make_depth_image, predict


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/health":
            self.send_json({
                "ok": True,
                "ai": {
                    "configured": not is_placeholder_key(AI_API_KEY),
                    "base_url": AI_BASE_URL,
                    "model": AI_MODEL,
                    "style": AI_API_STYLE,
                },
                "model_files": {
                    "nutrition": os.path.exists(resolve_path(DEFAULT_NUTRITION_CKPT)),
                    "depth": os.path.exists(project_path("pth", f"depth_anything_v2_{DEFAULT_ENCODER}.pth")),
                },
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
                self.send_json(generate_advice(payload))
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
    cv2, np, _, _, _, make_depth_image, predict = import_inference_dependencies()
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
    _, _, torch, build_depth_model, build_nutrition_model, _, _ = import_inference_dependencies()
    cache_key = (ckpt_path, depth_ckpt_path, encoder)
    with MODEL_CACHE_LOCK:
        if cache_key not in MODEL_CACHE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            depth_model = build_depth_model(encoder, depth_ckpt_path, device)
            nutrition_modules = build_nutrition_model(ckpt_path, device)
            MODEL_CACHE[cache_key] = depth_model, nutrition_modules, device
        return MODEL_CACHE[cache_key]


def generate_advice(payload):
    if is_placeholder_key(AI_API_KEY):
        return {
            "advice": generate_local_advice(payload),
            "source": "local_fallback",
            "warning": "AI API key is not configured. Set AI_API_KEY or OPENAI_API_KEY in .env.",
        }

    return {
        "advice": generate_llm_advice(payload),
        "source": "llm",
        "model": AI_MODEL,
    }


def is_placeholder_key(api_key):
    cleaned = (api_key or "").strip()
    lowered = cleaned.lower()
    return (
        not cleaned
        or cleaned.startswith("aaaa")
        or "replace_with" in lowered
        or "your_api_key" in lowered
        or lowered in {"changeme", "none", "null"}
    )


def generate_local_advice(payload):
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
    if AI_API_STYLE == "responses":
        data = request_responses_api(prompt)
    else:
        data = request_chat_completions_api(prompt)

    return extract_response_text(data)


def request_responses_api(prompt):
    request_body = {
        "model": AI_MODEL,
        "input": prompt,
        "reasoning": {
            "effort": AI_REASONING_EFFORT
        },
        "store": False,
    }

    return post_ai_json(f"{AI_BASE_URL.rstrip('/')}/responses", request_body)


def request_chat_completions_api(prompt):
    request_body = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个谨慎、简洁的中文营养建议助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
    }

    return post_ai_json(f"{AI_BASE_URL.rstrip('/')}/chat/completions", request_body)


def post_ai_json(url, request_body):
    request = Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=AI_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM API network error: {exc.reason}") from exc


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

    choices = data.get("choices")
    if isinstance(choices, list):
        texts = []
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") in ("text", "output_text")
                )
        if texts:
            return "\n".join(texts).strip()

    texts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                texts.append(content.get("text", ""))

    return "\n".join(texts).strip() or "大模型未返回建议文本。"


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DemoHandler)
    print(f"Demo server running at http://{HOST}:{PORT}/demo_vue.html")
    print(f"Nutrition API: http://{HOST}:{PORT}/api/nutrition/predict")
    print(f"AI configured: {not is_placeholder_key(AI_API_KEY)}")
    print(f"AI endpoint: {AI_BASE_URL.rstrip('/')} ({AI_API_STYLE})")
    print(f"AI model: {AI_MODEL}")
    server.serve_forever()
