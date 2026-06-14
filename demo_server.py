import json
import os
from email.parser import BytesParser
from email.policy import default
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


HOST = "127.0.0.1"
PORT = 8000

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "aaaaaaaaaaaaaaaa")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://tokenflux.dev/v1")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "xhigh")


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
                result = predict_nutrition_placeholder(payload)
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


def predict_nutrition_placeholder(payload):
    """Reserved API shape for OmniFood8K inference.

    Final implementation can call scripts/infer_nutrition.py functions:
    build_depth_model -> make_depth_image -> build_nutrition_model -> predict.
    Expected response stays the same so demo_vue.html will not need changes.
    """
    filename = (payload.get("filename") or "").lower()

    if "salad" in filename or "vegetable" in filename:
        nutrition = {"calories": 260, "mass": 280, "fat": 9, "carb": 28, "protein": 13}
    elif "chicken" in filename or "fish" in filename:
        nutrition = {"calories": 430, "mass": 320, "fat": 16, "carb": 26, "protein": 42}
    elif "cake" in filename or "fried" in filename:
        nutrition = {"calories": 760, "mass": 260, "fat": 38, "carb": 82, "protein": 16}
    else:
        nutrition = {"calories": 520, "mass": 310, "fat": 18, "carb": 62, "protein": 24}

    return {
        "source": "mock-server",
        "model": "reserved-omnifood8k",
        "nutrition": nutrition,
        "received": {
            "filename": payload.get("filename"),
            "image_size": payload.get("image_size"),
            "mode": payload.get("mode"),
            "profile": payload.get("profile"),
            "options": payload.get("options"),
        }
    }


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
