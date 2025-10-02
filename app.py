import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image, ExifTags
import io
import sys
import json
import re
from openai import OpenAI
from datetime import datetime
import os
import cv2
from fastapi.concurrency import run_in_threadpool
import math
from scipy.spatial import KDTree

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(
    title="PaddleOCR API",
    description="A simple API for OCR using PaddleOCR",
    version="1.0.0"
)

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    lang="ru"
)

def ocr_pdf_paddle(image):
    print("START COMPRESSING...")
    COMPRESS_SCALE = 0.5
    image_array = np.array(image)
    h, w = image_array.shape[:2]
    new_size = (int(w * COMPRESS_SCALE), int(h * COMPRESS_SCALE))
    img_resized = cv2.resize(image_array, new_size, interpolation=cv2.INTER_LINEAR)

    page_text = ocr.predict(img_resized)

    return page_text


def analyze_ocr_text(text: str, model: str = "Qwen/Qwen2-7B-Instruct-AWQ"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an OCR analyzer for Russian transport documents (ТТН). "
                        "You must extract structured information into JSON with the exact keys below. "
                        "Never merge multiple fields into one. "
                        "Always split cargo description into separate fields. "
                        "Always return full name + address + requisites for sender, receiver, and carrier. "
                        "Output ONLY valid JSON. "
                        "Do not add explanations, comments, markdown, or extra text. "
                        "Never include trailing commas in JSON. Always return strictly valid JSON."
                        "If a field is missing, return null. "
                        "All values must be strings or null. Keys must always be present.\n\n"
                        "- sender: full name, address, and requisites of the sender (грузоотправитель)\n"
                        "- date: the shipping date in format dd.mm.yyyy\n"
                        "- request_number: the request/application number (заявка)\n"
                        "- receiver: full name, address, and requisites of the receiver (грузополучатель)\n"
                        "- item_name: the name of the cargo (e.g. 'Бортовой камень')\n"
                        "- size: cargo dimensions (e.g. '1000 x 300 x 150 (с фаской 20 x 20)')\n"
                        "- quantity: cargo quantity (e.g. '198 шт')\n"
                        "- net_weight: net weight (e.g. '10,8 т')\n"
                        "- gross_weight: gross weight (e.g. '10,926 т')\n"
                        "- volume: cargo volume (e.g. '8,91 м³')\n"
                        "- carrier: full name, address, and requisites of the carrier (перевозчик)\n"
                        "- vehicle: vehicle details (model + license plate)\n\n"
                        "Example output:\n"
                        "{\n"
                        "  \"sender\": \"ООО Бекам, 125212, г. Москва, ул. Адмирала Макарова, д. 6, стр. 13, ИНН..., КПП...\",\n"
                        "  \"date\": \"06.08.2024\",\n"
                        "  \"request_number\": \"31795/B\",\n"
                        "  \"receiver\": \"ГБУ города Москвы 'Автомобильные дороги', 123007, г. Москва, ул. 1-я Магистральная, д. 23, ИНН..., КПП...\",\n"
                        "  \"item_name\": \"Бортовой камень\",\n"
                        "  \"size\": \"1000 x 300 x 150 (с фаской 20 x 20)\",\n"
                        "  \"quantity\": \"198 шт\",\n"
                        "  \"net_weight\": \"19,463 т\",\n"
                        "  \"gross_weight\": \"19,694 т\",\n"
                        "  \"volume\": \"8,91 м³\",\n"
                        "  \"carrier\": \"ООО Автопрофит, 125212, г. Москва, ул. Адмирала Макарова, д. 6, стр. 13, ИНН..., КПП...\",\n"
                        "  \"vehicle\": \"Бортовые автомобили, RENAULT, O 942 AE 797\"\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Extract the required fields and return JSON only:\n\n{text}"
                }
            ],
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
        raw = raw.strip()
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        return {k: data.get(k, None) for k in REQUIRED_KEYS}

    except Exception as e:
        return {"error": str(e)}

def euclidean_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def get_text_centers(ocr_result, min_score=0.85):
    boxes = ocr_result.get("rec_boxes", [])
    texts = ocr_result.get("rec_texts", [])
    scores = ocr_result.get("rec_scores", [])

    centers = []
    for box, text, score in zip(boxes, texts, scores):
        x_min, y_min, x_max, y_max = box
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        centers.append({
            "text": text.strip(),
            "center": (cx, cy),
            "box": box,
            "used": False
        })
    return centers

def group_texts(centers, coef=1.0):
    points = [c["center"] for c in centers]
    tree = KDTree(points)

    blocks = []
    for idx, c in enumerate(centers):
        if c["used"]:
            continue
        block = [c["text"]]
        c["used"] = True

        current_idx = idx
        while True:
            # находим ближайшего соседа
            dist, nearest_idx = tree.query(centers[current_idx]["center"], k=2)
            # k=2 потому что первый результат — это он сам
            dist, nearest_idx = dist[1], nearest_idx[1]

            if dist == float("inf"):
                break

            height = centers[current_idx]["box"][3] - centers[current_idx]["box"][1]
            max_dist = height * coef

            if not centers[nearest_idx]["used"] and dist <= max_dist:
                block.append(centers[nearest_idx]["text"])
                centers[nearest_idx]["used"] = True
                current_idx = nearest_idx
            else:
                break
        blocks.append(" ".join(block))
    return blocks

def fix_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if not exif:
            return img
        orientation = None
        for tag, value in exif.items():
            if ExifTags.TAGS.get(tag) == "Orientation":
                orientation = value
                break
        if orientation == 3:
            img = img.rotate(180, expand=True)
        elif orientation == 6:
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def preprocess_ocr_text(rec_texts: list[str]) -> list[str]:
    """
    Подчищает OCR-текст перед передачей в LLM.
    На вход: список строк от OCR.
    На выход: одна строка с очищенным текстом.
    """

    cleaned = []
    for line in rec_texts:
        if not line:
            continue

        # 1. Убираем комментарии в скобках (типа "(реквизиты...)")
        line = re.sub(r"\([^)]{20,}\)", "", line)

        # 2. Исправляем "О00" → "ООО"
        line = re.sub(r"\bО00\b", "ООО", line, flags=re.IGNORECASE)
        line = re.sub(r"(\d+)/B\b", r"\1/Б", line, flags=re.IGNORECASE)
        line = re.sub(r"(\d+)/62\b", r"\1/Б", line, flags=re.IGNORECASE)

        # 3. Нормализуем ИНН и КПП
        line = re.sub(r"\bИнН\b", "ИНН", line, flags=re.IGNORECASE)
        line = re.sub(r"\bкпП\b", "КПП", line, flags=re.IGNORECASE)
        line = re.sub(r"\bкПп\b", "КПП", line, flags=re.IGNORECASE)
        line = re.sub(r"\bкп\b", "КПП", line, flags=re.IGNORECASE)
        line = re.sub(r"KII(\d+)", r"КПП \1", line, flags=re.IGNORECASE)

        # 4. Исправляем склеенные слова: "Дата29.04.2024" → "Дата: 29.04.2024"
        line = re.sub(r"(Дата)(\d{2}\.\d{2}\.\d{4})", r"\1: \2", line)

        # 5. Убираем двойные пробелы
        line = re.sub(r"\s{2,}", " ", line).strip()

        if line:
            cleaned.append(line)

    return cleaned


@app.post("/ocrWithLlm")
async def perform_ocr_with_llm(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = fix_orientation(image)
    result = await run_in_threadpool(ocr_pdf_paddle, image)

    if isinstance(result, list) and len(result) > 0:
        centers = get_text_centers(result[0])
        rec_texts = group_texts(centers, 1.0)
    elif isinstance(result, dict):
        centers = get_text_centers(result)
        rec_texts = group_texts(centers, 1.0)
    else:
        rec_texts = []
    rec_texts = preprocess_ocr_text(rec_texts   )
    text_for_llm = "\n".join(rec_texts)
    llm_result = await run_in_threadpool(analyze_ocr_text, text_for_llm)

    return JSONResponse(content={
        "llmResult": llm_result
    })


client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://195.209.210.133:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "none")
)

REQUIRED_KEYS = [
    "sender","date","request_number","receiver","item_name",
    "size","quantity","net_weight","gross_weight","volume",
    "carrier","vehicle"
]