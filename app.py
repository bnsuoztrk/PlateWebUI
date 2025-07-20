from flask import Flask, request, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import os
import uuid
import numpy as np
import pytesseract

# Windows için Tesseract yolu (gerekirse değiştir)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Statik klasörler
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

model_path = "best.pt"

def load_model():
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"YOLO modeli yüklendi: {model_path}")
            return model
        else:
            print(f"Model dosyası bulunamadı: {model_path}")
            return None
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None

model = load_model()

def extract_plate_text(image, coords):
    try:
        x1, y1, x2, y2 = coords
        plate_region = image.crop((x1, y1, x2, y2))

        if plate_region.size[0] == 0 or plate_region.size[1] == 0:
            return "Plaka okunamadı"

        plate_region = plate_region.resize((plate_region.size[0]*3, plate_region.size[1]*3), Image.Resampling.LANCZOS)
        plate_gray = plate_region.convert("L")

        plate_np = np.array(plate_gray)
        plate_np = np.where(plate_np > 150, 255, 0).astype(np.uint8)

        text = pytesseract.image_to_string(plate_np, config="--psm 7")
        text = text.strip().replace('\n', '').replace(' ', '').replace('\r', '')

        return text if text else "Plaka okunamadı"
    except Exception as e:
        return f"OCR hatası: {str(e)}"

def draw_bounding_boxes(image, results, threshold=0.5):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            conf = box.conf.item()
            if conf < threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            text = f"PLATE {conf:.2f}"
            draw.text((x1, y1 - 20), text, fill="red", font=font)

    return draw_image

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        global model
        if model is None:
            model = load_model()
            if model is None:
                return render_template("index.html", error="Model henüz hazır değil.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="Lütfen bir dosya seçin!")

        unique_id = str(uuid.uuid4())
        original_filename = f"{unique_id}_original.jpg"
        result_filename = f"{unique_id}_result.jpg"

        original_image = Image.open(io.BytesIO(file.read())).convert("RGB")
        original_image.save(f"static/uploads/{original_filename}")

        img_np = np.array(original_image)
        results = model(img_np)

        detections = 0
        plate_texts = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = box.conf.item()
                if conf > 0.5:
                    detections += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    text = extract_plate_text(original_image, (x1, y1, x2, y2))
                    plate_texts.append(text)

        result_image = draw_bounding_boxes(original_image, results)
        result_image.save(f"static/results/{result_filename}")

        return render_template(
            "result.html",
            original_image=f"uploads/{original_filename}",
            result_image=f"results/{result_filename}",
            detections=detections,
            plate_texts=plate_texts,
        )

    except Exception as e:
        return render_template("index.html", error=f"Hata oluştu: {str(e)}")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
