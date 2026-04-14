import os
import base64
import re
import zipfile
from io import BytesIO
from fastapi import FastAPI, Request
from ultralytics import YOLO
import requests
from PIL import Image

app = FastAPI()

MODEL_PATH = "best.pt"
ZIP_PATH = "model.zip"
FILE_ID = "1jjtYyJlXLmqG1ewDGRjqf8xw5DJsLLLu"  # <-- paste your file ID here
MODEL_URL = f"https://drive.google.com/file/d/1jjtYyJlXLmqG1ewDGRjqf8xw5DJsLLLu/view?usp=drive_link"

def download_and_extract_model():
    if os.path.isfile(MODEL_PATH):
        print("Model already exists, skipping download.")
        return

    print("Downloading model zip from Google Drive...")
    session = requests.Session()
    response = session.get(MODEL_URL, stream=True, timeout=120)

    # Google Drive sends a confirm token for large files
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(MODEL_URL, params={"confirm": token}, stream=True, timeout=120)

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    print("Download complete.")

    # Check if it's a zip and extract
    if zipfile.is_zipfile(ZIP_PATH):
        print("Extracting zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(ZIP_PATH)
        print("Extraction complete.")
    else:
        # It's already a .pt file, just rename it
        os.rename(ZIP_PATH, MODEL_PATH)
        print("File was not a zip, renamed directly to best.pt")

    # Search for .pt file if not at expected path
    if not os.path.isfile(MODEL_PATH):
        for fname in os.listdir("."):
            if fname.endswith(".pt") and os.path.isfile(fname):
                os.rename(fname, MODEL_PATH)
                print(f"Renamed {fname} to {MODEL_PATH}")
                break

download_and_extract_model()

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("best.pt not found after download and extraction!")

model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

@app.get("/")
async def root():
    return {"status": "online", "message": "Poultry Droppings AI is running"}

@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        image_data = data.get("image_url")

        if not image_data:
            return {"error": "No image_url provided"}

        if image_data.startswith("data:image"):
            base64_data = re.sub(r'^data:image/.+;base64,', '', image_data)
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        else:
            response = requests.get(image_data, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')

        results = model(img)

        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            predicted_class = results[0].names[int(top_box.cls)]
            confidence = float(top_box.conf)
            return {"disease": predicted_class, "confidence": round(confidence, 4)}

        return {"disease": "Healthy", "confidence": 0.0}

    except Exception as e:
        return {"error": str(e)}