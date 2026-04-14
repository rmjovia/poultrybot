import os
import base64
import re
from io import BytesIO
from fastapi import FastAPI, Request
from ultralytics import YOLO
import requests
from PIL import Image

app = FastAPI()

model = YOLO("best.pt")
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