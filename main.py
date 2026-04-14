import os
import base64
import re
from io import BytesIO
from fastapi import FastAPI, Request
from ultralytics import YOLO
import requests
from PIL import Image

app = FastAPI()

# Load model once at startup
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

        # Handle Base64 Data
        if image_data.startswith("data:image"):
            base64_data = re.sub(r'^data:image/.+;base64,', '', image_data)
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        
        # Handle Image URL
        else:
            # ADDED: Standard Browser Headers to prevent 403 Forbidden errors
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
            }
            
            response = requests.get(image_data, headers=headers, timeout=15)
            
            # Check if the download actually worked
            if response.status_code != 200:
                return {"error": f"Failed to download image. Status code: {response.status_code}"}
            
            img = Image.open(BytesIO(response.content)).convert('RGB')

        # Run Inference
        results = model(img)

        # Process Results
        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            predicted_class = results[0].names[int(top_box.cls)]
            confidence = float(top_box.conf)
            return {"disease": predicted_class, "confidence": round(confidence, 4)}

        return {"disease": "Healthy", "confidence": 0.0}

    except Exception as e:
        # This will now catch the "cannot identify image file" error if the file is corrupted
        return {"error": f"Prediction error: {str(e)}"}