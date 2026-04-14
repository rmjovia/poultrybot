import os
import zipfile
from fastapi import FastAPI, Request
from ultralytics import YOLO
import requests
import base64
import re
from io import BytesIO
from PIL import Image

app = FastAPI()

# 1. Handle the "Model-is-a-Zip" issue
model_zip = 'best (2).pt' # The file you uploaded
extract_path = 'model_files'
model_path = None

if os.path.exists(model_zip):
    # If it's a zip/directory structure, extract it
    if zipfile.is_zipfile(model_zip):
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        # Point YOLO to the directory or the .pkl within it
        model_path = extract_path
    else:
        model_path = model_zip

# 2. Load the model using Ultralytics (much more stable than torch.load for YOLO)
if model_path:
    model = YOLO(model_path)
else:
    print("Warning: Model file not found!")

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

        # Handle Base64 (from Dify) or URL
        if image_data.startswith("data:image"):
            base64_data = re.sub(r'^data:image/.+;base64,', '', image_data)
            img_bytes = base64.b64decode(base64_data)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
        else:
            response = requests.get(image_data, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Run Inference
        results = model(img)
        
        # Extract Results
        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            predicted_class = results[0].names[int(top_box.cls)]
            confidence = float(top_box.conf)
            return {"disease": predicted_class, "confidence": confidence}
        
        return {"disease": "Healthy", "confidence": 0.0}

    except Exception as e:
        return {"error": str(e)}