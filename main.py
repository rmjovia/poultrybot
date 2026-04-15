import os
import base64
import re
import io
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import requests
from PIL import Image

app = FastAPI()

# Enable CORS for Flutter Web or Cross-Origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with half-precision (saves RAM on Render Free Tier)
model = YOLO("best.pt")

# ==========================================
# THE FIX: Helper function to prevent RAM exhaustion
# ==========================================
def optimize_image(img: Image.Image, max_dim: int = 640) -> Image.Image:
    """
    Resizes the image so its longest side is max_dim, preserving the aspect ratio.
    This prevents massive 12-megapixel phone photos from crashing the Render server.
    """
    # thumbnail modifies the image in-place and keeps aspect ratio intact
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

@app.get("/")
async def root():
    return {"status": "online", "system": "PoulPal AI"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(None), 
    request: Request = None
):
    try:
        img = None

        # Scenario 1: Raw File Upload (Matches the Flutter code provided)
        if file:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Scenario 2: JSON (Base64 or URL)
        else:
            data = await request.json()
            image_data = data.get("image_url")

            if not image_data:
                return {"error": "No image or URL provided"}

            if image_data.startswith("data:image"):
                base64_data = re.sub(r'^data:image/.+;base64,', '', image_data)
                img_bytes = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            else:
                headers = {"User-Agent": "PoulPal-App"}
                response = requests.get(image_data, headers=headers, timeout=10)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content)).convert('RGB')

        if img is None:
            return {"error": "Invalid image input"}

        # ==========================================
        # APPLY THE FIX: Resize BEFORE giving to YOLO
        # ==========================================
        img = optimize_image(img, max_dim=640)

        # Run Inference
        # imgsz=320 reduces RAM usage on Render
        results = model.predict(source=img, imgsz=320, conf=0.25)

        if len(results[0].boxes) > 0:
            # Get the detection with the highest confidence
            top_box = results[0].boxes[0]
            # Map the model's class name to PoulPal IDs
            predicted_class = results[0].names[int(top_box.cls)].lower()
            confidence = float(top_box.conf)
            
            # Ensure the class name matches your local KB (cocci, ncd, salmonella, healthy)
            return {
                "disease": predicted_class, 
                "confidence": round(confidence, 4)
            }

        return {"disease": "healthy", "confidence": 0.0}

    except Exception as e:
        # If anything fails, return the exact error so Flutter can display it
        return {"error": f"Internal Server Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)