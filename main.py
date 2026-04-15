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

        # Scenario 1: Raw File Upload (This is what Flutter uses!)
        if file is not None:
            contents = await file.read()
            img = Image.open(io.BytesIO(contents)).convert('RGB')

        # Scenario 2: JSON (Base64 or URL)
        elif request is not None:
            # Only try to parse JSON if we didn't get a file
            try:
                data = await request.json()
                image_data = data.get("image_url")

                if image_data:
                    if image_data.startswith("data:image"):
                        base64_data = re.sub(r'^data:image/.+;base64,', '', image_data)
                        img_bytes = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                    else:
                        headers = {"User-Agent": "PoulPal-App"}
                        response = requests.get(image_data, headers=headers, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(io.BytesIO(response.content)).convert('RGB')
            except Exception:
                pass # If JSON parsing fails, just move on

        if img is None:
            return {"error": "Invalid image input. Please send a file or a valid JSON payload."}

        # Resize BEFORE giving to YOLO to prevent RAM exhaustion
        img = optimize_image(img, max_dim=640)

        # Run Inference
        results = model.predict(source=img, imgsz=320, conf=0.25)

        if len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            predicted_class = results[0].names[int(top_box.cls)].lower()
            confidence = float(top_box.conf)
            
            return {
                "disease": predicted_class, 
                "confidence": round(confidence, 4)
            }

        return {"disease": "healthy", "confidence": 0.0}

    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)