from fastapi import FastAPI, UploadFile, File
import torch # or tensorflow
from PIL import Image

app = FastAPI()
model = torch.load('best (2).pt', map_content='cpu') # Load your Colab model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert('RGB')
    # ... process image and run model ...
    prediction = "Predicted Disease Name"
    return {"disease": prediction}
