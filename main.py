from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pickle
import os
import gdown

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Root Route to Confirm API is Live ===
@app.get("/")
def read_root():
    return {"message": "Leaf Disease Detection API is running!"}

# === Auto-download model if not exists ===
model_path = "CNN_modell.h5"
drive_url = "https://drive.google.com/uc?id=1tI0hG3s9wWbiP_XJDeFwbRf0Z4pGJ2S7"

if not os.path.exists(model_path):
    print("Downloading CNN_modell.h5 from Google Drive...")
    gdown.download(drive_url, model_path, quiet=False)

# === Load model and label encoder ===
model = load_model(model_path)

with open("label_encoderr.pkl", "rb") as f:
    le = pickle.load(f)

# === Preprocessing ===
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# === Prediction endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = preprocess_image(image_bytes)
    prediction = model.predict(img)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))
    return {
        "label": predicted_label,
        "confidence": confidence
    }
