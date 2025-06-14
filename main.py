from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = load_model("CNN_modell.h5")

# Load label encoder
with open("label_encoderr.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Prediction endpoint
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
