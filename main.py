from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(
    title="AgriSmart API",
    description="Detection de maladies des cultures - MobileNetV2",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_PATH  = "model/transfert_learning_first_model.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Saine", "Maladie Fongique", "Maladie Virale", "Dommage Insecte"]

print("Chargement du modele...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modele pret !")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.array(img, dtype=np.float32), axis=0)

@app.get("/")
def root():
    return {
        "message"  : "AgriSmart API operationnelle",
        "version"  : "1.0.0",
        "modele"   : "MobileNetV2 Transfer Learning",
        "precision": "93%",
        "docs"     : "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Utilisez JPG ou PNG.")
    try:
        img_array = preprocess_image(await file.read())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    predictions = model.predict(img_array, verbose=0)[0]
    idx         = int(np.argmax(predictions))
    confidence  = float(np.max(predictions))

    conseils = {
        0: "Culture saine. Continuez le suivi regulier.",
        1: "Maladie fongique detectee. Appliquez un fongicide adapte.",
        2: "Maladie virale detectee. Isolez les plants infectes.",
        3: "Dommage insectes detecte. Utilisez un insecticide cible.",
    }

    return {
        "prediction"   : CLASS_NAMES[idx],
        "confidence"   : f"{confidence * 100:.1f}%",
        "class_index"  : idx,
        "probabilities": {
            CLASS_NAMES[i]: f"{float(predictions[i])*100:.2f}%"
            for i in range(len(CLASS_NAMES))
        },
        "conseil": conseils[idx]
    }
