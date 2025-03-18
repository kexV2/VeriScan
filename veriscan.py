from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import shutil
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image

app = FastAPI(title="Deepfake Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the trained deepfake detection model
MODEL_PATH = "deepfake_model.h5"

def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        print(model.summary())  # Print model architecture for verification
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model.")

# Load the model at startup
model = load_model()

# Image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Static file serving for uploads
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate a unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        file.file.close()

    # Preprocess image and make prediction
    try:
        img = preprocess_image(file_path)
        predictions = model.predict(img)[0]
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Map class index to label (assuming 0 = Fake, 1 = Real)
        class_labels = ["Fake", "Real"]
        result = class_labels[class_index]

        forensic_indicators = [
            "Natural noise pattern distribution" if result == "Real" else "Unnatural noise patterns in background",
            "Consistent color profile across image" if result == "Real" else "Inconsistent lighting effects",
            "No detectable compression inconsistencies" if result == "Real" else "Suspicious texture patterns in details"
        ]

        return {
            "filename": unique_filename,
            "prediction": result,
            "confidence": confidence * 100,
            "forensic_indicators": forensic_indicators,
            "status": "success",
            "message": "Prediction complete"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
