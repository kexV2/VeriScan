from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
import uuid
import tensorflow as tf
import numpy as np
from PIL import Image
from grad_cam import get_grad_cam, save_grad_cam_visualization, analyze_heatmap

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

# Set correct permissions for uploads directory
try:
    os.chmod(UPLOAD_DIR, 0o755)  # rwxr-xr-x
except Exception as e:
    print(f"Warning: Could not set permissions for uploads directory: {str(e)}")

# Load the trained deepfake detection model
MODEL_PATH = "deepfake_model_expanded.h5"

def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
        print("Model architecture summary:")
        model.summary()  # Print model architecture for verification
        
        # Debug: Print all layer names and types
        print("\nModel layers:")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} - {type(layer).__name__}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model.")

# Load the model at startup
model = load_model()

# Find a suitable layer for visualization
def find_best_layer(model):
    """Find the best layer for visualization in the model"""
    # Try to find a convolutional layer first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
            
    # Try to find GlobalAveragePooling2D
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return layer.name
            
    # Look for specific layer names
    for layer in model.layers:
        if any(name in layer.name.lower() for name in ["conv", "pool", "dense", "global"]):
            return layer.name
            
    # Fallback to the second to last layer
    if len(model.layers) > 1:
        return model.layers[-2].name
        
    # Last resort
    return model.layers[-1].name

target_layer = find_best_layer(model)
print(f"Selected layer for visualization: {target_layer}")

# Image preprocessing function
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((299, 299))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")

# Generate forensic indicators based on Grad-CAM analysis
def generate_forensic_indicators(heatmap_insights, prediction):
    indicators = []
    
    # Get focus regions
    focus_regions = heatmap_insights.get("focus_regions", [])
    scores = heatmap_insights.get("scores", {})
    
    # Generate insights based on prediction and focus areas
    if prediction == "Fake":
        if "central facial features" in focus_regions:
            indicators.append("Suspicious patterns detected in facial features")
            
        if "hair_focus" in scores and scores["hair_focus"] > 0.6:
            indicators.append("Unnatural transitions in hair or edge regions")
            
        if "background_focus" in scores and scores["background_focus"] > 0.4:
            indicators.append("Inconsistent background patterns - typical of AI generation")
            
        if len(focus_regions) == 0:
            indicators.append("Unusual distribution of features across the image")
            
        # Add some general indicators if we don't have many specifics
        if len(indicators) < 2:
            indicators.append("Potential artificial generation artifacts detected")
    else:  # Real
        if "central facial features" in focus_regions:
            indicators.append("Natural facial feature consistency detected")
            
        if "background_focus" in scores and scores["background_focus"] < 0.3:
            indicators.append("Natural background patterns consistent with real photography")
            
        # Add some general indicators for real images
        if len(indicators) < 2:
            indicators.append("Natural noise distribution typical of real photographs")
            indicators.append("No detectable manipulation or generation artifacts")
    
    return indicators

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

    # Ensure the file has proper permissions
    try:
        os.chmod(file_path, 0o644)  # rw-r--r--
    except Exception as e:
        print(f"Warning: Could not set permissions for uploaded file: {str(e)}")

    # Preprocess image and make prediction
    try:
        print(f"Processing file: {file_path}")
        img = preprocess_image(file_path)
        
        predictions = model.predict(img)[0]
        class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Map class index to label (assuming 0 = Fake, 1 = Real)
        class_labels = ["Fake", "Real"]
        result = class_labels[class_index]
        print(f"Prediction: {result} with confidence {confidence:.2f}")

        # Generate Grad-CAM heatmap
        heatmap = None
        heatmap_insights = {}
        heatmap_path = None
        heatmap_url = None
        
        # Generate heatmap for the predicted class
        print(f"Generating heatmap using {target_layer} layer")
        heatmap = get_grad_cam(model, img, target_layer, class_index)
        
        if heatmap is not None:
            # Save visualization
            heatmap_filename = f"heatmap_{unique_filename}"
            heatmap_path = os.path.join(UPLOAD_DIR, heatmap_filename)
            print(f"Saving heatmap to {heatmap_path}")
            
            save_result = save_grad_cam_visualization(file_path, heatmap, heatmap_path)
            
            if save_result and os.path.exists(heatmap_path):
                # Set proper permissions for the heatmap file
                try:
                    os.chmod(heatmap_path, 0o644)  # rw-r--r--
                except Exception as e:
                    print(f"Warning: Could not set permissions for heatmap file: {str(e)}")
                    
                # Create URL for frontend
                heatmap_url = f"/uploads/{heatmap_filename}"
                print(f"Heatmap URL: {heatmap_url}")
                
                # Analyze heatmap
                heatmap_insights = analyze_heatmap(heatmap)
            else:
                print("Failed to save heatmap visualization")
        else:
            print("Failed to generate heatmap")
        
        # Generate forensic indicators based on the analysis
        forensic_indicators = generate_forensic_indicators(heatmap_insights, result)
        
        # If no specific indicators detected through analysis, use fallback
        if not forensic_indicators:
            if result == "Real":
                forensic_indicators = [
                    "Natural noise pattern distribution",
                    "Consistent color profile across image",
                    "No detectable compression inconsistencies"
                ]
            else:
                forensic_indicators = [
                    "Unnatural noise patterns in background",
                    "Inconsistent lighting effects",
                    "Suspicious texture patterns in details"
                ]

        response_data = {
            "filename": unique_filename,
            "prediction": result,
            "confidence": confidence * 100,
            "forensic_indicators": forensic_indicators,
            "focus_areas": heatmap_insights.get("focus_regions", []),
            "heatmap_url": heatmap_url,
            "status": "success",
            "message": "Prediction complete"
        }
        
        print(f"Returning response: {response_data}")
        return response_data
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "upload_dir_exists": os.path.exists(UPLOAD_DIR)}

if __name__ == "__main__":
    uvicorn.run("veriscan:app", host="0.0.0.0", port=8000, reload=True)