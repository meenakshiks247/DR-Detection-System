import os
import io
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import logging
import sys

# --- CONFIGURATION ---
MODEL_FILENAME = "fusion_dr_model.h5"  # Changed to .h5 format
BACKUP_FILENAME = "fusion_dr_model.keras"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable
model = None


# Class mapping
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}
# --- CUSTOM OBJECTS (Must match training) ---
def focal_loss_fn(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        return tf.reduce_mean(y_true) # Dummy for loading
    return focal_loss_fixed

# Import Layer
try:
    from model import ModelFusionLayer, build_fusion_model
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import ModelFusionLayer, build_fusion_model

def _model_path():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, MODEL_FILENAME)


def get_model():
    """Lazily load and return the global model. This avoids blocking server
    startup while heavy TensorFlow/Keras initialization happens.
    """
    global model
    if model is None:
        path_final = _model_path()
        if not os.path.exists(path_final):
            raise RuntimeError(f"Model file not found: {path_final}")
        logger.info(f"🔄 Loading model from: {path_final}")
        try:
            # Include any custom objects required by the saved model
            import keras
            keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(path_final, custom_objects={
                'ModelFusionLayer': ModelFusionLayer,
                'focal_loss_fn': focal_loss_fn
            }, safe_mode=False)
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise
    return model


app = FastAPI(title="DR Detection API")

# --- CORS: ALLOW REACT ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PREPROCESSING ---
def preprocess_image(image_bytes):
    """Preprocess image with error handling for invalid files"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        img = np.array(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize & Normalize
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Online", "system": "Cyber-Physical DR Detection"}

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        m = get_model()
        model_loaded = True
    except:
        model_loaded = False
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "service": "DR Detection API"
    }

@app.get("/model-info")
def model_info():
    """Return model metadata"""
    try:
        m = get_model()
        return {
            "model_name": "Fusion DR Model",
            "architecture": "VGG16 + ResNet50 + DenseNet121 (Parallel Fusion)",
            "input_shape": [224, 224, 3],
            "num_classes": 5,
            "total_parameters": m.count_params(),
            "classes": list(CLASS_NAMES.values())
        }
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        allowed_types = {'image/png', 'image/jpeg', 'image/bmp', 'image/jpg'}
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: PNG, JPG, BMP. Got: {file.content_type}"
            )
        
        # Ensure model is loaded (lazy load on first request)
        try:
            m = get_model()
        except Exception as e:
            logger.error(f"Model load error: {e}")
            raise HTTPException(status_code=503, detail=str(e))

        contents = await file.read()
        
        # Validate file size
        if len(contents) > 10 * 1024 * 1024:  # 10MB max
            raise HTTPException(status_code=413, detail="File too large. Max 10MB")
        
        try:
            processed_img = preprocess_image(contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Inference
        probs = m.predict(processed_img)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        
        return {
            "diagnosis": CLASS_NAMES[class_idx],
            "severity": "High" if class_idx >= 3 else "Low",
            "confidence": confidence,
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)