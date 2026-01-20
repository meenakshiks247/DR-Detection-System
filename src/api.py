import os
import io
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
import warnings
warnings.filterwarnings('ignore')

# Defer heavy TensorFlow import until model load to allow the API to start fast
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import logging
import sys

# --- CONFIGURATION ---
MODEL_FILENAME = "fusion_dr_model_final.keras"
BACKUP_FILENAME = "fusion_dr_model.h5"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable
model = None
generalist_model = None


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
        import tensorflow as tf
        return tf.reduce_mean(y_true)  # Dummy for loading
    return focal_loss_fixed

# Note: defer importing `model` until TensorFlow is available inside get_model()

def _model_path():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, MODEL_FILENAME)


def _generalist_model_path():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, 'generalist_model.h5')


def get_model():
    """Lazily load and return the global model. This avoids blocking server
    startup while heavy TensorFlow/Keras initialization happens.
    """
    global model
    if model is None:
        import tensorflow as tf
        import keras
        # Enable unsafe deserialization for Lambda layers
        keras.config.enable_unsafe_deserialization()
        # Import custom layer after TF is available
        try:
            from model import ModelFusionLayer
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from model import ModelFusionLayer
        path_final = _model_path()
        if not os.path.exists(path_final):
            raise RuntimeError(f"Model file not found: {path_final}")
        logger.info(f"🔄 Loading model from: {path_final}")
        try:
            # Include any custom objects required by the saved model
            model = tf.keras.models.load_model(
                path_final,
                custom_objects={
                    'ModelFusionLayer': ModelFusionLayer,
                    'focal_loss_fn': focal_loss_fn
                }
            )
            logger.info("✅ Model loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise
    return model


def get_generalist_model():
    """Lazily load and return the lightweight generalist model for Cataract/Glaucoma."""
    global generalist_model
    if generalist_model is None:
        import tensorflow as tf
        import keras
        # Enable unsafe deserialization for Lambda layers
        keras.config.enable_unsafe_deserialization()
        path = _generalist_model_path()
        if not os.path.exists(path):
            logger.warning(f"Generalist model not found at: {path}")
            return None
        logger.info(f"🔄 Loading generalist model from: {path}")
        try:
            # generalist_model.h5 is a small custom sequential model (no custom objects)
            generalist_model = tf.keras.models.load_model(path)
            logger.info("✅ Generalist model loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load generalist model: {e}")
            generalist_model = None
    return generalist_model


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


def preprocess_image_generalist(image_bytes):
    """Preprocess image for the generalist model: 128x128 RGB, normalized."""
    image = Image.open(io.BytesIO(image_bytes))
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.resize(img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Online", "system": "Cyber-Physical DR Detection"}

@app.get("/health")
def health():
    """Health check endpoint - doesn't load model to prevent blocking startup"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "DR Detection API"
    }

@app.get("/model-info")
def model_info():
    """Get model information - doesn't load model"""
    return {
        "model_name": "Fusion DR Model (Final)",
        "architecture": "EfficientNetB0 + MobileNetV3 Fusion",
        "input_shape": [1, 224, 224, 3],
        "num_classes": 5,
        "total_parameters": 53000000,
        "classes": list(CLASS_NAMES.values()),
        "status": "Model will be loaded on first prediction request"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Ensure model is loaded (lazy load on first request)
        try:
            m = get_model()
        except Exception as e:
            logger.error(f"Model load error: {e}")
            raise HTTPException(status_code=503, detail=str(e))

        contents = await file.read()

        # Preprocess for fusion DR model (224x224)
        processed_img = preprocess_image(contents)

        # Preprocess for generalist model (128x128)
        processed_img_gen = preprocess_image_generalist(contents)

        # Inference - Fusion DR model
        probs = m.predict(processed_img)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        dr_result = {
            "diagnosis": CLASS_NAMES[class_idx],
            "severity": "High" if class_idx >= 3 else "Low",
            "confidence": confidence,
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(5)}
        }

        # Inference - Generalist model (if available)
        gen = get_generalist_model()
        GENERALIST_CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
        other_condition = None
        generalist_result = None
        if gen is not None:
            try:
                probs_g = gen.predict(processed_img_gen)[0]
                idx_g = int(np.argmax(probs_g))
                conf_g = float(probs_g[idx_g])
                label_g = GENERALIST_CLASS_NAMES[idx_g]
                generalist_result = {
                    "label": label_g,
                    "confidence": conf_g,
                    "probabilities": {GENERALIST_CLASS_NAMES[i]: float(probs_g[i]) for i in range(4)}
                }
                # Report only Cataract or Glaucoma as "other_condition" when confident
                if label_g in ("Cataract", "Glaucoma") and conf_g >= 0.5:
                    other_condition = {"label": label_g, "confidence": conf_g}
            except Exception as e:
                logger.error(f"Generalist prediction failed: {e}")

        # Rule-based AI advice combining both results
        if other_condition:
            if dr_result["diagnosis"] != "No DR":
                advice = f"Signs of {dr_result['diagnosis']} detected (confidence {dr_result['confidence']:.2f}) and possible {other_condition['label']} (confidence {other_condition['confidence']:.2f}). Recommend urgent ophthalmology referral for comprehensive evaluation."
            else:
                advice = f"No Diabetic Retinopathy detected, but possible {other_condition['label']} (confidence {other_condition['confidence']:.2f}). Recommend specialist evaluation for that condition."
        else:
            if dr_result["diagnosis"] != "No DR":
                advice = f"Diabetic Retinopathy detected ({dr_result['diagnosis']}, confidence {dr_result['confidence']:.2f}). Follow DR care pathway. No strong evidence of Cataract or Glaucoma from generalist model."
            else:
                advice = "No Diabetic Retinopathy detected and no strong signs of Cataract or Glaucoma. Routine follow-up recommended."

        response = {
            "dr_result": dr_result,
            "other_condition": other_condition,
            "generalist_result": generalist_result,
            "ai_advice": advice
        }

        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)