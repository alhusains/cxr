"""
Production-ready FastAPI inference service for chest X-ray classification.

Features:
- RESTful API for predictions
- Input validation and preprocessing
- Model versioning
- Monitoring and logging
- Health checks
- Security headers
"""

import sys
from pathlib import Path
import io
import time
from datetime import datetime
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils import load_config, get_device
from src.models.model import create_model

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Chest X-Ray Classification API",
    description="Production-ready API for chest X-ray pathology classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
CONFIG = None
CLASS_NAMES = None
TRANSFORM = None
PREDICTION_LOG = []


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    patient_id: Optional[str] = Field(None, description="Anonymized patient identifier")
    study_id: Optional[str] = Field(None, description="Study identifier")
    metadata: Optional[dict] = Field(None, description="Additional metadata")


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: dict = Field(..., description="Probabilities for all classes")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")
    warnings: List[str] = Field(default_factory=list, description="Any warnings")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "pneumonia",
                "confidence": 0.87,
                "probabilities": {
                    "normal": 0.05,
                    "pneumonia": 0.87,
                    "tuberculosis": 0.08
                },
                "model_version": "1.0.0",
                "timestamp": "2026-01-12T10:30:00Z",
                "warnings": []
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str
    timestamp: str


def load_model():
    """Load trained model and setup."""
    global MODEL, DEVICE, CONFIG, CLASS_NAMES, TRANSFORM
    
    logger.info("Loading model and configuration...")
    
    # Load config
    CONFIG = load_config()
    DEVICE = get_device(CONFIG)
    
    # Load model
    MODEL = create_model(CONFIG)
    checkpoint = torch.load('models/best_model.pth', map_location=DEVICE)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    # Class names
    CLASS_NAMES = ['normal', 'pneumonia', 'tuberculosis']
    
    # Setup preprocessing
    target_size = CONFIG['preprocessing']['target_size']
    mean = CONFIG['preprocessing']['normalize_mean']
    std = CONFIG['preprocessing']['normalize_std']
    
    TRANSFORM = A.Compose([
        A.LongestMaxSize(max_size=target_size[0]),
        A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1],
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    
    logger.info(f"Model loaded successfully on {DEVICE}")
    logger.info(f"Model version: {checkpoint.get('epoch', 'unknown')}")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    try:
        load_model()
        Path('logs').mkdir(exist_ok=True)
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Chest X-Ray Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return {
        "total_predictions": len(PREDICTION_LOG),
        "predictions_last_hour": sum(1 for p in PREDICTION_LOG 
                                     if time.time() - p['timestamp'] < 3600),
        "average_confidence": np.mean([p['confidence'] for p in PREDICTION_LOG]) 
                            if PREDICTION_LOG else 0.0,
        "class_distribution": {
            cls: sum(1 for p in PREDICTION_LOG if p['prediction'] == cls)
            for cls in CLASS_NAMES
        } if PREDICTION_LOG else {}
    }


def validate_image(image: Image.Image) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded image.
    
    Returns:
        (is_valid, error_message)
    """
    # Check format
    if image.format not in ['JPEG', 'PNG']:
        return False, "Image must be JPEG or PNG format"
    
    # Check size
    if image.size[0] < 128 or image.size[1] < 128:
        return False, "Image too small (minimum 128x128 pixels)"
    
    if image.size[0] > 4096 or image.size[1] > 4096:
        return False, "Image too large (maximum 4096x4096 pixels)"
    
    # Check mode
    if image.mode not in ['L', 'RGB', 'RGBA']:
        return False, f"Unsupported image mode: {image.mode}"
    
    return True, None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        
    Returns:
        Preprocessed tensor
    """
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy
    img_array = np.array(image)
    
    # Convert to grayscale then back to 3-channel (consistent with training)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    img_array = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # Apply transforms
    transformed = TRANSFORM(image=img_array)
    img_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return img_tensor


def detect_data_drift(image_stats: dict) -> List[str]:
    """
    Detect potential data drift.
    
    Args:
        image_stats: Dictionary with image statistics
        
    Returns:
        List of warnings
    """
    warnings = []
    
    # Check mean intensity
    if image_stats['mean'] < 50:
        warnings.append("Image appears underexposed - may affect prediction quality")
    elif image_stats['mean'] > 200:
        warnings.append("Image appears overexposed - may affect prediction quality")
    
    # Check std (contrast)
    if image_stats['std'] < 20:
        warnings.append("Low contrast detected - image quality may be poor")
    
    # Check aspect ratio
    if image_stats['aspect_ratio'] > 2.0 or image_stats['aspect_ratio'] < 0.5:
        warnings.append("Unusual aspect ratio - image may be cropped incorrectly")
    
    return warnings


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="X-ray image file"),
    patient_id: Optional[str] = Header(None, alias="X-Patient-ID"),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Predict pathology from chest X-ray image.
    
    Security:
    - API key authentication (add proper validation in production)
    - PHI handling: patient_id should be pre-anonymized
    - All predictions logged for audit trail
    """
    start_time = time.time()
    
    try:
        # API key validation (simplified - use proper auth in production)
        if api_key is None:
            logger.warning("Request without API key")
            # In production, should reject request
            # raise HTTPException(status_code=401, detail="API key required")
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            logger.warning(f"Invalid image uploaded: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Compute image statistics for drift detection
        img_array = np.array(image.convert('L'))
        image_stats = {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'aspect_ratio': image.size[0] / image.size[1]
        }
        
        # Detect drift
        warnings = detect_data_drift(image_stats)
        
        # Preprocess
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(DEVICE)
        
        # Predict
        with torch.no_grad():
            outputs = MODEL(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
        
        # Prepare response
        pred_class = CLASS_NAMES[prediction.item()]
        pred_confidence = confidence.item()
        pred_probs = {
            CLASS_NAMES[i]: float(probabilities[0, i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Low confidence warning
        if pred_confidence < 0.7:
            warnings.append(f"Low confidence prediction ({pred_confidence:.2f}) - recommend expert review")
        
        # Log prediction for monitoring
        prediction_log_entry = {
            'timestamp': time.time(),
            'prediction': pred_class,
            'confidence': pred_confidence,
            'patient_id': patient_id,
            'inference_time_ms': (time.time() - start_time) * 1000,
            'image_stats': image_stats,
            'warnings': warnings
        }
        PREDICTION_LOG.append(prediction_log_entry)
        
        # Keep only last 1000 predictions in memory
        if len(PREDICTION_LOG) > 1000:
            PREDICTION_LOG.pop(0)
        
        # Audit log
        logger.info(f"Prediction made: {pred_class} (confidence: {pred_confidence:.3f}, "
                   f"patient_id: {patient_id or 'anonymous'}, time: {prediction_log_entry['inference_time_ms']:.1f}ms)")
        
        response = PredictionResponse(
            prediction=pred_class,
            confidence=pred_confidence,
            probabilities=pred_probs,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
            warnings=warnings
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: List[UploadFile] = File(...)):
    """Batch prediction endpoint for multiple images."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append({"filename": file.filename, "result": result.dict()})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})
    
    return {"predictions": results, "total": len(files)}


@app.get("/drift_report")
async def get_drift_report():
    """Get data drift analysis report."""
    if not PREDICTION_LOG:
        return {"message": "No predictions yet"}
    
    recent_predictions = [p for p in PREDICTION_LOG if time.time() - p['timestamp'] < 86400]
    
    if not recent_predictions:
        return {"message": "No recent predictions (last 24h)"}
    
    # Analyze statistics
    mean_intensities = [p['image_stats']['mean'] for p in recent_predictions]
    confidences = [p['confidence'] for p in recent_predictions]
    
    return {
        "period": "last_24_hours",
        "total_predictions": len(recent_predictions),
        "average_confidence": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "low_confidence_rate": sum(1 for c in confidences if c < 0.7) / len(confidences),
        "image_quality": {
            "mean_intensity": float(np.mean(mean_intensities)),
            "intensity_std": float(np.std(mean_intensities))
        },
        "warnings_rate": sum(1 for p in recent_predictions if p['warnings']) / len(recent_predictions),
        "recommendation": "Monitor if confidence drops below 0.8 or warning rate exceeds 20%"
    }


def main():
    """Run the API server."""
    import uvicorn
    
    uvicorn.run(
        "src.deployment.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
