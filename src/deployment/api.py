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
import os
from datetime import datetime
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy import stats
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

# Mount static files for web demo
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
MODEL = None
DEVICE = None
CONFIG = None
CLASS_NAMES = None
TRANSFORM = None
PREDICTION_LOG = []
EXPECTED_API_KEY = None
TRAINING_STATS = None


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


class UncertaintyPredictionResponse(PredictionResponse):
    """Response model for prediction with uncertainty estimation."""
    uncertainty: dict = Field(..., description="Epistemic uncertainty per class")
    prediction_variance: float = Field(..., description="Overall prediction variance")
    mc_iterations: int = Field(..., description="Number of Monte Carlo iterations")
    
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
    global MODEL, DEVICE, CONFIG, CLASS_NAMES, TRANSFORM, EXPECTED_API_KEY, TRAINING_STATS
    
    logger.info("Loading model and configuration...")
    
    # Load API key from environment
    EXPECTED_API_KEY = os.getenv("CXR_API_KEY", "dev-key-please-change-in-production")
    if EXPECTED_API_KEY == "dev-key-please-change-in-production":
        logger.warning("Using default API key - CHANGE THIS IN PRODUCTION!")
    
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
    
    # Load training distribution statistics for drift detection
    training_stats_path = Path('models/training_distribution_stats.npy')
    if training_stats_path.exists():
        TRAINING_STATS = np.load(training_stats_path, allow_pickle=True).item()
        logger.info("Loaded training distribution statistics for drift detection")
    else:
        logger.warning("Training distribution stats not found - drift detection will use fallback method")
        TRAINING_STATS = None
    
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


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves web demo if available."""
    demo_path = Path("static/index.html")
    if demo_path.exists():
        return demo_path.read_text()
    else:
        return """
        <html>
            <body>
                <h1>Chest X-Ray Classification API</h1>
                <p>Version: 1.0.0 | Status: Running</p>
                <ul>
                    <li><a href="/health">Health Check</a></li>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/metrics">Metrics</a></li>
                    <li><a href="/drift_report">Drift Report</a></li>
                </ul>
            </body>
        </html>
        """


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


def detect_data_quality_issues(image_stats: dict) -> List[str]:
    """
    Detect data quality issues (not drift, but image quality problems).
    
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


def detect_data_drift(recent_window_size: int = 50) -> dict:
    """
    Detect statistical drift by comparing recent predictions against training distribution.
    
    Uses Kolmogorov-Smirnov test to compare distributions and z-score for mean shifts.
    
    Args:
        recent_window_size: Number of recent predictions to analyze
        
    Returns:
        Dictionary with drift detection results
    """
    if TRAINING_STATS is None:
        return {
            'drift_detected': False,
            'method': 'fallback',
            'message': 'Training statistics not available - using quality checks only'
        }
    
    # Get recent predictions
    recent_predictions = PREDICTION_LOG[-recent_window_size:] if len(PREDICTION_LOG) >= recent_window_size else PREDICTION_LOG
    
    if len(recent_predictions) < 30:
        return {
            'drift_detected': False,
            'method': 'insufficient_data',
            'message': f'Need at least 30 samples for drift detection (have {len(recent_predictions)})',
            'sample_count': len(recent_predictions)
        }
    
    # Extract statistics from recent predictions
    recent_means = [p['image_stats']['mean'] for p in recent_predictions]
    recent_stds = [p['image_stats']['std'] for p in recent_predictions]
    recent_confidences = [p['confidence'] for p in recent_predictions]
    
    # Statistical tests
    drift_warnings = []
    drift_detected = False
    
    # 1. Z-score test for mean intensity shift
    current_mean = np.mean(recent_means)
    z_score_mean = abs(current_mean - TRAINING_STATS['mean_intensity']) / TRAINING_STATS['std_intensity']
    
    if z_score_mean > 3.0:  # 3-sigma threshold
        drift_warnings.append(f"Significant mean intensity shift detected (z-score: {z_score_mean:.2f})")
        drift_detected = True
    elif z_score_mean > 2.0:
        drift_warnings.append(f"Moderate mean intensity shift detected (z-score: {z_score_mean:.2f})")
    
    # 2. Kolmogorov-Smirnov test for distribution shift
    # Compare recent means against expected normal distribution with training parameters
    if 'intensity_samples' in TRAINING_STATS and len(TRAINING_STATS['intensity_samples']) > 0:
        ks_statistic, ks_pvalue = stats.ks_2samp(recent_means, TRAINING_STATS['intensity_samples'])
        
        if ks_pvalue < 0.01:  # Strong evidence of different distributions
            drift_warnings.append(f"Distribution shift detected (KS p-value: {ks_pvalue:.4f})")
            drift_detected = True
        elif ks_pvalue < 0.05:
            drift_warnings.append(f"Possible distribution shift (KS p-value: {ks_pvalue:.4f})")
    
    # 3. Confidence drift (performance degradation indicator)
    if 'mean_confidence' in TRAINING_STATS:
        current_confidence = np.mean(recent_confidences)
        confidence_drop = TRAINING_STATS['mean_confidence'] - current_confidence
        
        if confidence_drop > 0.15:  # 15% drop in confidence
            drift_warnings.append(f"Significant confidence drop: {confidence_drop:.2%}")
            drift_detected = True
        elif confidence_drop > 0.10:
            drift_warnings.append(f"Moderate confidence drop: {confidence_drop:.2%}")
    
    # 4. Standard deviation shift
    current_std_of_means = np.std(recent_means)
    if 'std_of_means' in TRAINING_STATS:
        std_ratio = current_std_of_means / TRAINING_STATS['std_of_means']
        if std_ratio > 2.0 or std_ratio < 0.5:
            drift_warnings.append(f"Variability change detected (ratio: {std_ratio:.2f})")
    
    return {
        'drift_detected': drift_detected,
        'method': 'statistical',
        'warnings': drift_warnings,
        'metrics': {
            'z_score_mean': float(z_score_mean),
            'current_mean_intensity': float(current_mean),
            'reference_mean_intensity': float(TRAINING_STATS['mean_intensity']),
            'current_mean_confidence': float(np.mean(recent_confidences)),
            'sample_count': len(recent_predictions)
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(..., description="X-ray image file"),
    patient_id: Optional[str] = Header(None, alias="X-Patient-ID"),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Predict pathology from chest X-ray image.
    
    Security:
    - API key authentication (required)
    - PHI handling: patient_id should be pre-anonymized
    - All predictions logged for audit trail
    """
    start_time = time.time()
    
    try:
        # API key validation (production-ready)
        if api_key is None:
            logger.warning(f"Request without API key from {request.client.host}")
            raise HTTPException(
                status_code=401, 
                detail="API key required. Include 'X-API-Key' header with valid key."
            )
        
        if api_key != EXPECTED_API_KEY:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Please check your credentials."
            )
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            logger.warning(f"Invalid image uploaded: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Compute image statistics for quality checks
        img_array = np.array(image.convert('L'))
        image_stats = {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'aspect_ratio': image.size[0] / image.size[1]
        }
        
        # Detect image quality issues
        warnings = detect_data_quality_issues(image_stats)
        
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


@app.post("/predict_with_uncertainty", response_model=UncertaintyPredictionResponse)
async def predict_with_uncertainty_endpoint(
    request: Request,
    file: UploadFile = File(..., description="X-ray image file"),
    patient_id: Optional[str] = Header(None, alias="X-Patient-ID"),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    mc_iterations: int = 10
):
    """
    Predict pathology with Monte Carlo Dropout uncertainty estimation.
    
    This endpoint provides epistemic uncertainty estimates using Bayesian approximation
    via Monte Carlo Dropout. Higher uncertainty suggests the model is less certain
    and human review is recommended.
    
    Args:
        mc_iterations: Number of forward passes with dropout (default: 10)
    """
    start_time = time.time()
    
    try:
        # API key validation
        if api_key is None:
            logger.warning(f"Request without API key from {request.client.host}")
            raise HTTPException(
                status_code=401, 
                detail="API key required. Include 'X-API-Key' header with valid key."
            )
        
        if api_key != EXPECTED_API_KEY:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Please check your credentials."
            )
        
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        is_valid, error_msg = validate_image(image)
        if not is_valid:
            logger.warning(f"Invalid image uploaded: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Compute image statistics for quality checks
        img_array = np.array(image.convert('L'))
        image_stats = {
            'mean': float(np.mean(img_array)),
            'std': float(np.std(img_array)),
            'aspect_ratio': image.size[0] / image.size[1]
        }
        
        # Detect image quality issues
        warnings = detect_data_quality_issues(image_stats)
        
        # Preprocess
        img_tensor = preprocess_image(image)
        img_tensor = img_tensor.to(DEVICE)
        
        # Monte Carlo Dropout prediction
        mean_probs, uncertainty, all_preds = predict_with_uncertainty(
            MODEL, img_tensor, DEVICE, n_iterations=mc_iterations
        )
        
        # Extract results
        mean_probs = mean_probs[0]  # Remove batch dimension
        uncertainty = uncertainty[0]
        
        pred_idx = np.argmax(mean_probs)
        pred_class = CLASS_NAMES[pred_idx]
        pred_confidence = float(mean_probs[pred_idx])
        pred_uncertainty = float(uncertainty[pred_idx])
        
        pred_probs = {
            CLASS_NAMES[i]: float(mean_probs[i])
            for i in range(len(CLASS_NAMES))
        }
        
        uncertainty_dict = {
            CLASS_NAMES[i]: float(uncertainty[i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Calculate overall variance
        prediction_variance = float(np.mean(uncertainty))
        
        # Uncertainty-based warnings
        if pred_uncertainty > 0.15:
            warnings.append(f"High epistemic uncertainty ({pred_uncertainty:.3f}) - model is uncertain about this prediction")
        elif pred_uncertainty > 0.10:
            warnings.append(f"Moderate uncertainty ({pred_uncertainty:.3f}) - consider expert review")
        
        # Low confidence warning
        if pred_confidence < 0.7:
            warnings.append(f"Low confidence prediction ({pred_confidence:.2f}) - recommend expert review")
        
        # Log prediction
        prediction_log_entry = {
            'timestamp': time.time(),
            'prediction': pred_class,
            'confidence': pred_confidence,
            'uncertainty': pred_uncertainty,
            'patient_id': patient_id,
            'inference_time_ms': (time.time() - start_time) * 1000,
            'image_stats': image_stats,
            'warnings': warnings,
            'mc_iterations': mc_iterations
        }
        PREDICTION_LOG.append(prediction_log_entry)
        
        # Keep only last 1000 predictions
        if len(PREDICTION_LOG) > 1000:
            PREDICTION_LOG.pop(0)
        
        # Audit log
        logger.info(f"MC Prediction made: {pred_class} (confidence: {pred_confidence:.3f}, "
                   f"uncertainty: {pred_uncertainty:.3f}, patient_id: {patient_id or 'anonymous'}, "
                   f"time: {prediction_log_entry['inference_time_ms']:.1f}ms)")
        
        response = UncertaintyPredictionResponse(
            prediction=pred_class,
            confidence=pred_confidence,
            probabilities=pred_probs,
            uncertainty=uncertainty_dict,
            prediction_variance=prediction_variance,
            mc_iterations=mc_iterations,
            model_version="1.0.0",
            timestamp=datetime.utcnow().isoformat(),
            warnings=warnings
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MC Prediction error: {str(e)}", exc_info=True)
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
    """
    Get comprehensive data drift analysis report.
    
    Compares recent predictions against training distribution using statistical tests.
    """
    if not PREDICTION_LOG:
        return {"message": "No predictions yet"}
    
    recent_predictions = [p for p in PREDICTION_LOG if time.time() - p['timestamp'] < 86400]
    
    if not recent_predictions:
        return {"message": "No recent predictions (last 24h)"}
    
    # Analyze statistics
    mean_intensities = [p['image_stats']['mean'] for p in recent_predictions]
    confidences = [p['confidence'] for p in recent_predictions]
    
    # Statistical drift detection
    drift_analysis = detect_data_drift(recent_window_size=min(100, len(recent_predictions)))
    
    # Basic metrics
    basic_metrics = {
        "period": "last_24_hours",
        "total_predictions": len(recent_predictions),
        "average_confidence": float(np.mean(confidences)),
        "confidence_std": float(np.std(confidences)),
        "low_confidence_rate": sum(1 for c in confidences if c < 0.7) / len(confidences),
        "image_quality": {
            "mean_intensity": float(np.mean(mean_intensities)),
            "intensity_std": float(np.std(mean_intensities))
        },
        "warnings_rate": sum(1 for p in recent_predictions if p['warnings']) / len(recent_predictions)
    }
    
    # Combine with drift analysis
    report = {
        **basic_metrics,
        "drift_detection": drift_analysis,
        "recommendation": _generate_recommendation(basic_metrics, drift_analysis)
    }
    
    return report


def predict_with_uncertainty(model, img_tensor, device, n_iterations=10):
    """
    Perform Monte Carlo Dropout for uncertainty estimation.
    
    This enables dropout during inference and runs multiple forward passes
    to estimate epistemic (model) uncertainty.
    
    Args:
        model: PyTorch model
        img_tensor: Input tensor
        device: Device to run on
        n_iterations: Number of MC iterations
        
    Returns:
        mean_probs: Mean probabilities across iterations
        uncertainty: Standard deviation (epistemic uncertainty)
        all_predictions: All probability predictions
    """
    model.train()  # Enable dropout
    
    predictions = []
    with torch.no_grad():
        for _ in range(n_iterations):
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predictions.append(probabilities.cpu().numpy())
    
    predictions = np.array(predictions)  # Shape: (n_iterations, batch_size, n_classes)
    
    # Compute statistics
    mean_probs = predictions.mean(axis=0)  # Mean across iterations
    uncertainty = predictions.std(axis=0)   # Epistemic uncertainty
    
    model.eval()  # Restore eval mode
    
    return mean_probs, uncertainty, predictions


def _generate_recommendation(basic_metrics: dict, drift_analysis: dict) -> str:
    """Generate actionable recommendation based on metrics and drift detection."""
    recommendations = []
    
    if drift_analysis.get('drift_detected'):
        recommendations.append("⚠️ ALERT: Significant data drift detected - immediate review recommended")
    
    if basic_metrics['average_confidence'] < 0.7:
        recommendations.append("Low average confidence - consider model retraining")
    
    if basic_metrics['low_confidence_rate'] > 0.3:
        recommendations.append("High rate of low-confidence predictions - increase human review")
    
    if basic_metrics['warnings_rate'] > 0.2:
        recommendations.append("Elevated warning rate - check data quality at source")
    
    if not recommendations:
        return "✓ System operating normally - continue monitoring"
    
    return " | ".join(recommendations)


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
