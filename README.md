# Chest X-Ray Classification System

![Demo](reports/figures/UI-demo.gif)

## Overview

This production-ready deep learning system classifies chest X-rays into three categories: **Normal**, **Pneumonia**, and **Tuberculosis**, achieving strong performance with robust deployment infrastructure for clinical integration.

### Key Highlights

- **Performance**: 77.8% accuracy, 0.94 macro AUC-ROC on test set
- **Architecture**: DenseNet-121 with ImageNet pretraining
- **Training**: Bayesian hyperparameter optimization with Focal Loss for class imbalance
- **Explainability**: Grad-CAM visualization validates anatomically correct attention
- **Robustness**: 1-2% performance degradation under clinical scanner variations
- **Deployment**: FastAPI with monitoring, drift detection, and HIPAA-compliant design

## Quick Start

### Prerequisites

- Python 3.8+ (tested with 3.10)
- CUDA-capable GPU (optional but recommended)
- 10GB disk space for dataset

### Installation

```bash
# Clone repository and navigate
cd cxr

# Create virtual environment
make create_environment
source venv/bin/activate

# Install dependencies
make requirements

# Download and prepare data (requires Kaggle API credentials)
make data
```

### Training and Evaluation

```bash
# Train model with hyperparameter tuning (Bayesian optimization)
make train-tune

# Evaluate on test set
make evaluate

# Robustness analysis
make evaluate-robustness

# Generate Grad-CAM explainability visualizations
make explainability
```

### Deployment

```bash
# Compute training distribution statistics (required for drift detection)
python src/deployment/compute_training_stats.py

# Start API server (development)
# Set API key via environment variable (default: dev-key-please-change-in-production)
export CXR_API_KEY="your-secure-api-key-here"
make serve

# Access web demo (interactive UI)
# Visit: http://localhost:8001/

# Access API documentation
# Visit: http://localhost:8001/docs

# Run API tests
python tests/test_api.py

# Example usage
python examples/api_usage.py
```

### Docker Deployment

```bash
# Build Docker image
make docker-build

# Run containerized service
make docker-run

# Stop services
make docker-stop
```

## Documentation

This repository includes comprehensive documentation:

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**: Complete technical report covering EDA findings, modeling choices, performance metrics, robustness analysis, explainability insights, and deployment strategy
- **[reports/figures/](reports/figures/)**: All visualizations (confusion matrix, ROC curves, Grad-CAM examples, etc.)
- **[reports/metrics/](reports/metrics/)**: Quantitative evaluation results in JSON/CSV format

## Project Structure

```
.
├── config/                  # Hydra configuration files
├── data/
│   ├── raw/                 # Original dataset (not in repo)
│   ├── processed/           # Preprocessed data and splits
│   └── external/            # External data sources
├── models/                  # Trained model checkpoints
├── reports/
│   ├── figures/             # Visualizations (confusion matrix, ROC, Grad-CAM)
│   └── metrics/             # Performance metrics (JSON, CSV)
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering utilities
│   ├── models/              # Model architecture and training
│   ├── evaluation/          # Evaluation and robustness testing
│   ├── explainability/      # Grad-CAM and interpretability
│   └── deployment/          # FastAPI inference server
├── tests/                   # Unit and integration tests
├── examples/                # API usage examples
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-service orchestration
├── Makefile                 # Reproducible command interface
├── pyproject.toml           # Python package configuration
├── requirements.txt         # Python dependencies
└── TECHNICAL_REPORT.md      # Main technical report
```

## Performance Summary

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 77.8% |
| Macro F1 | 0.79 |
| Macro AUC-ROC | 0.94 |
| Macro Precision | 0.86 |
| Macro Recall | 0.78 |

### Per-Class Performance

| Class | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------| 
| Normal | 0.62 | 0.97 | 0.76 | 0.88 |
| Pneumonia | 0.96 | 0.73 | 0.83 | 0.99 |
| Tuberculosis | 0.98 | 0.64 | 0.77 | 0.94 |

### Robustness Evaluation

Performance under various corruption conditions:
- **Clean (test set)**: 77.8% accuracy, 0.94 macro AUC-ROC
- **Clinical corruptions**: 76.4% accuracy (-1.4%), 0.89 macro AUC-ROC
- **Severe corruptions**: 77.3% accuracy (-0.5%), 0.92 macro AUC-ROC

Clinical corruptions simulate realistic scanner variations (brightness/contrast shifts, blur, noise, resolution changes). The model demonstrates excellent robustness with <2% accuracy degradation under clinical conditions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | **Interactive web demo** |
| `/health` | GET | Service health check |
| `/predict` | POST | Single image prediction |
| `/predict_with_uncertainty` | POST | Prediction with MC Dropout uncertainty |
| `/batch_predict` | POST | Batch predictions (max 10) |
| `/metrics` | GET | API performance metrics |
| `/drift_report` | GET | Data drift analysis |
| `/docs` | GET | Interactive API documentation |

### Example Usage

```python
import requests

# Single prediction with API key authentication
headers = {"X-API-Key": "your-secure-api-key-here"}

with open("chest_xray.jpg", "rb") as f:
    files = {"file": ("xray.jpg", f, "image/jpeg")}
    response = requests.post(
        "http://localhost:8001/predict", 
        files=files,
        headers=headers
    )
    
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

## Reproducibility

All experiments are fully reproducible:
- **Fixed random seeds**: Python (42), NumPy (42), PyTorch (42)
- **Environment**: `pyproject.toml` + `requirements-installed.txt`
- **Experiment tracking**: MLflow logs all hyperparameters, metrics, and artifacts
- **Containerization**: Docker ensures environment consistency
- **CI/CD Pipeline**: GitHub Actions automated testing and linting

```bash
# View experiment history
mlflow ui
# Visit: http://localhost:5000
```

## Testing

```bash
# Run all tests
make test

# Run API integration tests
python tests/test_api.py

# Lint code
make lint

# Format code
make format
```

## Dataset

**Source**: [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/muhammadrehan00/chest-xray-dataset)

**Statistics**:
- Total images: 25,553
- Classes: Normal (35.6%), Pneumonia (22.8%), Tuberculosis (41.6%)
- Splits: Train (80%), Validation (10%), Test (10%)
- Resolution: Mean 1,342×1,215 pixels