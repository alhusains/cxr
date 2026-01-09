# Chest X-Ray Classification System

A production-ready deep learning system for chest X-ray pathology classification.

## Project Organization

```
.
├── config/                 # Configuration files (Hydra configs)
├── data/
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Cleaned, processed data
│   └── external/          # External data sources
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── reports/               # Generated analysis and figures
│   ├── figures/          # Visualizations
│   └── metrics/          # Performance metrics
├── src/                   # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model architectures and training
│   ├── evaluation/       # Evaluation metrics and analysis
│   ├── explainability/   # Grad-CAM and interpretability
│   └── deployment/       # Inference API and serving
├── tests/                 # Unit and integration tests
├── requirements.txt       # Python dependencies
├── setup.py              # Package installation
├── Makefile              # Reproducible commands
└── .env                  # Environment variables
```

## Setup

### Prerequisites
- Python 3.8+ (tested with 3.10-3.13)
- CUDA-capable GPU (recommended)
- Kaggle API credentials

### Installation

```bash
# 1. Clone and navigate
git clone <repository-url>
cd cxr

# 2. Create and activate virtual environment
make create_environment
source venv/bin/activate

# 3. Install all dependencies
make requirements

# 4. Configure Kaggle API
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Download and prepare data
make data
```
