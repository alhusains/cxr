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
- Python 3.8+
- CUDA-capable GPU (recommended)
- Kaggle API credentials

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cxr
```

2. Create virtual environment:
```bash
make create_environment
source venv/bin/activate
```

3. Install dependencies:
```bash
make requirements

```

4. Verify installation:
```bash
python --version  # Should show Python 3.8-3.11
which python      # Should show path to venv/bin/python
pip list          # Shows installed packages in environment
```

5. Configure Kaggle credentials:
```bash
# Place kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

6. Download and prepare data:
```bash
make data
```

For detailed reproducibility information, see `REPRODUCIBILITY.md`.

## Quick Start

### Training
```bash
make train
```

### Evaluation
```bash
make evaluate
```