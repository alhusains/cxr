# Reproducibility Guide

This document explains how we ensure complete reproducibility of all experiments and results.

## Environment Isolation

### Virtual Environment (venv)
The project uses Python's built-in `venv` for isolation:

```bash
make create_environment
source venv/bin/activate
```

**Isolation guarantees:**
- Completely isolated from system Python packages
- Isolated from other virtual environments
- No interference with other users' environments on shared workstations
- Packages installed here do NOT affect global Python or other projects

**Verification:**
```bash
which python  # Should show: /home/alhusain/scratch/cxr/venv/bin/python
pip list      # Shows only packages in this environment
```

### Alternative: Conda (if preferred)
```bash
conda create -n cxr python=3.10
conda activate cxr
pip install -e .
```

## Dependency Management

### Two Installation Options

**Option 1: pyproject.toml (Recommended for development)**
```bash
make requirements  # Installs from pyproject.toml
```
- Modern Python standard (PEP 517/518)
- Installs compatible versions within specified ranges
- Good for development and updates

**Option 2: requirements-lock.txt (Strict reproducibility)**
```bash
make requirements-lock  # Installs exact pinned versions
```
- Exact versions that were tested
- Guaranteed reproducibility
- Use this for final evaluation/submission

### Generating Your Own Lock File

After installation, capture your exact environment:
```bash
pip freeze > installed-versions-$(date +%Y%m%d).txt
```

This creates a snapshot of your exact dependency versions.

## Random Seed Management

All sources of randomness are seeded:

```python
# In src/utils.py
def set_seed(seed: int = 42):
    random.seed(seed)           # Python random
    np.random.seed(seed)        # NumPy
    torch.manual_seed(seed)     # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

This is called at the start of every script (training, evaluation, data processing).

## Hardware and Software Documentation

### Automatic Logging

All experiments log:
- Python version
- PyTorch version
- CUDA version
- GPU model and count
- System memory
- Timestamp
- Git commit hash

### Manual Documentation

Document your hardware in `TECHNICAL_REPORT.md`:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Experiment Tracking

### MLflow Integration

All experiments tracked with:
- All hyperparameters
- Training/validation metrics per epoch
- Model checkpoints
- Configuration files
- System information

View experiments:
```bash
mlflow ui
```

### Configuration Management

All runs use Hydra/YAML configs:
- Configs versioned in git
- Each experiment logs its config
- Configs saved with model checkpoints

## Data Versioning

### Download Reproducibility
```bash
# Kaggle dataset is versioned
# Download script records:
- Dataset version
- Download timestamp
- SHA256 checksums
```

### Data Splits
- Fixed random seed for splits
- Split indices saved to disk
- Same train/val/test split across all runs

## Model Checkpointing

Model files include:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'config': ...,  # Complete configuration
    'epoch': ...,
    'train_metrics': ...,
    'val_metrics': ...,
    'random_state': ...,  # RNG states
    'python_version': ...,
    'pytorch_version': ...,
    'git_commit': ...
}
```

## Running Reproducible Experiments

### Step-by-step workflow:

1. **Clean environment**
```bash
make clean
rm -rf venv/
```

2. **Fresh setup**
```bash
make create_environment
source venv/bin/activate
make requirements-lock  # Use locked versions
```

3. **Verify installation**
```bash
pip list > verification-$(whoami).txt
# Compare with requirements-lock.txt
```

4. **Download data**
```bash
make data  # Uses fixed seed
```

5. **Train model**
```bash
make train  # All seeds set automatically
```

6. **Results**
All outputs in `reports/` and `mlruns/` are now reproducible.

## Verification Checklist

Before claiming reproducibility:

- [ ] Virtual environment is isolated and documented
- [ ] All package versions are pinned (requirements-lock.txt)
- [ ] Random seeds are set in all scripts
- [ ] Hardware specifications documented
- [ ] Git commit hash recorded
- [ ] Data splits are saved and reused
- [ ] Model checkpoints include all metadata
- [ ] MLflow experiments contain complete configs
- [ ] Can rebuild environment from scratch
- [ ] Can rerun training and get same results (within numerical precision)

## Known Sources of Non-Determinism

Despite best efforts, some operations may have minor numerical differences:

1. **CUDA operations**: Some CUDA operations are non-deterministic
   - Mitigation: `torch.use_deterministic_algorithms(True)` where possible
   
2. **Data loading**: Multi-threaded data loading
   - Mitigation: `num_workers=0` for full determinism (slower)
   
3. **Hardware differences**: Different GPUs may have slight numerical differences
   - Mitigation: Document GPU model
   
4. **Library versions**: Different CUDA/cuDNN versions
   - Mitigation: Pin all versions

## Sharing Your Environment

For reviewers to reproduce:

1. **Provide requirements-lock.txt** with exact versions
2. **Document hardware** (GPU model, CUDA version)
3. **Include git commit hash**
4. **Share MLflow experiments**
5. **Provide configuration files**

They can then:
```bash
git clone <repo>
cd cxr
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-lock.txt
make data
make train
```

And should get identical results (within floating-point precision).
