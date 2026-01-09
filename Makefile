.PHONY: help create_environment requirements requirements-dev data train evaluate serve clean test lint format

help:
	@echo "Available commands:"
	@echo "  make create_environment  - Create Python virtual environment"
	@echo "  make requirements        - Install Python dependencies (exact versions)"
	@echo "  make requirements-dev    - Install dev dependencies"
	@echo "  make data               - Download and process data from Kaggle"
	@echo "  make train              - Train model with default config"
	@echo "  make evaluate           - Evaluate trained model"
	@echo "  make serve              - Launch inference API"
	@echo "  make test               - Run tests"
	@echo "  make lint               - Run linting checks"
	@echo "  make format             - Format code with black"
	@echo "  make clean              - Remove build artifacts and cache"

create_environment:
	python3 -m venv venv
	@echo "=========================================="
	@echo "Virtual environment created successfully!"
	@echo "Activate with: source venv/bin/activate"
	@echo "=========================================="

requirements:
	pip install --upgrade pip setuptools wheel
	pip install -e .
	@echo "=========================================="
	@echo "Dependencies installed from pyproject.toml"
	@echo "For reproducibility verification, run:"
	@echo "  pip freeze > installed-versions.txt"
	@echo "=========================================="

requirements-dev:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"

requirements-lock:
	pip install --upgrade pip setuptools wheel
	pip install -r requirements-lock.txt
	@echo "=========================================="
	@echo "Exact locked versions installed!"
	@echo "=========================================="

data:
	python src/data/download_data.py
	python src/data/prepare_data.py

train:
	python src/models/train.py

evaluate:
	python src/evaluation/evaluate.py

serve:
	uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
