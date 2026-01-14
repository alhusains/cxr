.PHONY: help create_environment requirements data train train-tune evaluate serve clean test lint format

help:
	@echo "Available commands:"
	@echo "  make create_environment  - Create Python virtual environment"
	@echo "  make requirements        - Install all dependencies"
	@echo "  make data               - Download and process data from Kaggle"
	@echo "  make train              - Train model with default hyperparameters"
	@echo "  make train-tune         - Train with hyperparameter tuning (Bayesian optimization)"
	@echo "  make evaluate           - Evaluate trained model on test set"
	@echo "  make serve              - Launch inference API"
	@echo "  make test               - Run tests"
	@echo "  make lint               - Run linting checks"
	@echo "  make format             - Format code with black"
	@echo "  make clean              - Remove build artifacts and cache"

create_environment:
	python3 -m venv venv
	@echo "=========================================="
	@echo "Virtual environment created!"
	@echo "Activate: source venv/bin/activate"
	@echo "=========================================="

requirements:
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	pip freeze > requirements-installed.txt
	@echo "=========================================="
	@echo "Dependencies installed successfully!"
	@echo "Installed versions saved to requirements-installed.txt"
	@echo "=========================================="

data:
	python src/data/download_data.py
	python src/data/eda.py

train:
	python src/models/train.py

train-tune:
	python src/models/train.py --tune --n_trials 15

evaluate:
	python src/evaluation/evaluate.py

evaluate-robustness:
	python src/evaluation/robustness.py

explainability:
	python src/explainability/gradcam.py

serve:
	uvicorn src.deployment.api:app --host 0.0.0.0 --port 8001 --reload

serve-prod:
	uvicorn src.deployment.api:app --host 0.0.0.0 --port 8001 --workers 4

docker-build:
	docker build -t cxr-api:1.0.0 .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

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
