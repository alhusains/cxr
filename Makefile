.PHONY: help create_environment requirements data train evaluate serve clean test

help:
	@echo "Available commands:"
	@echo "  make create_environment  - Create Python virtual environment"
	@echo "  make requirements        - Install Python dependencies"
	@echo "  make data               - Download and process data from Kaggle"
	@echo "  make train              - Train model with default config"
	@echo "  make evaluate           - Evaluate trained model"
	@echo "  make serve              - Launch inference API"
	@echo "  make test               - Run tests"
	@echo "  make clean              - Remove build artifacts and cache"

create_environment:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

requirements:
	pip install --upgrade pip
	pip install -r requirements.txt

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

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf mlruns/
