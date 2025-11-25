.PHONY: help install train export test clean demo format lint

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the model"
	@echo "  make export     - Export model to ONNX"
	@echo "  make test       - Run tests"
	@echo "  make demo       - Run demonstration"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code with flake8"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

export:
	python scripts/export_onnx.py

test:
	pytest tests/ -v

demo:
	python examples/demo.py

format:
	black .

lint:
	flake8 .

clean:
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf */*/__pycache__
	rm -rf .pytest_cache
	rm -rf models/checkpoints/*.pth
	rm -rf models/checkpoints/*.pkl
	rm -rf models/onnx/*.onnx
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
