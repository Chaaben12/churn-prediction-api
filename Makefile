.PHONY: help install format lint typecheck test run docker-build docker-up docker-down

help:
	@echo "install       sync dev environment (uv)"
	@echo "format        apply ruff formatting"
	@echo "lint          ruff lint + mypy strict"
	@echo "test          pytest with coverage gate"
	@echo "run           local API on http://localhost:8000"
	@echo "docker-build  build the production image"
	@echo "docker-up     start the compose stack"
	@echo "docker-down   stop the compose stack"

install:
	uv sync --dev

format:
	uv run ruff format src tests training

lint:
	uv run ruff check src tests training
	uv run mypy src training

typecheck:
	uv run mypy src training

test:
	uv run pytest

run:
	uv run uvicorn churn_api.main:app --reload

docker-build:
	docker build -t churn-prediction-api:latest .

docker-up:
	docker compose up --build

docker-down:
	docker compose down
