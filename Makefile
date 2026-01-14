.PHONY: help install dev docker-up docker-down migrate run test lint type-check format clean celery

help:
	@echo "SignalForge Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install dependencies with uv"
	@echo "  dev          Install dev dependencies"
	@echo "  docker-up    Start Docker services (PostgreSQL + Redis)"
	@echo "  docker-down  Stop Docker services"
	@echo "  migrate      Run database migrations"
	@echo "  run          Start the FastAPI development server"
	@echo "  test         Run tests"
	@echo "  lint         Run linter (ruff)"
	@echo "  type-check   Run type checker (mypy)"
	@echo "  format       Format code with ruff"
	@echo "  celery       Start Celery worker"
	@echo "  clean        Clean up cache files"

install:
	uv sync

dev:
	uv sync --all-extras

docker-up:
	docker compose up -d

docker-down:
	docker compose down

migrate:
	uv run alembic upgrade head

migrate-create:
	@read -p "Migration message: " msg; \
	uv run alembic revision --autogenerate -m "$$msg"

run:
	uv run uvicorn signalforge.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	uv run pytest

test-cov:
	uv run pytest --cov=signalforge --cov-report=html

lint:
	uv run ruff check src tests

type-check:
	uv run mypy src

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

celery:
	uv run celery -A signalforge.workers.celery_app worker --loglevel=info

celery-beat:
	uv run celery -A signalforge.workers.celery_app beat --loglevel=info

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
