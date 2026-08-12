.PHONY: help install up down logs api worker ui setup migrate lint fmt type test test-security eval clean

help:
	@echo "install         install deps (CPU torch wheel first)"
	@echo "up / down       start / stop the data plane (qdrant, postgres, valkey)"
	@echo "setup           download models, init qdrant, run migrations"
	@echo "api/worker/ui   run a process locally"
	@echo "lint fmt type   ruff check / ruff format / mypy"
	@echo "test            full suite    test-security  authz regression only"
	@echo "eval            golden-set baseline run"

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[ui,eval,dev]"

up:
	docker compose up -d qdrant postgres valkey

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

setup:
	python scripts/download_models.py
	python scripts/init_qdrant.py
	alembic upgrade head

migrate:
	alembic upgrade head

api:
	python scripts/run_api.py   # not plain uvicorn - see the script docstring

worker:
	celery -A app.workers.celery_app worker --loglevel=INFO --concurrency=2

ui:
	streamlit run ui/app.py

lint:
	ruff check app ui eval tests scripts

fmt:
	ruff format app ui eval tests scripts
	ruff check --fix app ui eval tests scripts

type:
	mypy app

test:
	pytest -v

test-security:
	pytest -v -m security

eval:
	python -m eval.harness --dataset eval/golden/questions.jsonl

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
