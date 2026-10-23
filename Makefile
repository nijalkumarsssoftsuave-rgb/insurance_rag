.PHONY: help install bootstrap up down logs api worker ui setup migrate ingest seed lint fmt type test test-security eval clean

help:
	@echo "bootstrap       one-shot: up, setup, ingest sample docs, seed claims"
	@echo "install         install deps (CPU torch wheel first)"
	@echo "up / down       start / stop the data plane (qdrant, postgres, valkey)"
	@echo "setup           download models, init qdrant, run migrations"
	@echo "ingest          load pdf/ into the vector store (default tenant)"
	@echo "seed            insert the sample claims used by Lane B and the demos"
	@echo "api/worker/ui   run a process locally"
	@echo "lint fmt type   ruff check / ruff format / mypy"
	@echo "test            full suite    test-security  authz regression only"
	@echo "eval            golden-set baseline run"

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[ui,eval,dev]"

# A stranger's first run: data plane up and healthy, models/collection/schema
# ready, sample policy PDFs ingested, sample claims seeded. `up --wait` needs
# the healthchecks docker-compose.yml already defines for qdrant/postgres/valkey.
bootstrap: up setup ingest seed
	@echo "Bootstrap complete. Run 'make api' and 'make ui' in two more shells."

up:
	docker compose up -d --wait qdrant postgres valkey

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

ingest:
	python scripts/ingest_folder.py --path pdf

seed:
	python scripts/seed_claims.py

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
	QUERY_EXPANSION_ENABLED=false python -m eval.harness -k 3 --label baseline --out eval/runs/baseline.json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
