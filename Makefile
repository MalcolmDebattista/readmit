# Task 26 pipeline. Recipes must start with a TAB.
# Read with sed so it works on any python3 (tomllib needs 3.11+)
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
IMAGE ?= readmit
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
SOURCE ?=
LABELS := $(if $(SOURCE),--label org.opencontainers.image.source=$(SOURCE))

.PHONY: install fetch check-data train lint test test-model gate build publish up down load version ci register champion drift

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install --no-deps -e ".[dev]"
	python3 -m pip install -r requirements-dev.txt

fetch:
	readmit-fetch

check-data:
	readmit-check-data

# Session 06: train and record the run in MLflow (TRAIN_START_Q=0.35 for a later window; SPLIT=random to compare)
TRAIN_START_Q ?=
SPLIT ?= forward
train:
	readmit-train --split $(SPLIT) $(if $(TRAIN_START_Q),--train-start-q $(TRAIN_START_Q))

# Register the run just recorded: refused unless it passed the gate on the frozen holdout from committed code
register: train
	readmit-registry register

# Point the champion alias at version V and restart the API, which loads it at start-up.
# Rolling back is the same command with the previous version: make champion V=1
champion:
	readmit-registry alias champion $(V)
	docker compose restart api

# Monitoring evidence: PSI and the model's metric per slice of the order (reports/drift.csv, reports/drift.png)
drift:
	readmit-drift

# The quality gate on the frozen holdout (exit 1 below the threshold, 2 if the holdout changed)
gate:
	readmit-gate

lint:
	ruff check src tests

test:
	pytest -q

# The tests that need the model the gate just judged: serving skew, the API, the notebook's numbers (C2)
test-model:
	pytest -q tests/test_model.py tests/test_api.py

build:
	docker build --build-arg APP_VERSION=$(VERSION) -t $(IMAGE):$(VERSION) -t $(IMAGE):$(VERSION)-$(GIT_SHA) $(LABELS) .

publish:
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):$(VERSION)-$(GIT_SHA)

# The whole check, as GitHub runs it (Session 05): lint, tests, data validation, the gate, then the tests that need
# the judged model, and the image. make stops at the first step that fails.
ci: fetch lint test check-data gate test-model build

up:
	APP_VERSION=$(VERSION) docker compose up -d --build

down:
	docker compose down

load:
	python3 monitoring/export_requests.py
	python3 monitoring/replay.py --rate 5 --count 1500 --label steady --out reports/load.jsonl

version:
	@echo $(VERSION)
