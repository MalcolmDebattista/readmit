# Task 26 pipeline. Recipes must start with a TAB.
VERSION := $(shell python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
IMAGE ?= readmit
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)

.PHONY: install fetch check-data train lint test build up down load version

install:
	python3 -m pip install -r requirements.txt
	python3 -m pip install --no-deps -e ".[dev]"
	python3 -m pip install -r requirements-dev.txt

fetch:
	readmit-fetch

check-data:
	readmit-check-data

train:
	readmit-train

lint:
	ruff check src tests

test:
	pytest -q

build:
	docker build --build-arg MODEL_VERSION=$(VERSION) -t $(IMAGE):$(VERSION) -t $(IMAGE):$(VERSION)-$(GIT_SHA) .

up:
	MODEL_VERSION=$(VERSION) docker compose up -d --build

down:
	docker compose down

load:
	python3 monitoring/export_requests.py
	python3 monitoring/replay.py --rate 5 --count 1500 --label steady --out reports/load.jsonl

version:
	@echo $(VERSION)
