# Task 26 pipeline. Recipes must start with a TAB.
# Read with sed so it works on any python3 (tomllib needs 3.11+)
VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
IMAGE ?= readmit
GIT_SHA := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
SOURCE ?=
LABELS := $(if $(SOURCE),--label org.opencontainers.image.source=$(SOURCE))

.PHONY: install fetch check-data train lint test build publish up down load version ci

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
	docker build --build-arg MODEL_VERSION=$(VERSION) -t $(IMAGE):$(VERSION) -t $(IMAGE):$(VERSION)-$(GIT_SHA) $(LABELS) .

publish:
	docker push $(IMAGE):$(VERSION)
	docker push $(IMAGE):$(VERSION)-$(GIT_SHA)

# The whole check, as GitHub runs it: data check, then train (fails if the notebook's numbers are not reproduced),
# lint, the 31 tests, and the image build (which trains again inside the image).
ci: fetch check-data train lint test build

up:
	MODEL_VERSION=$(VERSION) docker compose up -d --build

down:
	docker compose down

load:
	python3 monitoring/export_requests.py
	python3 monitoring/replay.py --rate 5 --count 1500 --label steady --out reports/load.jsonl

version:
	@echo $(VERSION)
