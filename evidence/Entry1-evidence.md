# Diary Entry 1: evidence (Task 26, readmission risk)

This file holds the measured facts only. The reasoning, the AI Act tier, the online-vs-batch decision and the reflection
are for **you** to write (the spec allows AI for code and language only). Places where a judgement is needed are
marked **YOUR CALL**.

All measurements were taken on 11 Nov 2026. The repo is github.com/MalcolmDebattista/readmit, commits `aa12716` (package) and
`9bc2b13` (Makefile fix). The VM is LUbuntu (2 vCPU, 4.3 GB RAM) with Docker.

## 1. Headline metric and training time

| What | Value |
|---|---|
| Holdout (new patients, frozen, SHA-256 `86c7d8ac388e6f54…`) | AUPRC **0.153** (base rate 0.088, lift 1.74), ROC-AUC 0.637 |
| At capacity (25 calls / 195 discharges a week) | precision **16.5 %**, recall 24.1 % (simple rule: 12.7 %) |
| Returning patients in the test period | AUPRC 0.286, precision at capacity 34.4 % |
| Champion | "at discharge" (16 features); adding KB or codes did not beat it by the bootstrap rule |
| Training time, champion only | 1.8 s on the laptop, 5.3–5.7 s in the container |
| Whole pipeline (load → KB → 4 models → holdout) | 13.8 s on the laptop, 40–41 s in the container; the notebook takes 58 s by its own timer (93 s wall) |
| Package vs notebook | Identical to 0.0005 on every number; `readmit-train` exits 1 if not |

## 2. Fresh copy

| Step | Result |
|---|---|
| `git clone` into an empty `python:3.12-slim` container, `pip install -r requirements.txt` (pinned), `-e .[dev]` | 144 s |
| `readmit-fetch` → `readmit-check-data` → `readmit-train` | 59 s; AUPRC 0.153 reproduced |
| `ruff check` / `pytest -q` | clean / **25 passed** |
| Total, from an empty container to tested model | **222 s**, no manual steps |
| Issue found | `make build` failed on the VM: `python3` there is 3.10 (no `tomllib`), so the version came out empty (`invalid tag "readmit:"`). Fixed by reading the version with `sed` (commit `9bc2b13`) |

## 3. Container

| What | Value |
|---|---|
| Image | `readmit:0.1.0` and `readmit:0.1.0-9bc2b13`, **1.03 GB** (python:3.12-slim + LightGBM/pandas/pyarrow/rdflib) |
| Build | 281 s; the model is trained inside the build, and the build fails if the notebook's numbers are not reproduced |
| Stack (`make up`) | api :8888, Postgres 18, Prometheus, Grafana |

## 4. Data quirks found (counts from the real CSV)

| Quirk | Count | What breaks if ignored | Handling, and test |
|---|---|---|---|
| Death dispositions (ids 11, 19, 20, 21) | 1,652 encounters | Model "learns" the dead are never readmitted, and they reach the call list | Not eligible; API returns no score (`test_traps.py`) |
| Hospice (ids 13, 14) | 771 | Same | Same |
| …of which in the test period | 280, and **3 of them reached the call list** (the notebook's own count) | Calls to families of the dead | — |
| "?" and "None" are real values | weight "?" 96.9 %, A1C "None" 83.3 %, glucose "None" 94.7 % | `dropna()` keeps **0 rows** | `keep_default_na=False`; schema rejects NaN |
| ICD-9 codes stored as floats | 866 codes (457 rare); 2,635 primary diagnoses misfiled, 1,688 of them "38" (= 038, septicaemia) | Wrong chapter, wrong group | `norm_icd9`; SHACL flags 43 malformed raw codes |
| Patient history from the future (Trap 1) | — | AUPRC jumps 0.153 → **0.280** (fake) | `prior_enc` = earlier encounters only |
| **New:** stays after the patient's own "Expired" discharge | **14 encounters, 13 patients** | Data is not a clean timeline | Reported by `readmit-check-data` (warning, not removed) |
| "<30" labels with no later encounter in the file | 2,166 of 11,357 | Shows the label comes from outside this extract | Noted only |

## 5. Training–serving skew

| Check | Result |
|---|---|
| `test_model.py::test_serving_gives_the_same_scores_as_training` | 200 holdout records scored through `serve.score_records` = training scores (atol 1e-12) |
| `test_api.py` | 20 records through HTTP `POST /risk` = training scores (4 dp) |
| Category alignment experiment | Without aligning categories the scores are **still identical**, because LightGBM stores its own category list. Alignment is kept only as a safety net |

## 6. Online service under load

Load = real frozen-holdout discharges replayed in order (`monitoring/replay.py`, open loop).
Demand: **195 discharges/week ≈ 28/day ≈ 0.0003/s**. Raw results are in `load-tests.jsonl`, and the screenshot is `grafana-load-test.jpg`.

| Run | Rate | Sent / OK | p50 | **p95** | p99 |
|---|---|---|---|---|---|
| Realistic burst: a day's discharges per minute, kept up for 10 min | 0.5/s | 300 / 300 | 547 ms | **758 ms** | 824 ms |
| 1.5 weeks of discharges in 5 min | 1/s | 300 / 300 | 565 ms | **928 ms** | 1,494 ms |
| Stress | 1.5/s | 90 / 90 | 24 s | 40 s | 42 s |
| Stress | 2/s | 120 / 55 (65 timed out at 60 s) | 43 s | 53 s | 55 s |

- **Capacity is about 1 request/s.** Above that the queue grows without limit.
- With a 10 s timeout at 2/s, 109 of 120 requests failed. The server kept scoring the abandoned requests for minutes afterwards (no load shedding), and `/health` took 14 s. It needed a restart.
- **Prometheus p95 = 0.96 s against a client p95 of 0.76 s.** The latency histogram buckets jump from 0.5 s to 1.0 s, so Prometheus interpolates. The buckets were designed for the Gozo model (milliseconds) and are too coarse here.
- **Postgres log:** 1,385 eligible scores plus 1 death discharge (id 11, `eligible=false`, no score), all tagged `model_version 0.1.0`, with no patient identifiers.

### Where the time goes (one request, inside the container)

| Step | ms |
|---|---|
| Record → frame (`record_frame`, ICD-9 normalising) | 35 |
| `build_features` (the notebook's, about 100 columns, pandas) | **334** |
| Whole `score_records` | 438 |
| Insert into Postgres (new connection per request) | 56 |

### Online vs batch: cost per discharge (same code, same container)

| Discharges scored together | Total | Per discharge |
|---|---|---|
| 1 (online) | 395 ms | **395 ms** |
| 30 (one day) | 362 ms | 12 ms |
| 195 (one week) | 380 ms | **1.9 ms** |
| 10,781 (whole holdout) | 1.36 s | 0.13 ms |

The cost is almost all fixed per call (pandas building the whole feature frame), not per row.

**YOUR CALL:**
- Online (score each discharge as the ADT feed records it), batch (score the day's discharges nightly or each afternoon), or human-in-the-loop on top of either? Points to weigh:
  - the nurse calls within a few days, not seconds;
  - 395 ms vs 2 ms;
  - the capacity of 1/s against a demand of 0.0003/s;
  - how fresh the list needs to be.
- What would you change first? For example:
  - compute only the champion's 16 features when serving;
  - async or batched endpoint;
  - load shedding or timeouts;
  - finer latency buckets;
  - a connection pool.

## 7. Regulation and architecture, facts to reason from

- **The output:** a ranked call list of discharged patients, which a nurse uses to decide whom to phone. The API note says it "orders the call worklist; a nurse decides".
- **Data used:** health data (diagnoses, medications, stays). That is GDPR Art. 9 special-category data. Ids are salted hashes, and no identifiers are logged.
- **YOUR CALL:**
  - EU AI Act tier: is this a medical device under MDR Rule 11 (software informing clinical decisions → class IIa or higher → Annex I high-risk), or a care-coordination tool (limited or minimal risk)?
  - What does your answer require (risk management, logging, human oversight, accuracy on subgroups)?
- **EA four layers, mapped from the build:**

| Layer | What this project has |
|---|---|
| Business | 25 calls a week to reduce readmissions |
| Data | the CSV, the knowledge base, the Postgres log |
| Application | the package and the API |
| Technology | Docker, compose, Prometheus/Grafana, the VM |

  **YOUR CALL:** where are the gaps?

## Files

- `grafana-load-test.jpg`: the dashboard during the load runs (19:50–20:06). Panels:
  - requests per second: the 1/s and 2/s stress runs, then the 0.5/s realistic run;
  - latency p50/p95;
  - inputs.
- `load-tests.jsonl`: every load run's summary.
- `postgres-log-summary.txt`: prediction log counts by model version.
- Reproduce:
  - `make build up`
  - `python monitoring/export_requests.py`
  - `python3 monitoring/replay.py --rate 0.5 --count 300`
