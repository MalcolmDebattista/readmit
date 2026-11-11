# readmit: 30-day readmission risk after discharge (ARI3217 Task 26)

The Task 26 notebook (Diabetes 130-US hospitals, UCI dataset 296) turned into an installable package, a scoring
service and a monitored deployment. The notebook remains the reference. The package reproduces its numbers exactly
and the build fails if it does not.

## Quick start (fresh copy)

```bash
git clone https://github.com/MalcolmDebattista/readmit && cd readmit
python -m pip install -r requirements.txt          # pinned runtime libraries
python -m pip install --no-deps -e ".[dev]"
python -m pip install -r requirements-dev.txt      # pytest, ruff, pandera, httpx
cp .env.example .env
readmit-fetch          # downloads the UCI zip into data/raw/diabetes130/
readmit-check-data     # input schema + data quirks
readmit-train          # writes models/model.joblib and reports/metrics.json, checks against the notebook
pytest -q
```

On Linux, `make install fetch check-data train test` does the same.

## Package

| Command / module | What it does |
|---|---|
| `readmit-fetch` (`data.py`) | Downloads and unpacks the raw CSV and the ID mapping |
| `readmit-check-data` (`validate.py`) | pandera schema on the raw CSV, plus quirk checks (unknown disposition ids, stays after death) |
| `readmit-train` (`train.py`) | Notebook pipeline: split by time, knowledge base, feature ladder, champion by bootstrap, frozen holdout |
| `kb.py` | ICD-9 knowledge base (rdflib), SHACL code checks, competency queries CQ1–CQ3 |
| `features.py`, `model.py` | Feature builder and model code, unchanged from the notebook |
| `serve.py`, `api.py` | Scoring of raw discharge records; FastAPI `POST /risk` |
| `config/config.yaml` | Every notebook constant, plus the numbers the package must reproduce |

## Data traps handled (and tested in `tests/test_traps.py`)

| Trap | Handling |
|---|---|
| Patient history from the future | `prior_enc` counts only earlier encounters (cumcount in encounter order) |
| Label is "<30" only | `y = readmitted == "<30"`; ">30" counts as negative |
| Deaths and hospice | Disposition ids 11, 19, 20, 21 and 13, 14 are not eligible; the API returns `eligible: false` and no score |
| "None" and "?" are values | Read with `keep_default_na=False`; the schema rejects NaN |
| ICD-9 codes stored as floats | `norm_icd9` restores them (e.g. `250.1` vs `250.10`, `V57`); SHACL flags malformed raw codes |
| Stays after death (found here) | 14 encounters of 13 patients after an "Expired" discharge; reported by `readmit-check-data` |

## Service

```bash
make build        # docker image readmit:<version>, model trained inside the build
make up           # api :8888, Postgres, Prometheus :9090, Grafana :3000
curl -X POST localhost:8888/risk -H 'Content-Type: application/json' -d @example.json
make load         # replays the holdout discharges against /risk and records latency in reports/load.jsonl
make down
```

Every score is written to Postgres (`risk_scores`) with the model version. `/metrics` exposes request counts,
latency, the score distribution, the `number_inpatient` input distribution and the count of not-eligible requests.
The Grafana dashboard "Readmission risk service" is provisioned automatically.

## Tests

`pytest -q` runs 31 tests: the traps, input values outside the code tables (refused with 422), the model helpers, a check that the package reproduces the notebook's
metrics, a check that serving scores equal training scores, and API tests. Tests that need the data or the trained
model are skipped on a fresh clone until `readmit-fetch` and `readmit-train` have been run.

## Data

The data is not in this repository. `readmit-fetch` downloads it from the UCI Machine Learning Repository
(dataset 296, CC BY 4.0). Patient ids are replaced by salted hashes when loaded.
