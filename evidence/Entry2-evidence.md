# Diary Entry 2: evidence (Task 26, readmission risk)

This file holds the measured facts only. The reasoning (why this metric, threshold, alert and trigger) and the
reflection are for **you** to write. Places that need your judgement are marked **YOUR CALL**.

- **Measured:** 21 Nov 2026, code commit `2bbf4aa` (version 0.2.0).
- **Where things ran:** MLflow 3.17.2 server, Postgres and the API in the LUbuntu VM (Docker). Training ran from the
  Windows repo (Python 3.12), recording to that server through an SSH tunnel.

## 1. Splits: random vs forward (the given model, "at discharge")

| Split | Test rows | Precision at capacity | AUPRC | ROC-AUC | MLflow run |
|---|---|---|---|---|---|
| Random 70/30 of eligible discharges (seed 3217) | 29,804 | **0.261** | 0.242 | 0.681 | `ee1b73e72708420b91f604cfd13a60ce` |
| Forward: frozen holdout, new patients from order 0.85 | 10,781 | **0.165** | 0.153 | 0.637 | `2462ae6941de47f89389e85640058dcb` |

- In the random split, 38 % of test rows share a patient with training.
- The notebook's Trap 2 shows that splitting by patient (still random) gives almost the same: AUPRC 0.225 against
  0.232 for rows. So most of the gap comes from testing the whole decade at once, rather than new patients later.
- Returning patients in the test period score higher: precision 0.344, against 0.165 for new patients.

## 2. The gate

Decision: precision at capacity ≥ 0.15 (ADR `docs/decisions/0002-gate-metric.md`; the reasons are **YOUR CALL**).

| Metric | Model | 95 % interval (patient bootstrap, 1,000) | Prior-stays rule | Chance |
|---|---|---|---|---|
| Precision at capacity (1,382 calls) | 0.165 | 0.146 to 0.185 | 0.127 (0.108 to 0.147) | 0.088 |
| AUPRC | 0.153 | 0.136 to 0.171 | 0.117 (0.104 to 0.132) | 0.088 |
| ROC-AUC | 0.637 | 0.618 to 0.656 | 0.561 (0.542 to 0.581) | 0.5 |

- `make ci` order: fetch, lint, tests (36), data validation, **gate**, model tests, image.
- Failing CI run: **to do, needs a push**. Planned: a branch that trains only on patients from order 0.50 fails
  the gate at 0.145.
- Passing CI run: **to do, needs a push**.

## 3. MLflow runs and registry (experiment `readmit`)

All four runs: commit `2bbf4aa`, `git_dirty=false`, data SHA-256 `0689e7ec…`.

| Run | What | Precision at capacity | Gate | Registry |
|---|---|---|---|---|
| `2462ae69` | given model, all training patients | 0.165 | pass | **version 1, champion** |
| `ee1b73e7` | random split (comparison only) | 0.261 | not on the holdout | refused |
| `2b9fd697` | training patients from order 0.35 (33,133 discharges) | 0.159 | pass | **version 2, challenger** |
| `7d81cb70` | training patients from order 0.50 (16,303 discharges) | 0.145 | **fail** | refused: "did not pass the gate" |

- Holdout fingerprint `86c7d8ac…` is identical for the three forward runs.
- The registry refuses runs that failed the gate or came from uncommitted changes.

## 4. Serving and rollback (VM, image `readmit:0.2.0`, code only, 1.12 GB)

| Check | Result |
|---|---|
| API start | alive after 14 s; `/ready` after 35 s: `{"model_version": "readmit/1", "app_version": "0.2.0"}` |
| `example.json` | v1 score 0.0735; v2 score 0.0767 |
| Rollback: `readmit-registry alias champion 2` + `docker compose restart api` | serving `readmit/2` **23 s** after the alias move |
| Roll forward to v1 | `readmit/1` again |
| `risk_scores` table | rows tagged `readmit/1` and `readmit/2` |

## 5. Drift and the metric on one axis (`evidence/entry2/drift.png`, `drift.csv`)

- Reference: the training period.
- Slices: 5 % of the encounter order after training, about 4,950 eligible discharges each (all of them, returning
  patients included).
- Watched inputs: the champion's top 4 by gain.

| Slice | PSI score | PSI number_inpatient | PSI discharge_disposition | PSI num_lab_procedures | PSI medical_specialty | Precision at capacity | AUPRC | Observed / expected |
|---|---|---|---|---|---|---|---|---|
| 0.70–0.75 | 0.055 | 0.013 | 0.311 | 0.060 | 0.604 | 0.240 | 0.224 | 0.90 |
| 0.75–0.80 | 0.054 | 0.004 | 0.343 | 0.062 | 0.771 | 0.231 | 0.199 | 0.86 |
| 0.80–0.85 | 0.029 | 0.001 | 0.394 | 0.040 | 0.852 | 0.251 | 0.224 | 0.91 |
| 0.85–0.90 | 0.022 | 0.001 | 0.355 | 0.053 | 0.680 | 0.250 | 0.221 | 0.88 |
| 0.90–0.95 | 0.046 | 0.002 | 0.362 | 0.049 | 0.690 | 0.218 | 0.203 | 0.84 |
| 0.95–1.00 | 0.036 | 0.001 | 0.297 | 0.052 | 0.744 | 0.226 | 0.214 | 0.94 |

What stands out:
- Two inputs stay above PSI 0.25 the whole time: medical specialty (recording habits; "unknown" rises) and
  discharge disposition.
- Meanwhile the score PSI stays at 0.02–0.06, and precision at 0.22–0.25.
- The metric and the input drift do **not** move together.

**YOUR CALL:** the alert threshold (on which signal?) and the retraining trigger, justified by these numbers. A
starting point from the notebook: score PSI above 0.10, quarterly retraining on matured labels (30 days plus coding).

## 6. Model card

- Draft: `docs/model_card.md` (3,042 characters; the registry keeps up to 5,000).
- Left for you (**YOUR CALL**): the alert, the trigger, and who may move the alias.
- After that: the peer review with a classmate, one change, then `readmit-registry card 1 docs/model_card.md`.

## Found on the way

- The VM has no Python 3.12, and the pinned numpy needs it, so training ran from Windows into the VM's MLflow.
- `pip install .` inside a container left a root-owned `build/` folder in the repo. It was removed through Docker.
- Removing a second, full `mlflow` package from the venv also deleted files that the pinned `mlflow-skinny` shares
  with it. Fixed by reinstalling the pinned package.
- The image no longer trains the model or holds patient data. It is still 1.12 GB, because the MLflow client brings
  its own dependencies.
