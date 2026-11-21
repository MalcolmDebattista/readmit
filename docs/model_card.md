# Model card: readmit, version 1 (champion)

*Owner: Malcolm Debattista. ARI3217 Task 26. Attach with `readmit-registry card 1 docs/model_card.md`.*

## Intended use

Orders the discharge nurses' follow-up call list. Each diabetic discharge is scored when it is recorded, and the
nurses phone the highest 25 of about 195 a week. **A nurse decides**: they can add or skip anyone, and the score never
withholds care.

- **Users:** the transitional-care nurses, through `POST /risk`.
- **Not for:** deciding who receives care or treatment; clinical advice; staff performance; other conditions or
  hospitals; patients who died or went to hospice (never scored).

## Model

The notebook's LightGBM model, "at discharge" feature set (16 inputs held by the EHR at discharge), unchanged. Run
`2462ae6941de47f89389e85640058dcb`, commit `2bbf4aa`, seed 3217.

## Data

- Diabetes 130-US hospitals 1999–2008 (Strack et al., 2014): 101,766 encounters, de-identified, **no dates**.
  SHA-256 `0689e7ec0312…`.
- Time is the `encounter_id` order. Training: 69,609 discharges of patients first met in the first 70 %.
- **Holdout (frozen):** new patients from order 0.85, 10,781 discharges, base rate 8.8 %, SHA-256 `86c7d8ac…`.

## Metrics (frozen holdout)

| | Model | 95 % interval | Prior-stays rule |
|---|---|---|---|
| **Precision at 25 calls/week (gate ≥ 0.15)** | **0.165** | 0.146–0.185 | 0.127 |
| Recall at 25 calls/week | 0.241 | | |
| AUPRC | 0.153 | 0.136–0.171 | 0.117 |

- One call in six reaches a patient who is readmitted within 30 days. Three readmissions in four are not called.
- On a random 70/30 split the same model scores 0.261. That number is not honest: 38 % of the test rows share a
  patient with training.
- Version 2 (training patients from order 0.35) scored 0.159.

## Limitations

- Ranks, but is not calibrated: it over-predicts (observed / expected 0.85), so read scores as an order, not a risk.
- Misses men's readmissions more often than women's (79.1 % against 73.3 % missed; notebook, Section 5).
- Never sees the medicines' effect, the patient's home situation, or anything after discharge.
- The label comes from outside the extract: 2,166 `<30` labels have no later encounter in the file.
- Data from 1999–2008, US hospitals. Recording habits change along the order (medical specialty PSI up to 0.85).

## Monitoring

`readmit-drift`, per 5 % slice of the order after training:

| | Range seen |
|---|---|
| Score PSI | 0.02–0.06 |
| PSI medical specialty / discharge disposition | 0.60–0.85 / 0.30–0.39 |
| Precision at capacity, all eligible discharges | 0.22–0.25 |

Every score is stored with its registry version (`readmit/1`).

- **Alert threshold:** **YOUR CALL**
- **Retraining trigger:** **YOUR CALL**

## Rollback

- `make champion V=2`: moves the `champion` alias and restarts the API. Measured: serving `readmit/2` 23 s later,
  no rebuild.
- **Who may move the alias:** **YOUR CALL**
- **Afterwards:** check that `/ready` and the newest `risk_scores` rows show the new version.
