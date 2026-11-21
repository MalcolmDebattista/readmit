# ADR 0002: The release gate is precision at capacity, at least 15 %, on the frozen holdout

- **Status:** accepted (21 November 2026)
- **Decider:** Malcolm Debattista
- **Applies to:** `readmit-gate` and `make ci` (`config/config.yaml`: `gate_metric`, `gate_threshold`)

## Context

A model version may be released only if it passes one check on data it never saw. The check runs on every push
(`make ci`: lint, tests, data validation, gate).

**The holdout.** New patients first met in the last 15 % of the `encounter_id` order, from order 0.85 on. The data has
no dates, so the order stands in for time. That makes 10,781 eligible discharges, with a base rate of 8.8 %. It is
frozen by SHA-256 `86c7d8ac388e6f5424d54b34d5e3067b2ebd7aff02e833977a5137f0b74c655f`. If the fingerprint changes,
the gate refuses to run (exit 2).

**The operating point.** 25 calls per 195 discharges, so the top 12.8 % (1,382 holdout discharges) are called.

**Candidates.** Each is scored on the holdout, with a 95 % patient-level bootstrap interval (1,000 resamples) and a
baseline. The baseline is the prior-stays rule: call the patients with the most inpatient stays in the past year.

| Metric | Given model | 95 % interval | Prior-stays rule | Chance |
|---|---|---|---|---|
| **Precision at capacity** | **0.165** | 0.146 to 0.185 | 0.127 (0.108 to 0.147) | 0.088 |
| AUPRC | 0.153 | 0.136 to 0.171 | 0.117 (0.104 to 0.132) | 0.088 |
| ROC-AUC | 0.637 | 0.618 to 0.656 | 0.561 (0.542 to 0.581) | 0.5 |

Commit `2bbf4aa`, MLflow run `2462ae6941de47f89389e85640058dcb` (registered as `readmit` version 1).

**What the threshold does to the runs we have:**

| Run | Training window | Precision at capacity | Gate at 0.15 |
|---|---|---|---|
| `2462ae69` | all training patients | 0.165 | pass |
| `2b9fd697` | patients from order 0.35 | 0.159 | pass |
| `7d81cb70` | patients from order 0.50 | 0.145 | **fail** |
| `ee1b73e7` | random 70/30 split (not the holdout) | 0.261 | not gated: it is not scored on the frozen holdout |

## Decision

Gate on **precision at capacity ≥ 0.15** on the frozen holdout.

## Reasons

> **YOUR CALL: write this in your own words.** Questions to answer:
> - Why precision at capacity rather than AUPRC or ROC-AUC? Who feels this number, and what does each of the
>   others hide at 25 calls a week?
> - Why 0.15? How does it relate to the rule's upper interval (0.147) and to the model's own lower interval (0.146)?
> - What would you lose with a higher threshold, for example 0.16? (The given model itself would then sit inside
>   its own noise.)

## Rejected options

> **YOUR CALL:** for AUPRC and ROC-AUC, say in one line each why it lost, and what it would have let through.
> Hint from the table: ROC-AUC rewards ranking among the 87 % who are never called.

## Consequences

- A version whose 25 weekly calls reach readmissions no more often than the prior-stays rule cannot be released.
- With 1,382 called discharges, the interval is about ±2 points. A model 1 point worse than today may pass or fail
  by chance. **YOUR CALL:** is that acceptable, and what would you do about it?
- Changing the threshold needs a new ADR that supersedes this one, written before the new model is scored.
