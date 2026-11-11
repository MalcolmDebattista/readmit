"""Scoring one discharge, for the API. It reuses the training code path: add_codes -> build_features -> the champion's
columns, so serving cannot drift from training (checked by tests/test_serving.py on real holdout discharges)."""
import pandas as pd

from readmit.features import build_features
from readmit.kb import MEDS, add_codes

NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses", "prior_enc",
           "admission_type_id", "admission_source_id", "discharge_disposition_id"]
TEXT = ["max_glu_serum", "A1Cresult", "change", "diabetesMed", "medical_specialty", "diag_1", "diag_2", "diag_3"]
SERVING_INPUTS = NUMERIC + TEXT + MEDS


def record_frame(records: list[dict]) -> pd.DataFrame:
    """Raw discharge records (as the EHR / CSV spell them) -> the frame build_features expects."""
    df = pd.DataFrame(records)
    for c in NUMERIC:
        df[c] = df[c].astype(int)
    for c in TEXT + MEDS:
        df[c] = df[c].astype(str)
    return add_codes(df)


def is_eligible(discharge_disposition_id: int, bundle: dict) -> bool:
    """Trap 4: a patient who died or went to hospice cannot be readmitted, so is never put on the call list."""
    t = bundle["tables"]
    return discharge_disposition_id not in t["DEATH_IDS"] + t["HOSPICE_IDS"]


def score_records(records: list[dict], bundle: dict):
    X = build_features(record_frame(records), bundle["cq1"], bundle["cq2"], bundle["tables"])[bundle["features"]]
    # the categories the model was trained with, in the same order (unseen values become missing, as in LightGBM)
    for c, cats in bundle["categories"].items():
        X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    return bundle["model"].predict_proba(X)[:, 1]
