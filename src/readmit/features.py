"""Section 3 of the notebook: build_features and the four feature sets of the ladder.

build_features is the notebook's, unchanged, except that the code tables (ADM_TYPE, ADM_SRC, DISP) and the KB
lookups are arguments. The SAME function is used for training and for serving (no training-serving skew).
"""
import pandas as pd

from readmit.data import UNKNOWN_WORDS
from readmit.kb import CHAPTER_LABEL, MED_CLASS, MEDS


def unknown_to_one(series: pd.Series, table: dict) -> pd.Series:
    return series.map(lambda v: "unknown" if table.get(v, "NULL") in UNKNOWN_WORDS else str(v))


def build_features(df: pd.DataFrame, lookup, dm, t: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    # ---- base: the stay and the prior year, as the EHR knows them at discharge ----
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
              "number_emergency", "number_inpatient", "number_diagnoses", "prior_enc"]:
        X[c] = df[c]
    X["admission_type"] = unknown_to_one(df.admission_type_id, t["ADM_TYPE"])
    X["admission_source"] = unknown_to_one(df.admission_source_id, t["ADM_SRC"])
    X["discharge_disposition"] = unknown_to_one(df.discharge_disposition_id, t["DISP"])
    for c in ["max_glu_serum", "A1Cresult", "change", "diabetesMed", "medical_specialty"]:
        X[c] = df[c]
    # ---- raw: the codes as the CSV spells them ----
    for i in (1, 2, 3):
        X[f"diag_{i}"] = df[f"diag_{i}"]
    for m in MEDS:
        X[f"med_{m}"] = df[m]
    # ---- KB: the same facts, through the ontology ----
    for i in (1, 2, 3):
        X[f"chapter_{i}"] = df[f"code_{i}"].map(lookup.chapter).fillna("missing")
    X["group_1"] = df.code_1.map(lookup.group).fillna("missing")
    for ch in CHAPTER_LABEL:
        X[f"n_chapter_{ch}"] = sum((X[f"chapter_{i}"] == ch).astype(int) for i in (1, 2, 3))
    dm_code = df.code_1.where(df.code_1.isin(dm.index)).fillna(df.code_2.where(df.code_2.isin(dm.index))).fillna(
        df.code_3.where(df.code_3.isin(dm.index)))
    X["dm_complication"] = dm_code.map(dm.complication).fillna("no diabetes code")
    X["dm_type1"] = (dm_code.map(dm.type1) == "true").astype(int)
    X["dm_uncontrolled"] = (dm_code.map(dm.uncontrolled) == "true").astype(int)
    for cl in sorted({k for v in MED_CLASS.values() for k in v}):
        members = [m for m, ks in MED_CLASS.items() if cl in ks]
        X[f"class_{cl}_on"] = (df[members] != "No").any(axis=1).astype(int)
        X[f"class_{cl}_changed"] = df[members].isin(["Up", "Down"]).any(axis=1).astype(int)
    X["n_meds_on"] = (df[MEDS] != "No").sum(axis=1)
    X["n_meds_up"] = (df[MEDS] == "Up").sum(axis=1)
    X["n_meds_down"] = (df[MEDS] == "Down").sum(axis=1)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype).startswith("str"):
            X[c] = X[c].astype("category")
    return X


# what the EHR holds when the discharge is recorded ...
AT_DISCHARGE = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
                "number_emergency", "number_inpatient", "prior_enc", "admission_type", "admission_source",
                "discharge_disposition", "max_glu_serum", "A1Cresult", "change", "diabetesMed", "medical_specialty"]


def ladder(X: pd.DataFrame) -> dict:
    """The notebook's four feature sets, from what exists at discharge to what exists after coding."""
    med_raw = [f"med_{m}" for m in MEDS]
    med_kb = [c for c in X.columns if c.startswith(("class_", "n_meds_"))]
    dx_raw = ["number_diagnoses"] + [f"diag_{i}" for i in (1, 2, 3)]
    dx_kb = ["number_diagnoses"] + [c for c in X.columns if c.startswith(("chapter_", "group_", "n_chapter_", "dm_"))]
    return {"at discharge": AT_DISCHARGE,
            "at discharge + medication classes (KB)": AT_DISCHARGE + med_kb,
            "after coding + raw codes": AT_DISCHARGE + med_raw + dx_raw,
            "after coding + knowledge base": AT_DISCHARGE + med_kb + dx_kb}
