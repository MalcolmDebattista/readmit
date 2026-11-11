"""The card's traps, each turned into a test, so the package fails loudly if one is mishandled (Diary Entry 1, task 2)."""
import pandas as pd
import pandera.pandas as pa
import pytest

from readmit.data import prepare, read_raw_csv
from readmit.kb import add_codes, all_codes, build_kb, norm_icd9, shacl_violations
from readmit.serve import is_eligible
from readmit.validate import SCHEMA, after_death

TABLES = {"DISP": {1: "Discharged to home", 11: "Expired", 13: "Hospice / home"}, "ADM_SRC": {}, "ADM_TYPE": {},
          "DEATH_IDS": [11], "HOSPICE_IDS": [13]}


def toy(rows):
    df = pd.DataFrame(rows, columns=["patient", "order", "discharge_disposition_id", "readmitted"])
    return prepare(df, TABLES, "<30")


# ---- Trap 1: history must count EARLIER encounters only --------------------------------------------------
def test_prior_encounters_count_only_the_past():
    df = toy([["a", 0.1, 1, "NO"], ["b", 0.2, 1, "NO"], ["a", 0.3, 1, "<30"], ["a", 0.4, 1, "NO"]])
    assert df.prior_enc.tolist() == [0, 0, 1, 2]       # a whole-file count would give 3, 1, 3, 3


# ---- Trap 3: only "<30" is positive -------------------------------------------------------------------------
def test_label_is_thirty_day_readmission_only():
    df = toy([["a", 0.1, 1, "NO"], ["b", 0.2, 1, ">30"], ["c", 0.3, 1, "<30"]])
    assert df.y.tolist() == [0, 0, 1]


# ---- Trap 4: death and hospice, decoded from the mapping file, are never eligible ---------------------------
def test_death_and_hospice_ids_decoded_from_mapping_file(tables):
    assert tables["DEATH_IDS"] == [11, 19, 20, 21]
    assert tables["HOSPICE_IDS"] == [13, 14]


def test_death_and_hospice_are_not_eligible():
    df = toy([["a", 0.1, 1, "NO"], ["b", 0.2, 11, "NO"], ["c", 0.3, 13, "<30"]])
    assert df.eligible.tolist() == [True, False, False]
    assert df.destination.tolist() == ["home", "died", "hospice"]


def test_api_never_scores_a_hospice_discharge(bundle):
    assert not is_eligible(13, bundle) and not is_eligible(11, bundle) and is_eligible(1, bundle)


# ---- Trap 5: '?' and 'None' are values, not missing ---------------------------------------------------------
def test_none_and_question_mark_survive_loading(tmp_path):
    from readmit.data import INT_COLS
    row = {c: "1" for c in INT_COLS} | {"A1Cresult": "None", "race": "?", "max_glu_serum": "None"}
    path = tmp_path / "raw.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    df = read_raw_csv(path)
    assert df.A1Cresult.iloc[0] == "None" and df.race.iloc[0] == "?"
    assert not df.isna().any(axis=None)


def test_input_check_rejects_a_default_pandas_read():
    """pd.read_csv's default turns 'None' into NaN; the schema must refuse such a frame."""
    good = {"encounter_id": [1], "patient_nbr": [1], "readmitted": ["NO"], "time_in_hospital": [3], "number_inpatient": [0],
            "discharge_disposition_id": [1], "diag_1": ["428"], "diag_2": ["?"], "diag_3": ["250.01"],
            "A1Cresult": ["None"], "max_glu_serum": ["None"], "insulin": ["No"]}
    SCHEMA.validate(pd.DataFrame(good))
    with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
        SCHEMA.validate(pd.DataFrame(good | {"A1Cresult": [float("nan")]}), lazy=True)


# ---- Trap 6: ICD-9 codes as the CSV spells them --------------------------------------------------------------
@pytest.mark.parametrize("raw,code", [("38", "038"), ("8", "008"), ("250.1", "250.10"), ("250", "250.00"),
                                      ("V57", "V57"), ("428", "428"), ("?", None)])
def test_icd9_codes_are_normalised(raw, code):
    assert norm_icd9(raw) == code


def test_shacl_fails_raw_codes_and_passes_normalised_ones():
    df = add_codes(pd.DataFrame({"diag_1": ["38"], "diag_2": ["250.1"], "diag_3": ["428"]}))
    raw_of = {"038": "38", "250.10": "250.1", "428": "428"}
    naive = shacl_violations(build_kb(all_codes(df), notation=lambda c: raw_of[c]))
    assert naive.get("not a well-formed ICD-9-CM code", 0) == 1          # '38' is not a valid notation
    assert shacl_violations(build_kb(all_codes(df))).get("not a well-formed ICD-9-CM code", 0) == 0


# ---- The quirk the card does not list: encounters after the patient's own death --------------------------
def test_encounters_after_death_are_found():
    df = pd.DataFrame({"encounter_id": [1, 2, 3, 4], "patient_nbr": [7, 7, 8, 7], "discharge_disposition_id": [1, 11, 1, 1]})
    ghosts = after_death(df, [11])
    assert ghosts.encounter_id.tolist() == [4]
