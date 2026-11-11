"""readmit-check-data: the input check. The raw CSV is validated before anything trains on it, so the card's
traps (and the one it does not list) fail loudly instead of quietly changing the model.

Exit code 0: the file is as the card describes; 1: a check failed (each failure is logged with its count).
"""
import logging
import sys

import pandas as pd
import pandera.pandas as pa
from dotenv import load_dotenv

from readmit.config import load_config, raw_dir, setup_logging
from readmit.data import DOSE, ICD9_AS_CSV, LAB, code_tables, read_id_tables, read_raw_csv

logger = logging.getLogger(__name__)


SCHEMA = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(int, unique=True),
        "patient_nbr": pa.Column(int),
        "readmitted": pa.Column(str, pa.Check.isin(["NO", ">30", "<30"])),            # Trap 3: three values
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),                   # card: stays of 1-14 days
        "number_inpatient": pa.Column(int, pa.Check.ge(0)),
        "discharge_disposition_id": pa.Column(int),
        "diag_1": pa.Column(str, pa.Check.str_matches(ICD9_AS_CSV)),                  # Trap 6: as the CSV spells them
        "diag_2": pa.Column(str, pa.Check.str_matches(ICD9_AS_CSV)),
        "diag_3": pa.Column(str, pa.Check.str_matches(ICD9_AS_CSV)),
        "A1Cresult": pa.Column(str, pa.Check.isin(LAB)),                                # Trap 5: 'None' is a value
        "max_glu_serum": pa.Column(str, pa.Check.isin(LAB)),
        "insulin": pa.Column(str, pa.Check.isin(DOSE)),
    },
    checks=pa.Check(lambda d: d.notna().all(axis=None), error="NaN found: read with dtype=str, keep_default_na=False (Trap 5)"),
    strict=False,
    coerce=False,
)


def after_death(df: pd.DataFrame, death_ids: list) -> pd.DataFrame:
    """The quirk the card does not list: encounters dated (by id order) after the patient's own 'expired' discharge."""
    first_death = df[df.discharge_disposition_id.isin(death_ids)].groupby("patient_nbr").encounter_id.min()
    later = df[df.patient_nbr.isin(first_death.index)]
    return later[later.encounter_id > later.patient_nbr.map(first_death)]


def check(df: pd.DataFrame, t: dict) -> list:
    problems = []
    try:
        SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        problems.append("schema:\n" + err.failure_cases[["column", "check", "failure_case"]].head(20).to_string())
    unknown = set(df.discharge_disposition_id) - set(t["DISP"])
    if unknown:
        problems.append(f"discharge_disposition_id not in the mapping file: {sorted(unknown)}")
    dead = df[df.discharge_disposition_id.isin(t["DEATH_IDS"])]
    if (dead.readmitted != "NO").any():                                                  # Trap 4
        problems.append(f"{(dead.readmitted != 'NO').sum()} deaths carry a readmission label")
    return problems


def main():
    load_dotenv()
    setup_logging()
    config = load_config()
    raw = raw_dir(config)
    t = code_tables(read_id_tables(raw / "IDs_mapping.csv"))
    df = read_raw_csv(raw / "diabetic_data.csv")
    problems = check(df, t)
    ghosts = after_death(df, t["DEATH_IDS"])
    # Known and accepted: logged with its count, not fatal, because removing rows would change the given model (C2).
    logger.warning("%d encounters (%d patients) come after the patient's own 'expired' discharge; recorded, not removed",
                   len(ghosts), ghosts.patient_nbr.nunique())
    for p in problems:
        logger.error(p)
    if not problems:
        logger.info("%s: %d rows match the card", raw / "diabetic_data.csv", len(df))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
