"""Section 1 of the notebook: loading, governance on load, and the code tables.

The function bodies are the notebook's; the only change is that the notebook's globals (RAW, PROCESSED, SALT,
KAGGLE_HANDLE, UCI_ZIP, DISP ...) are passed in as arguments, so the code runs outside the notebook.
"""
import hashlib
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from readmit.config import load_config, processed_dir, raw_dir, setup_logging

logger = logging.getLogger(__name__)

INT_COLS = ["encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id",
            "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "number_diagnoses"]
UNKNOWN_WORDS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"}


def get_raw(filename: str, raw: Path, kaggle_handle: str) -> Path:
    local = raw / filename
    if local.exists():
        return local
    import kagglehub
    return Path(kagglehub.dataset_download(kaggle_handle, path=filename))


def get_ids_mapping(raw: Path, uci_zip: str) -> Path:
    """IDs_mapping.csv is not in the Kaggle packaging; the UCI original ships it (as IDS_mapping.csv)."""
    local = raw / "IDs_mapping.csv"
    if not local.exists():
        with urllib.request.urlopen(uci_zip, timeout=120) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
            local.write_bytes(z.read(next(n for n in z.namelist() if n.lower().endswith("ids_mapping.csv"))))
    return local


def read_id_tables(path: Path) -> dict:
    """The file stacks three two-column tables (admission type, discharge disposition, admission source)."""
    tables, current = {}, None
    for key, desc in pd.read_csv(path, header=None, dtype=str, keep_default_na=False).itertuples(index=False):
        key, desc = key.strip(), desc.strip()
        if key.endswith("_id"):
            current = key; tables[current] = {}
        elif key:
            tables[current][int(key)] = desc
    return tables


def read_raw_csv(path) -> pd.DataFrame:
    """dtype=str and keep_default_na=False: '?' and 'None' stay exactly as written (Trap 5)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df[INT_COLS] = df[INT_COLS].astype(int)
    return df


def load_encounters(raw_csv: Path, cache: Path, salt: str) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    df = read_raw_csv(raw_csv)
    df = df.sort_values("encounter_id").reset_index(drop=True)
    # ---- governance on load: keep only the ORDER of encounter_id, pseudonymise the patient, drop weight ----
    df["order"] = (np.arange(len(df)) + 0.5) / len(df)
    df["patient"] = [hashlib.sha256(f"{salt}{v}".encode()).hexdigest()[:16] for v in df.patient_nbr]
    df = df.drop(columns=["encounter_id", "patient_nbr", "weight"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def code_tables(ids: dict) -> dict:
    """DISP, ADM_SRC, ADM_TYPE and the death / hospice ids, decoded from the mapping file (Trap 4)."""
    disp = ids["discharge_disposition_id"]
    death = sorted(k for k, v in disp.items() if "expired" in v.lower())
    hospice = sorted(k for k, v in disp.items() if "hospice" in v.lower() and k not in death)
    return {"DISP": disp, "ADM_SRC": ids["admission_source_id"], "ADM_TYPE": ids["admission_type_id"],
            "DEATH_IDS": death, "HOSPICE_IDS": hospice}


def destination(i: int, t: dict) -> str:
    d = t["DISP"].get(i, "Unknown/Invalid")
    if i in t["DEATH_IDS"]: return "died"
    if i in t["HOSPICE_IDS"]: return "hospice"
    if d in UNKNOWN_WORDS: return "not recorded"
    if d == "Left AMA": return "left against advice"
    if d == "Discharged to home": return "home"
    if "to home" in d: return "home with services"
    return "another facility or hospital"


def prepare(df: pd.DataFrame, t: dict, target: str) -> pd.DataFrame:
    """Notebook cells 4, 6 and 8: patient key, label, eligibility, destination and the as-of history."""
    df = df.copy()
    df["pid"] = pd.factorize(df.patient)[0]          # integer key: grouping on 16-character strings is slow
    df["y"] = (df.readmitted == target).astype(int)
    df["eligible"] = ~df.discharge_disposition_id.isin(t["DEATH_IDS"] + t["HOSPICE_IDS"])
    df["destination"] = df.discharge_disposition_id.map(lambda i: destination(i, t))
    # this patient's EARLIER encounters only, in id order (Trap 1: never count later ones)
    df["prior_enc"] = df.groupby("pid", sort=False).cumcount()
    return df


def fetch_main():
    """readmit-fetch: download diabetic_data.csv and IDs_mapping.csv from UCI (no Kaggle account needed)."""
    setup_logging()
    config = load_config()
    raw = raw_dir(config)
    raw.mkdir(parents=True, exist_ok=True)
    processed_dir(config).mkdir(parents=True, exist_ok=True)
    if (raw / "diabetic_data.csv").exists() and (raw / "IDs_mapping.csv").exists():
        logger.info("Data already in %s", raw)
        return
    with urllib.request.urlopen(config["uci_zip"], timeout=300) as r, zipfile.ZipFile(io.BytesIO(r.read())) as z:
        for name in z.namelist():
            if name.lower().endswith("diabetic_data.csv"):
                (raw / "diabetic_data.csv").write_bytes(z.read(name))
            elif name.lower().endswith("ids_mapping.csv"):
                (raw / "IDs_mapping.csv").write_bytes(z.read(name))
    logger.info("Downloaded the UCI files into %s", raw)
