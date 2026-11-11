"""Write every frozen-holdout discharge (deaths and hospice included) as a /risk request body (JSON lines), in discharge order.

    python monitoring/export_requests.py            # -> data/processed/holdout_requests.jsonl
"""
import json

from readmit.config import load_config, processed_dir, raw_dir
from readmit.data import code_tables, load_encounters, prepare, read_id_tables
from readmit.kb import MEDS
from readmit.model import split
from readmit.serve import SERVING_INPUTS

config = load_config()
t = code_tables(read_id_tables(raw_dir(config) / "IDs_mapping.csv"))
df = prepare(load_encounters(raw_dir(config) / "diabetic_data.csv", processed_dir(config) / "26_encounters.parquet",
                             config["salt"]), t, config["target"])
te = split(df, config["train_end_q"], config["valid_end_q"])["test"]
out = processed_dir(config) / "holdout_requests.jsonl"
with open(out, "w", encoding="utf-8") as f:
    for rec in df.loc[te, SERVING_INPUTS].to_dict("records"):
        body = {k: (int(v) if hasattr(v, "item") else v) for k, v in rec.items() if k not in MEDS}
        body["medications"] = {m: rec[m] for m in MEDS if rec[m] != "No"}
        f.write(json.dumps(body) + "\n")
print(f"{int(te.sum())} requests written to {out}")
