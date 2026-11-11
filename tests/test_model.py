import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from readmit.config import load_config
from readmit.model import precision_at_k, split
from readmit.serve import score_records

CONFIG = load_config()


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) == (0.5, 0.5)


def test_split_keeps_each_patient_in_one_period():
    """Trap 2: a returning patient is set aside, never in two periods."""
    df = pd.DataFrame({"pid": [0, 1, 0, 2, 3], "order": [0.1, 0.5, 0.8, 0.75, 0.9], "eligible": True})
    m = split(df, 0.7, 0.85)
    assert m["train"].tolist() == [True, True, False, False, False]
    assert m["valid"].tolist() == [False, False, False, True, False]
    assert m["test"].tolist() == [False, False, False, False, True]
    assert m["returning_test"].tolist() == [False, False, False, False, False]   # patient 0 at 0.8 is "returning" in valid


def test_report_reproduces_the_notebook():
    """C2: the last readmit-train run matched the notebook's numbers."""
    path = Path(CONFIG["report_path"])
    if not path.exists():
        pytest.skip("run readmit-train first")
    r, e = json.loads(path.read_text()), CONFIG["expected"]
    assert r["rows"] == {"train": e["train"], "valid": e["valid"], "test": e["test"]}
    assert round(r["holdout"]["AUPRC"], 3) == e["holdout_auprc"]
    assert round(r["holdout"]["ROC-AUC"], 3) == e["holdout_roc_auc"]
    assert round(r["holdout"]["precision@capacity"], 3) == e["precision_at_capacity"]


def test_serving_gives_the_same_scores_as_training(bundle):
    """Training-serving skew: 200 real holdout discharges, scored through the API's path (raw record -> features)
    must equal the batch scores computed during training."""
    check = bundle["serving_check"]
    served = score_records(check["records"], bundle)
    np.testing.assert_allclose(served, check["scores"], rtol=0, atol=1e-12)
