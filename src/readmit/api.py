"""POST /risk: score one discharge when the ADT feed records it (online prediction); a nurse then decides whom to
call (human-in-the-loop). The score orders the worklist; it never withholds care."""
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from readmit.config import load_config, model_path, setup_logging
from readmit.data import ICD9_AS_CSV
from readmit.db import save_score
from readmit.kb import MEDS
from readmit.metrics import INPUTS, LATENCY, NOT_ELIGIBLE, PREDICTIONS, SCORES
from readmit.model import load_bundle
from readmit.serve import is_eligible, score_records

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

config = load_config()
# The code's version (pyproject.toml, through `make build`). The model's version comes from the registry.
APP_VERSION = os.environ.get("APP_VERSION", "dev")
# Which model to serve: a registry alias (models:/readmit@champion, as compose sets it) or a local bundle file.
MODEL_URI = os.environ.get("MODEL_URI", str(model_path(config)))

state = {"bundle": None, "model_version": None}


def load_model(uri):
    """Load the bundle named by uri. A registry alias is resolved to a version first, so we know what we serve."""
    if not uri.startswith("models:/"):
        return load_bundle(Path(uri)), f"file:{uri}"
    import mlflow
    from readmit.registry import load_registered_bundle
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]))
    name, alias = uri.removeprefix("models:/").split("@")
    version = mlflow.MlflowClient().get_model_version_by_alias(name, alias).version
    return load_registered_bundle(name, version), f"{name}/{version}"


def load_until_ready(uri, wait=5):
    """Runs in the background, so /health answers at once. Keeps trying until the model can be loaded."""
    while state["bundle"] is None:
        try:
            bundle, version = load_model(uri)
            state["bundle"], state["model_version"] = bundle, version
            logger.info("Serving %s (champion: %s, holdout %s...), app version %s", version, bundle["champion"],
                        bundle["holdout_sha256"][:12], APP_VERSION)
        except Exception as err:
            logger.warning("No model loaded yet, retrying in %d s: %s", wait, err)
            time.sleep(wait)


@asynccontextmanager
async def lifespan(app):
    # Alive at once; ready only when a model is loaded. Until then /ready and /risk answer 503.
    threading.Thread(target=load_until_ready, args=(MODEL_URI,), daemon=True).start()
    yield


app = FastAPI(title="Readmission risk after discharge (ARI3217 Task 26)", lifespan=lifespan)


class Discharge(BaseModel):
    """One discharge, as the EHR holds it when the discharge is recorded. Codes as the CSV spells them."""
    model_config = ConfigDict(extra="forbid")
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(ge=0)
    num_procedures: int = Field(ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(ge=0)
    number_emergency: int = Field(ge=0)
    number_inpatient: int = Field(ge=0)
    number_diagnoses: int = Field(ge=0)
    prior_enc: int = Field(ge=0, description="this patient's EARLIER encounters only (Trap 1)")
    admission_type_id: int
    admission_source_id: int
    discharge_disposition_id: int
    # Text values: only the values the hospital extract uses, so a typo is refused instead of silently scored
    max_glu_serum: Literal["None", "Norm", ">200", ">300"] = "None"     # 'None' means not measured (Trap 5)
    A1Cresult: Literal["None", "Norm", ">7", ">8"] = "None"
    change: Literal["No", "Ch"] = "No"
    diabetesMed: Literal["No", "Yes"] = "No"
    medical_specialty: str = Field("?", max_length=60)
    diag_1: str = Field("?", pattern=ICD9_AS_CSV)                        # ICD-9 as the CSV spells it, or '?'
    diag_2: str = Field("?", pattern=ICD9_AS_CSV)
    diag_3: str = Field("?", pattern=ICD9_AS_CSV)
    medications: dict[Literal[tuple(MEDS)], Literal["No", "Steady", "Up", "Down"]] = Field(
        default_factory=dict, description="medication column -> No / Steady / Up / Down")

    @field_validator("discharge_disposition_id", "admission_type_id", "admission_source_id")
    @classmethod
    def known_code(cls, v, info):
        """An id missing from the hospital's code table is refused: it could be a new code for a death (Trap 4)."""
        if state["bundle"] is None:          # no model yet: /risk answers 503 before using the value
            return v
        table = state["bundle"]["tables"][{"discharge_disposition_id": "DISP", "admission_type_id": "ADM_TYPE",
                                  "admission_source_id": "ADM_SRC"}[info.field_name]]
        if v not in table:
            raise ValueError(f"unknown {info.field_name} {v}: not in the hospital's code table")
        return v


@app.get("/health")
def health():
    """Liveness: is the process running?"""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    """Readiness: can it score? Only once a model is loaded."""
    if state["bundle"] is None:
        response.status_code = 503
        return {"ready": False, "model_version": None}
    return {"ready": True, "model_version": state["model_version"], "app_version": APP_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/risk")
def risk(d: Discharge):
    bundle, version = state["bundle"], state["model_version"]
    if bundle is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    with LATENCY.time():
        record = d.model_dump(exclude={"medications"}) | {m: d.medications.get(m, "No") for m in MEDS}
        eligible = is_eligible(d.discharge_disposition_id, bundle)
        score = float(score_records([record], bundle)[0]) if eligible else None
        save_score(d.discharge_disposition_id, d.number_inpatient, d.prior_enc, eligible, score, version)
    PREDICTIONS.inc()
    INPUTS.observe(d.number_inpatient)
    if eligible:
        SCORES.observe(score)
    else:
        NOT_ELIGIBLE.inc()
    return {"eligible": eligible, "score": None if score is None else round(score, 4), "model_version": version,
            "note": "orders the call worklist; a nurse decides" if eligible else "death or hospice: never on the call list"}
