import time

import pytest


@pytest.fixture(scope="module")
def client(bundle):
    """The API with the locally trained bundle (MODEL_URI = a file; no registry needed). It loads in the background."""
    from fastapi.testclient import TestClient
    import readmit.api as api
    with TestClient(api.app) as c:
        for _ in range(100):
            if c.get("/ready").status_code == 200:
                break
            time.sleep(0.1)
        yield c, api


def as_request(record):
    from readmit.kb import MEDS
    body = {k: v for k, v in record.items() if k not in MEDS}
    body["medications"] = {m: record[m] for m in MEDS if record[m] != "No"}
    return body


def test_health_and_ready(client):
    c, api = client
    assert c.get("/health").json() == {"status": "ok"}
    body = c.get("/ready").json()
    assert body["ready"] is True and body["model_version"] == f"file:{api.MODEL_URI}"


def test_alive_but_not_ready_without_a_model(client, monkeypatch, tmp_path, bundle):
    """Session 06: /health 200 at once, /ready and /risk 503 until a model is loaded."""
    from fastapi.testclient import TestClient
    _, api = client
    saved = dict(api.state)
    monkeypatch.setattr(api, "MODEL_URI", str(tmp_path / "missing.joblib"))
    api.state.update(bundle=None, model_version=None)
    try:
        with TestClient(api.app) as c:
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 503
            assert c.post("/risk", json=as_request(bundle["serving_check"]["records"][0])).status_code == 503
    finally:
        api.state.update(saved)


def test_api_matches_the_training_scores(client, bundle):
    """Real holdout discharges through HTTP give the batch scores from training (to the 4 decimals returned)."""
    c, _ = client
    check = bundle["serving_check"]
    for record, expected in list(zip(check["records"], check["scores"]))[:20]:
        body = c.post("/risk", json=as_request(record)).json()
        assert body["eligible"] and body["score"] == pytest.approx(round(expected, 4), abs=1e-9)


def test_hospice_is_never_scored(client, bundle):
    c, _ = client
    record = dict(bundle["serving_check"]["records"][0], discharge_disposition_id=13)
    body = c.post("/risk", json=as_request(record)).json()
    assert body == {"eligible": False, "score": None, "model_version": body["model_version"],
                    "note": "death or hospice: never on the call list"}


def test_bad_request_is_rejected(client, bundle):
    c, _ = client
    record = as_request(bundle["serving_check"]["records"][0])
    assert c.post("/risk", json={k: v for k, v in record.items() if k != "prior_enc"}).status_code == 422
    assert c.post("/risk", json=record | {"time_in_hospital": 30}).status_code == 422
    assert c.post("/risk", json=record | {"patient_name": "x"}).status_code == 422     # no identifiers accepted


def test_metrics_are_exposed(client, bundle):
    c, _ = client
    c.post("/risk", json=as_request(bundle["serving_check"]["records"][1]))
    text = c.get("/metrics").text
    for name in ("predictions_total", "prediction_latency_seconds_bucket", "prediction_input_bucket", "prediction_score_bucket"):
        assert name in text


@pytest.mark.parametrize("change", [
    {"A1Cresult": "banana"},                       # not a lab value
    {"medications": {"insulln": "Up"}},             # misspelt medicine
    {"medications": {"insulin": "Maybe"}},          # not a dose change
    {"discharge_disposition_id": 99},               # unknown code: could be a new code for a death
    {"admission_type_id": 999},
    {"diag_1": "NOTACODE"},
])
def test_values_outside_the_code_tables_are_refused(client, bundle, change):
    c, _ = client
    body = as_request(bundle["serving_check"]["records"][0]) | change
    assert c.post("/risk", json=body).status_code == 422
