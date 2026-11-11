import pytest


@pytest.fixture(scope="module")
def client(bundle):
    from fastapi.testclient import TestClient
    import readmit.api as api
    return TestClient(api.app), api


def as_request(record):
    from readmit.kb import MEDS
    body = {k: v for k, v in record.items() if k not in MEDS}
    body["medications"] = {m: record[m] for m in MEDS if record[m] != "No"}
    return body


def test_health(client):
    c, _ = client
    assert c.get("/health").json()["status"] == "ok"


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
