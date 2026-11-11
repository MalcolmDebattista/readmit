-- Runs once, when the database is first created. One row per scored discharge; no identifiers.
CREATE TABLE risk_scores (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    discharge_disposition_id INTEGER NOT NULL,
    number_inpatient INTEGER NOT NULL,
    prior_enc INTEGER NOT NULL,
    eligible BOOLEAN NOT NULL,
    score DOUBLE PRECISION,          -- NULL when not eligible (death or hospice)
    model_version TEXT NOT NULL
);
