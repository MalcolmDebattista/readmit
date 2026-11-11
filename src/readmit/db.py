"""Every scored discharge is logged with its model version (as in Gozo Stage 10). No identifiers are stored."""
import logging
import os

import psycopg

logger = logging.getLogger(__name__)

INSERT_SQL = """
    INSERT INTO risk_scores (discharge_disposition_id, number_inpatient, prior_enc, eligible, score, model_version)
    VALUES (%s, %s, %s, %s, %s, %s)
"""


def save_score(discharge_disposition_id, number_inpatient, prior_enc, eligible, score, model_version):
    url = os.environ.get("DATABASE_URL")
    if not url:
        logger.debug("DATABASE_URL not set; score not stored")
        return
    try:
        with psycopg.connect(url) as conn:
            conn.execute(INSERT_SQL, (discharge_disposition_id, number_inpatient, prior_enc, eligible, score, model_version))
    except psycopg.Error:
        logger.exception("Could not store the score")
