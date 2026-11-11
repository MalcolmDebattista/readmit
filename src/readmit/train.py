"""readmit-train: the notebook's Parts A-B as one command. Trains the four-model ladder on the training patients,
chooses the champion on validation, reports the frozen holdout once, and checks the notebook's numbers (C2)."""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from readmit.config import load_config, model_path, processed_dir, raw_dir, setup_logging
from readmit.data import code_tables, get_ids_mapping, get_raw, load_encounters, prepare, read_id_tables
from readmit.features import build_features, ladder
from readmit.kb import add_codes, all_codes, build_kb, competency_tables, shacl_violations
from readmit.model import boot_weights, choose_champion, evaluate, fit, holdout_fingerprint, save_bundle, split
from readmit.serve import SERVING_INPUTS

logger = logging.getLogger(__name__)


def run(config):
    t_start = time.perf_counter()
    raw = raw_dir(config)
    csv = get_raw("diabetic_data.csv", raw, config["kaggle_handle"])
    t = code_tables(read_id_tables(get_ids_mapping(raw, config["uci_zip"])))
    df = load_encounters(csv, processed_dir(config) / "26_encounters.parquet", config["salt"])
    df = add_codes(prepare(df, t, config["target"]))
    logger.info("%d encounters, %d patients; death ids %s, hospice ids %s", len(df), df.pid.nunique(), t["DEATH_IDS"], t["HOSPICE_IDS"])

    kb = build_kb(all_codes(df))
    violations = shacl_violations(kb)
    code_violations = int(violations.drop("medication has no therapeutic class", errors="ignore").sum())
    if code_violations:
        logger.error("SHACL: %d code violations in the knowledge base:\n%s", code_violations, violations)
        raise SystemExit(1)
    cq1, cq2, _ = competency_tables(kb)
    X = build_features(df, cq1, cq2, t)

    m = split(df, config["train_end_q"], config["valid_end_q"])
    tr, va, te = m["train"], m["valid"], m["test"]
    y = df.y
    y_va, y_te = y[va].values, y[te].values
    call_share = config["calls_per_week"] / config["discharges_per_week"]
    params = dict(config["lgb_params"], random_state=config["seed"], n_jobs=config["n_jobs"])
    np.random.seed(config["seed"])

    models, p_val, fit_seconds, valid_rows = {}, {}, {}, []
    for name, cols in ladder(X).items():
        t0 = time.perf_counter()
        models[name] = fit(X, y, cols, tr, params)
        fit_seconds[name] = round(time.perf_counter() - t0, 2)
        p_val[name] = models[name].predict_proba(X.loc[va, cols])[:, 1]
        valid_rows.append(evaluate(name, y_va, p_val[name], call_share))
    champion, low = choose_champion(y_va, p_val, boot_weights(df.pid[va].values, config["n_boot"], config["seed"]))
    feats, model = ladder(X)[champion], models[champion]

    p_te = model.predict_proba(X.loc[te, feats])[:, 1]          # the holdout, reported once
    holdout = evaluate(f"holdout: new patients ({champion})", y_te, p_te, call_share)
    p_ret = model.predict_proba(X.loc[m["returning_test"], feats])[:, 1]
    returning = evaluate("test-period returning patients", y[m["returning_test"]].values, p_ret, call_share)
    sha = holdout_fingerprint(df, X, te, feats)
    total = round(time.perf_counter() - t_start, 1)

    sample = df.loc[te, SERVING_INPUTS].head(200)
    bundle = {"model": model, "features": feats, "champion": champion, "cq1": cq1, "cq2": cq2, "tables": t,
              "categories": {c: list(X[c].cat.categories.astype(str)) for c in feats if str(X[c].dtype) == "category"},
              "holdout_sha256": sha, "version": os.environ.get("MODEL_VERSION", "dev"),
              "serving_check": {"records": sample.to_dict("records"), "scores": p_te[:200].tolist()}}
    report = {"rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
              "validation_ladder": valid_rows, "champion": champion, "champion_rule_ci_low": round(low, 4),
              "holdout": holdout, "returning_patients": returning, "holdout_sha256": sha,
              "fit_seconds": fit_seconds, "champion_fit_seconds": fit_seconds[champion], "pipeline_seconds": total,
              "shacl": violations.to_dict()}
    return bundle, report


def check_against_notebook(report, config):
    """C2: the package must reproduce the notebook's numbers within the stated tolerance."""
    e, tol, h = config["expected"], config["tolerance"], report["holdout"]
    problems = [f"{k}: {report['rows'][k]} != {e[k]}" for k in ("train", "valid", "test") if report["rows"][k] != e[k]]
    for key, got in [("holdout_auprc", h["AUPRC"]), ("holdout_roc_auc", h["ROC-AUC"]), ("precision_at_capacity", h["precision@capacity"])]:
        if abs(round(got, 3) - e[key]) > tol:
            problems.append(f"{key}: {got:.4f} vs notebook {e[key]}")
    return problems


def main():
    load_dotenv()
    setup_logging()
    config = load_config()
    try:
        bundle, report = run(config)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
    save_bundle(bundle, model_path(config))
    out = Path(config["report_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    h = report["holdout"]
    logger.info("Champion: %s (fit %.1f s; whole pipeline %.1f s)", report["champion"], report["champion_fit_seconds"], report["pipeline_seconds"])
    logger.info("Holdout: AUPRC %.3f, ROC-AUC %.3f, precision@capacity %.1f%%, recall@capacity %.1f%%",
                h["AUPRC"], h["ROC-AUC"], 100 * h["precision@capacity"], 100 * h["recall@capacity"])
    problems = check_against_notebook(report, config)
    if problems:
        logger.error("Does NOT reproduce the notebook: %s", "; ".join(problems))
        sys.exit(1)
    logger.info("Reproduces the notebook's numbers (tolerance %s). Model saved to %s", config["tolerance"], model_path(config))


if __name__ == "__main__":
    main()
