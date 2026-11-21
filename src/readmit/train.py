"""readmit-train: the notebook's Parts A-B as one command. Trains the four-model ladder on the training patients,
chooses the champion on validation, reports the frozen holdout once, and checks the notebook's numbers (C2).

Since Entry 2 (Session 06) every run is recorded in MLflow: parameters, metrics, the git commit and the
fingerprints of the data and the holdout. MLFLOW_TRACKING_URI=none trains without recording (tests, offline work).

  readmit-train                          the given model, forward split (must reproduce the notebook)
  readmit-train --split random           the same model on a random 70/30 split of discharges (for comparison only)
  readmit-train --train-start-q 0.35     train only on training patients met from order 0.35 on (a later window)
"""
import argparse
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from readmit.config import load_config, model_path, processed_dir, raw_dir, setup_logging
from readmit.data import code_tables, get_ids_mapping, get_raw, load_encounters, prepare, read_id_tables
from readmit.features import build_features, ladder
from readmit.gate import assess, prior_stays_rule, verdict
from readmit.kb import add_codes, all_codes, build_kb, competency_tables, shacl_violations
from readmit.model import boot_weights, choose_champion, evaluate, fit, holdout_fingerprint, save_bundle, split
from readmit.serve import SERVING_INPUTS

logger = logging.getLogger(__name__)

LAST_RUN = "models/last_run_id"


def load_all(config):
    """Sections 1-3: the encounters, the code tables, the knowledge base and the feature frame."""
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
    return df, build_features(df, cq1, cq2, t), t, cq1, cq2, violations


def run(config, split_kind="forward"):
    """Train and score. Returns the model bundle, the report, and the scored test rows (for the gate)."""
    t_start = time.perf_counter()
    df, X, t, cq1, cq2, violations = load_all(config)

    m = split(df, config["train_end_q"], config["valid_end_q"])
    tr, va, te = m["train"], m["valid"], m["test"]
    if config.get("train_start_q"):
        # A later training window: only training patients met from this point in the order on.
        first_order = df.groupby("pid").order.transform("min")
        tr = tr & (first_order >= config["train_start_q"])
        logger.info("Training window starts at order %.2f: %d training discharges", config["train_start_q"], tr.sum())
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
    scored = {"y": y_te, "p": p_te, "pid": df.pid[te].values,
              "rule": prior_stays_rule(X.loc[te, "number_inpatient"].values, config["seed"])}

    if split_kind == "random":
        # The notebook's Trap 2: random 70/30 rows of eligible discharges, same champion and parameters.
        el = np.flatnonzero(df.eligible.values)
        a, b = train_test_split(el, test_size=0.3, random_state=config["seed"])
        model = lgb.LGBMClassifier(**params).fit(X.iloc[a][feats], y.iloc[a])
        p_b = model.predict_proba(X.iloc[b][feats])[:, 1]
        holdout = evaluate(f"random 30 % of discharges ({champion})", y.iloc[b].values, p_b, call_share)
        holdout["shared_patients"] = float(np.isin(df.pid.values[b], df.pid.values[a]).mean())
        scored = {"y": y.iloc[b].values, "p": p_b, "pid": df.pid.values[b],
                  "rule": prior_stays_rule(X.iloc[b]["number_inpatient"].values, config["seed"])}
        sha = "random split: not the frozen holdout"
    total = round(time.perf_counter() - t_start, 1)

    sample = df.loc[te, SERVING_INPUTS].head(200)
    bundle = {"model": model, "features": feats, "champion": champion, "cq1": cq1, "cq2": cq2, "tables": t,
              "categories": {c: list(X[c].cat.categories.astype(str)) for c in feats if str(X[c].dtype) == "category"},
              "holdout_sha256": sha, "version": os.environ.get("MODEL_VERSION", "dev"),
              "serving_check": {"records": sample.to_dict("records"), "scores": model.predict_proba(X.loc[te, feats].head(200))[:, 1].tolist()}}
    report = {"split": split_kind, "train_start_q": config.get("train_start_q"),
              "rows": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())},
              "validation_ladder": valid_rows, "champion": champion, "champion_rule_ci_low": round(low, 4),
              "holdout": holdout, "returning_patients": returning, "holdout_sha256": sha,
              "fit_seconds": fit_seconds, "champion_fit_seconds": fit_seconds[champion], "pipeline_seconds": total,
              "shacl": violations.to_dict()}
    return bundle, report, scored


def check_against_notebook(report, config):
    """C2: the package must reproduce the notebook's numbers within the stated tolerance."""
    e, tol, h = config["expected"], config["tolerance"], report["holdout"]
    problems = [f"{k}: {report['rows'][k]} != {e[k]}" for k in ("train", "valid", "test") if report["rows"][k] != e[k]]
    for key, got in [("holdout_auprc", h["AUPRC"]), ("holdout_roc_auc", h["ROC-AUC"]), ("precision_at_capacity", h["precision@capacity"])]:
        if abs(round(got, 3) - e[key]) > tol:
            problems.append(f"{key}: {got:.4f} vs notebook {e[key]}")
    return problems


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    """The commit the code came from, and whether tracked files differ from it. A run from a dirty tree cannot be
    rebuilt from its commit, so it is recorded as such, and readmit-registry refuses to register it."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), str(bool(git("status", "--porcelain", "--untracked-files=no"))).lower()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def track(config, report, bundle, gate, split_kind):
    """Record the run in MLflow and log the bundle as a model the registry can hold."""
    import mlflow
    from readmit.registry import BundleModel

    h = report["holdout"]
    mlflow.log_params({
        "seed": config["seed"], "split": split_kind, "train_end_q": config["train_end_q"], "valid_end_q": config["valid_end_q"],
        "train_start_q": config.get("train_start_q") or "none", "champion": report["champion"],
        "calls_per_week": config["calls_per_week"], "discharges_per_week": config["discharges_per_week"],
        "train_rows": report["rows"]["train"], "test_rows": len(gate["y"]) if "y" in gate else report["rows"]["test"],
        **{f"lgb_{k}": v for k, v in config["lgb_params"].items()},
    })
    mlflow.log_metrics({
        "holdout_auprc": h["AUPRC"], "holdout_roc_auc": h["ROC-AUC"], "holdout_lift": h["lift"],
        "holdout_precision_at_capacity": h["precision@capacity"], "holdout_recall_at_capacity": h["recall@capacity"],
        "valid_auprc": next(r["AUPRC"] for r in report["validation_ladder"] if r["model"] == report["champion"]),
        "gate_low": gate["result"]["metrics"][config["gate_metric"]]["low"],
        "gate_high": gate["result"]["metrics"][config["gate_metric"]]["high"],
        "baseline_rule": gate["result"]["metrics"][config["gate_metric"]]["rule"],
        "pipeline_seconds": report["pipeline_seconds"],
    })
    commit, dirty = git_state()
    csv = raw_dir(config) / "diabetic_data.csv"
    mlflow.set_tags({
        "git_commit": commit, "git_dirty": dirty,
        "data_sha256": sha256_file(csv), "holdout_sha256": report["holdout_sha256"],
        "holdout_frozen": str(report["holdout_sha256"] == config["holdout_sha256"]).lower(),
        "gate_metric": config["gate_metric"], "gate_threshold": config["gate_threshold"],
        "gate_passed": str(gate["passed"]).lower(),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.joblib"
        save_bundle(bundle, path)
        logged = mlflow.pyfunc.log_model(name="model", python_model=BundleModel(), artifacts={"bundle": str(path)},
                                         pip_requirements=["readmit", "lightgbm", "pandas", "scikit-learn"])
        mlflow.log_dict(report, "report.json")
    mlflow.set_tag("model_uri", logged.model_uri)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the readmission model and track the run in MLflow.")
    parser.add_argument("--split", choices=["forward", "random"], default="forward")
    parser.add_argument("--train-start-q", type=float, help="train only on training patients first met from this order on")
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging()
    config = load_config()
    if args.train_start_q is not None:
        config["train_start_q"] = args.train_start_q or None

    uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    if uri.lower() == "none":
        logger.warning("Tracking is off: this run is not recorded")
        run_context = contextlib.nullcontext()
    else:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow"]["experiment"])
        run_context = mlflow.start_run()

    with run_context as tracked:
        try:
            bundle, report, scored = run(config, args.split)
        except SystemExit:
            raise
        except Exception:
            logger.exception("Training failed")
            sys.exit(1)
        result = assess(scored["y"], scored["p"], scored["rule"], scored["pid"], config)
        passed = args.split == "forward" and report["holdout_sha256"] == config["holdout_sha256"] and verdict(result, config)
        report["gate"] = {"metric": config["gate_metric"], "threshold": config["gate_threshold"], "passed": passed, **result}

        save_bundle(bundle, model_path(config))
        out = Path(config["report_path"]); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        h = report["holdout"]
        logger.info("Champion: %s (fit %.1f s; whole pipeline %.1f s)", report["champion"], report["champion_fit_seconds"], report["pipeline_seconds"])
        logger.info("%s: AUPRC %.3f, ROC-AUC %.3f, precision@capacity %.1f%%, recall@capacity %.1f%%; gate %s",
                    h["model"], h["AUPRC"], h["ROC-AUC"], 100 * h["precision@capacity"], 100 * h["recall@capacity"],
                    "passed" if passed else "not passed")
        if tracked is not None:
            track(config, report, bundle, {"result": result, "passed": passed, "y": scored["y"]}, args.split)
            Path(LAST_RUN).parent.mkdir(parents=True, exist_ok=True)
            Path(LAST_RUN).write_text(tracked.info.run_id)
            logger.info("Run %s recorded in MLflow", tracked.info.run_id)

    # C2 applies to the given model as configured: forward split, the whole training window.
    if args.split == "forward" and not config.get("train_start_q"):
        problems = check_against_notebook(report, config)
        if problems:
            logger.error("Does NOT reproduce the notebook: %s", "; ".join(problems))
            sys.exit(1)
        logger.info("Reproduces the notebook's numbers (tolerance %s). Model saved to %s", config["tolerance"], model_path(config))


if __name__ == "__main__":
    main()
