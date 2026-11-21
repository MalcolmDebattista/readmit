"""readmit-gate: the quality gate on the frozen holdout (Diary Entry 2, Session 05).

Exit codes: 0 pass, 1 below the threshold, 2 the frozen holdout has changed, 3 the data is missing.

The metric and threshold come from config.yaml (gate_metric, gate_threshold); the reasons are in
docs/decisions/0002-gate-metric.md. Every candidate metric is printed next to its baseline and a patient-level
bootstrap interval, so the choice stays visible.
"""
import argparse
import logging
import sys

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score

from readmit.config import load_config, raw_dir, setup_logging
from readmit.model import boot_weights

logger = logging.getLogger(__name__)


def precision_at_capacity(y, p, share, w=None):
    """Share of the calls that reach a readmission, when the top `share` of discharges (by score) are called.
    With bootstrap weights, a patient drawn twice counts twice."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind="stable")
    if w is None:          # exactly the notebook's precision_at_k, with k = round(share x discharges)
        return float(y[order[:int(round(share * len(y)))]].mean())
    w = np.asarray(w, float)
    take = order[np.cumsum(w[order]) <= share * w.sum() + 1e-9]
    return float((y[take] * w[take]).sum() / w[take].sum())


METRICS = {
    "precision@capacity": lambda y, p, share, w=None: precision_at_capacity(y, p, share, w),
    "AUPRC": lambda y, p, share, w=None: float(average_precision_score(y, p, sample_weight=w)),
    "ROC-AUC": lambda y, p, share, w=None: float(roc_auc_score(y, p, sample_weight=w)),
}


def prior_stays_rule(number_inpatient, seed):
    """The rule a care team may already use: most inpatient stays in the past year (ties broken at random)."""
    rng = np.random.default_rng(seed)
    return np.asarray(number_inpatient, float) + rng.random(len(number_inpatient)) * 1e-3


def assess(y, p, rule, pids, config):
    """Every metric for the model and the prior-stays rule, with 95 % patient-level bootstrap intervals."""
    share = config["calls_per_week"] / config["discharges_per_week"]
    weights = boot_weights(np.asarray(pids), config.get("gate_boot", 1000), config["seed"])
    rows = {}
    for name, f in METRICS.items():
        model = [f(y, p, share, w) for w in weights]
        base = [f(y, rule, share, w) for w in weights]
        rows[name] = {"model": f(y, p, share), "low": float(np.percentile(model, 2.5)), "high": float(np.percentile(model, 97.5)),
                      "rule": f(y, rule, share), "rule_low": float(np.percentile(base, 2.5)), "rule_high": float(np.percentile(base, 97.5))}
    return {"metrics": rows, "base_rate": float(np.mean(y)), "n": int(len(y)), "calls": int(round(share * len(y)))}


def verdict(result, config):
    name, threshold = config["gate_metric"], config["gate_threshold"]
    return result["metrics"][name]["model"] >= threshold


def print_result(result, config, label):
    print(f"Scored on {label}: {result['n']:,} discharges, base rate {result['base_rate']:.3f}, {result['calls']:,} calls at capacity")
    print(f"  {'metric':<20}{'model':>8}  {'95 % interval':<16}{'prior-stays rule':>17}")
    for name, r in result["metrics"].items():
        mark = "  <- gate" if name == config["gate_metric"] else ""
        print(f"  {name:<20}{r['model']:>8.3f}  {r['low']:.3f} to {r['high']:.3f}  {r['rule']:>10.3f} ({r['rule_low']:.3f} to {r['rule_high']:.3f}){mark}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the model on the frozen holdout and apply the quality gate.")
    parser.parse_args(argv)
    load_dotenv()
    setup_logging()
    config = load_config()

    if not (raw_dir(config) / "diabetic_data.csv").exists():
        logger.error("Cannot run the gate: %s not found (run readmit-fetch)", raw_dir(config) / "diabetic_data.csv")
        sys.exit(3)

    import json
    from pathlib import Path

    from readmit.config import model_path
    from readmit.model import save_bundle
    from readmit.train import run          # imported here: the training pipeline is heavy
    bundle, report, holdout = run(config)
    # Keep what was judged, so the model tests (serving skew, API, the notebook's numbers) run on this exact model.
    save_bundle(bundle, model_path(config))
    out = Path(config["report_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")

    # 1. The goalposts: refuse to judge on a holdout that is not the frozen one.
    expected, actual = config["holdout_sha256"], report["holdout_sha256"]
    if actual != expected:
        logger.error("Holdout fingerprint changed: expected %s, found %s", expected[:12], actual[:12])
        sys.exit(2)

    # 2. The metrics, with their baselines and intervals.
    result = assess(holdout["y"], holdout["p"], holdout["rule"], holdout["pid"], config)
    print_result(result, config, f"the frozen holdout (new patients from order {config['valid_end_q']})")

    # 3. The gate.
    name, threshold = config["gate_metric"], config["gate_threshold"]
    value = result["metrics"][name]["model"]
    if not verdict(result, config):
        print(f"GATE FAIL: {name} {value:.3f} is below {threshold:.3f}")
        sys.exit(1)
    print(f"GATE PASS: {name} {value:.3f} meets {threshold:.3f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
