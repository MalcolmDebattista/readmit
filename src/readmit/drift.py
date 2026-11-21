"""readmit-drift: the monitoring plan's evidence (Diary Entry 2, Sessions 04-06).

There is no calendar, so the time axis is the encounter order. Everything after the training period (order 0.70 to
1.00) is cut into slices of drift_slice (5 %), and for each slice, against the training period:

- PSI of the score, and of the champion's most important inputs (numeric: 10 quantile bins; categories: their values);
- the model's metrics on that slice: precision at capacity (the gate metric), AUPRC, observed / expected.

PSI and psi_cat are the notebook's (Section 7). Output: reports/drift.csv and, with matplotlib, reports/drift.png.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score

from readmit.config import load_config, setup_logging
from readmit.features import ladder
from readmit.gate import precision_at_capacity
from readmit.model import fit, split

logger = logging.getLogger(__name__)


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def psi_cat(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.value_counts(normalize=True); c = cur.value_counts(normalize=True)
    idx = r.index.union(c.index); r = r.reindex(idx, fill_value=0) + 1e-6; c = c.reindex(idx, fill_value=0) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


def feature_psi(ref: pd.Series, cur: pd.Series) -> float:
    if str(ref.dtype) == "category" or ref.dtype == object:
        return psi_cat(ref.astype(str), cur.astype(str))
    return psi(ref.values.astype(float), cur.values.astype(float))


def drift_table(df, X, config, top=4):
    """One row per slice of the order after training: PSI of the score and top inputs, and the metrics."""
    m = split(df, config["train_end_q"], config["valid_end_q"])
    tr = m["train"]
    params = dict(config["lgb_params"], random_state=config["seed"], n_jobs=config["n_jobs"])
    feats = ladder(X)["at discharge"]
    model = fit(X, df.y, feats, tr, params)
    gain = pd.Series(model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
    watched = list(gain.index[:top])
    share = config["calls_per_week"] / config["discharges_per_week"]

    p_tr = model.predict_proba(X.loc[tr, feats])[:, 1]
    # every eligible discharge after the training period, as the deployed model would have scored it
    post = df.eligible & (df.order >= config["train_end_q"])
    p_post = model.predict_proba(X.loc[post, feats])[:, 1]
    step = config["drift_slice"]
    slice_id = ((df.order[post] - config["train_end_q"]) // step).astype(int).values
    rows = []
    for i in sorted(set(slice_id)):
        k = slice_id == i
        idx = df.index[post][k]
        y, p = df.y[idx].values, p_post[k]
        row = {"slice": f"{config['train_end_q'] + step * i:.2f}-{config['train_end_q'] + step * (i + 1):.2f}",
               "start": config["train_end_q"] + step * i, "discharges": int(k.sum()),
               "PSI score": psi(p_tr, p)}
        for f in watched:
            row[f"PSI {f}"] = feature_psi(X.loc[tr, f], X.loc[idx, f])
        row |= {"precision@capacity": precision_at_capacity(y, p, share), "AUPRC": float(average_precision_score(y, p)),
                "base rate": float(y.mean()), "observed / expected": float(y.mean() / p.mean())}
        rows.append(row)
    return pd.DataFrame(rows), watched, gain


def plot(table, watched, config, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = table.start + config["drift_slice"] / 2
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for col, style in zip(["PSI score"] + [f"PSI {f}" for f in watched], ["-", "--", ":", "-.", (0, (5, 2, 1, 2))]):
        ax.plot(x, table[col], linestyle=style, marker="o", ms=4, lw=2 if col == "PSI score" else 1.4, label=col)
    ax.axhline(0.10, color="grey", lw=0.8); ax.text(x.iloc[0], 0.105, "PSI 0.10", fontsize=8, color="grey")
    ax.axhline(0.25, color="grey", lw=0.8, ls="--"); ax.text(x.iloc[0], 0.255, "PSI 0.25", fontsize=8, color="grey")
    ax.set_xlabel("encounter order (proxy for time; training ends at 0.70, holdout starts at 0.85)")
    ax.set_ylabel("PSI against the training period")
    ax.axvline(config["valid_end_q"], color="black", lw=0.6)
    ax2 = ax.twinx()
    # All eligible discharges in the slice, returning patients included, so not comparable with the gate (new patients)
    ax2.plot(x, table["precision@capacity"], color="black", lw=2.2, marker="s",
             label="precision at capacity, all eligible discharges (right)")
    ax2.set_ylabel("precision at capacity")
    ax2.set_ylim(0, max(0.35, table["precision@capacity"].max() * 1.15))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", bbox_to_anchor=(1.08, 1), frameon=False)
    ax.set_title("Input drift and model metric along the encounter order", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    load_dotenv()
    setup_logging()
    config = load_config()
    from readmit.train import load_all
    df, X, *_ = load_all(config)
    table, watched, gain = drift_table(df, X, config)
    out = Path(config["report_path"]).parent; out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "drift.csv", index=False)
    (out / "drift_features.json").write_text(json.dumps({"watched": watched, "gain": gain.round(1).to_dict()}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table.drop(columns="start").to_string(index=False))
    try:
        plot(table, watched, config, out / "drift.png")
        logger.info("Figure written to %s", out / "drift.png")
    except ImportError:
        logger.warning("matplotlib not installed: no figure (the table is in %s)", out / "drift.csv")


if __name__ == "__main__":
    main()
