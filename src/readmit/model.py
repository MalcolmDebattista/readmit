"""Sections 4-5 of the notebook: split, fit, evaluate, bootstrap, champion choice and the holdout fingerprint."""
import hashlib

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def split(df: pd.DataFrame, train_end_q: float, valid_end_q: float) -> dict:
    """New patients only in each period; returning patients are set aside (Trap 2: split patients, in order)."""
    first_order = df.groupby("pid").order.transform("min")
    period = np.select([df.order < train_end_q, df.order < valid_end_q], ["train", "valid"], "test")
    first_period = np.select([first_order < train_end_q, first_order < valid_end_q], ["train", "valid"], "test")
    part = np.where(period == first_period, period, "returning")
    masks = {p: (part == p) & df.eligible.values for p in ("train", "valid", "test")}
    masks["returning_test"] = (period == "test") & (part == "returning") & df.eligible.values
    return {k: pd.Series(v, index=df.index) for k, v in masks.items()}


def precision_at_k(y_true, score, k):
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    yv = np.asarray(y_true)
    return yv[top].mean(), yv[top].sum() / yv.sum()


def fit(X, y, cols, mask, params):
    return lgb.LGBMClassifier(**params).fit(X.loc[mask, cols], y[mask])


def evaluate(name, y_true, p, call_share, k=None):
    k = int(round(call_share * len(y_true))) if k is None else k
    pk, rk = precision_at_k(y_true, p, k)
    return {"model": name, "AUPRC": average_precision_score(y_true, p), "lift": average_precision_score(y_true, p) / np.mean(y_true),
            "ROC-AUC": roc_auc_score(y_true, p), "precision@capacity": pk, "recall@capacity": rk}


def boot_weights(pids, n, seed):
    """Patient-level bootstrap weights: each resample redraws patients, so a patient's encounters move together."""
    r = np.random.default_rng(seed)
    idx = pd.factorize(pids)[0]; n_pat = idx.max() + 1
    return [r.multinomial(n_pat, np.full(n_pat, 1 / n_pat))[idx] for _ in range(n)]


def choose_champion(y_va, p_val, w_va):
    """The notebook's rule, fixed before looking: take the richer at-discharge model only if its validation gain
    has a bootstrap interval above zero, otherwise the simpler one."""
    a, b = "at discharge + medication classes (KB)", "at discharge"
    diffs = [average_precision_score(y_va, p_val[a], sample_weight=w) - average_precision_score(y_va, p_val[b], sample_weight=w)
             for w in w_va]
    low = float(np.percentile(diffs, 2.5))
    return (a if low > 0 else b), low


def holdout_fingerprint(df, X, te, feats):
    holdout_frame = pd.concat([df.loc[te, ["patient", "order", "readmitted"]], X.loc[te, feats]], axis=1)
    return hashlib.sha256(pd.util.hash_pandas_object(holdout_frame, index=False).values.tobytes()).hexdigest()


def save_bundle(bundle: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path) -> dict:
    return joblib.load(path)
