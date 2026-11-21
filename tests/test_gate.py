"""The quality gate (Diary Entry 2): the metric is computed as the notebook does, and the gate follows config."""
import numpy as np
import pytest

from readmit.config import load_config
from readmit.gate import METRICS, assess, precision_at_capacity, verdict
from readmit.model import precision_at_k

CONFIG = load_config()


def test_precision_at_capacity_matches_the_notebook():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 1000), rng.random(1000)
    share = 25 / 195
    assert precision_at_capacity(y, p, share) == precision_at_k(y, p, int(round(share * 1000)))[0]


def test_weights_of_one_change_nothing():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 780), rng.random(780)
    assert precision_at_capacity(y, p, 0.1, np.ones(780)) == pytest.approx(precision_at_capacity(y, p, 0.1))


def test_gate_settings_are_sane():
    assert CONFIG["gate_metric"] in METRICS
    assert 0 < CONFIG["gate_threshold"] < 1
    assert len(CONFIG["holdout_sha256"]) == 64


def test_gate_passes_a_good_model_and_fails_a_random_one():
    rng = np.random.default_rng(2)
    n = 3000
    y = (rng.random(n) < 0.1).astype(int)
    good = y + rng.random(n)                     # every readmission ranked first
    noise = rng.random(n)
    pids = np.arange(n)
    config = dict(CONFIG, gate_boot=50)
    assert verdict(assess(y, good, noise, pids, config), config)
    assert not verdict(assess(y, noise, noise, pids, config), config)
