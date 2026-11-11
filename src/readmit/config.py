import logging
import os
import sys
from pathlib import Path

import yaml


def load_config(path=None):
    path = path or os.environ.get("CONFIG_PATH", "config/config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(config) -> Path:
    """DATASETS_TASKS_DATA (the notebook's variable) overrides config.yaml."""
    return Path(os.environ.get("DATASETS_TASKS_DATA", config["data_dir"]))


def raw_dir(config) -> Path:
    return data_dir(config) / "raw" / "diabetes130"


def processed_dir(config) -> Path:
    return data_dir(config) / "processed"


def model_path(config) -> Path:
    return Path(os.environ.get("MODEL_PATH", config["model_path"]))


def setup_logging(level=None):
    logging.basicConfig(level=level or os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
