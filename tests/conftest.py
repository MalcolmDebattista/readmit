from pathlib import Path

import pytest

from readmit.config import load_config, model_path, raw_dir
from readmit.model import load_bundle

CONFIG = load_config()
RAW = raw_dir(CONFIG)


@pytest.fixture(scope="session")
def tables():
    """The code tables decoded from the real mapping file (skipped on a fresh clone before readmit-fetch)."""
    from readmit.data import code_tables, read_id_tables
    path = RAW / "IDs_mapping.csv"
    if not path.exists():
        pytest.skip("IDs_mapping.csv not downloaded: run readmit-fetch")
    return code_tables(read_id_tables(path))


@pytest.fixture(scope="session")
def bundle():
    path = Path(model_path(CONFIG))
    if not path.exists():
        pytest.skip("no trained model: run readmit-train")
    return load_bundle(path)
