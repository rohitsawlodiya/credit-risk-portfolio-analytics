import json
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def portfolio():
    return pd.read_csv(ROOT / "data/processed/portfolio.csv", dtype={"origination_year": str})


@pytest.fixture(scope="session")
def report():
    return json.loads((ROOT / "artifacts/report.json").read_text())
