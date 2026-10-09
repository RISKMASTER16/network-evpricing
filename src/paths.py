from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
INPUT_FIGURES = FIGURES / "inputs"

LOAD_PROFILE_CSV = PROCESSED / "london_ev_load_profile.csv"

for _d in (PROCESSED, RESULTS, FIGURES, INPUT_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
