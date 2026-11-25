"""Paths and tunables. Override the data directory with MODELMATE_DATA_DIR (used by tests)."""
import os
from pathlib import Path

# Avoids a noisy loky warning on Windows machines without `wmic`.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

DATA_DIR = Path(os.environ.get("MODELMATE_DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "modelmate.db"

# Number of random-search iterations used to tune the best candidate model.
TUNE_ITER = int(os.environ.get("MODELMATE_TUNE_ITER", "8"))
# Rows above which model *selection* runs on a stratified sample (final model always uses all rows).
SELECTION_SAMPLE_ROWS = int(os.environ.get("MODELMATE_SELECTION_ROWS", "30000"))

MAX_UPLOAD_MB = 200

for d in (UPLOAD_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
