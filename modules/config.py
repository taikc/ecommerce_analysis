# modules/config.py
from pathlib import Path

# Base project directory — dynamically resolves regardless of machine
BASE_DIR = Path(__file__).resolve().parent.parent

# Subdirectories
DATA_RAW     = BASE_DIR / "data" / "raw"
DATA_EXPORTS = BASE_DIR / "data" / "exports"
FIGURES      = BASE_DIR / "outputs" / "figures"
REPORTS      = BASE_DIR / "outputs" / "reports"
DB_PATH      = BASE_DIR / "data" / "ecommerce.db"

# Create directories if they don't exist
for path in [DATA_RAW, DATA_EXPORTS, FIGURES, REPORTS]:
    path.mkdir(parents=True, exist_ok=True)
