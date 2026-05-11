"""Project path configuration."""

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def timestamped_run_dir(output_group: Path, now: datetime | None = None) -> Path:
    """Return a unique timestamped run directory below an experiment output group."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run_dir = output_group / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = output_group / f"{timestamp}_{suffix:02d}"
        suffix += 1
    return run_dir
