"""Simple logging setup for the project."""
import logging
from pathlib import Path


def setup_logging(log_path: Path = None):
    log_dir = Path(log_path or Path(__file__).resolve().parents[1] / "models" / "artifacts")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "pipeline.log"

    logger = logging.getLogger("dlrl")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(logfile)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


logger = setup_logging()
