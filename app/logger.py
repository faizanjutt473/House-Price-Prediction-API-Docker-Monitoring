"""Request logging: one JSON object per line, written to logs/requests.log (rotating) and stdout."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger("requests")
    if logger.handlers:  # already configured (e.g. uvicorn reload)
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "requests.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:  # read-only filesystem etc. -> keep stdout logging
        logger.warning('{"event": "file_logging_disabled", "error": "%s"}', exc)

    return logger
