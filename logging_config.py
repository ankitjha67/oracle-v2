"""
Oracle V2 — Logging Configuration
Structured logging with configurable levels and formatters.
"""

import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    """Configure structured logging for the Oracle engine.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
               Can also be set via ORACLE_LOG_LEVEL env var.
    """
    log_level = level or os.getenv("ORACLE_LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Root oracle logger
    root_logger = logging.getLogger("oracle")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on re-init
    if root_logger.handlers:
        return

    # Console handler with structured format
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)-20s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "sklearn", "xgboost", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
