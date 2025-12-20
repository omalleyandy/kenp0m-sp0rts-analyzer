"""
Structured logging configuration for the application
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_dir: Path, level: str = "INFO", app_name: str = "ncaab-xgboost"
) -> logging.Logger:
    """
    Setup structured logging for the application

    Args:
        log_dir: Directory to store log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        app_name: Application name for log files

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, level))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    log_file = log_dir / f"{app_name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Separate error log
    error_log = log_dir / f"{app_name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


# Global logger instance
logger = setup_logging(Path("logs"))
