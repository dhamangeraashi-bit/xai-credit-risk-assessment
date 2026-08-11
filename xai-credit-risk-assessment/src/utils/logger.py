"""
logger.py
=========
Shared logging configuration. Every script logs to both the console and a
per-run file under outputs/logs/, so a reviewer can inspect exactly what
happened during a reproduction run.
"""

import logging
import os
import sys

from src.config import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"{name}.log"), mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
