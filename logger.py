"""
logger.py
=========
Lightweight prediction logger.
Writes each prediction to a session-based CSV log and keeps an
in-memory history list that the Streamlit UI reads directly.
"""

import os
import csv
import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_log.csv")

COLUMNS = ["timestamp", "url", "prediction", "confidence_phishing", "confidence_safe"]


def _ensure_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def log_prediction(url: str, label: str, conf_phishing: float, conf_safe: float):
    """Append one prediction row to the CSV log."""
    _ensure_log()
    row = {
        "timestamp":           datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url":                 url,
        "prediction":          label,
        "confidence_phishing": round(conf_phishing, 4),
        "confidence_safe":     round(conf_safe, 4),
    }
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)
    return row


def load_log() -> list:
    """Return all logged predictions as a list of dicts (newest first)."""
    _ensure_log()
    with open(LOG_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return list(reversed(rows))


def clear_log():
    """Wipe the log file."""
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
