"""Append-only local review events; path can point to a persistent mounted volume."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import ROOT


def database_path() -> Path:
    return Path(os.getenv("REVIEW_DB_PATH") or ROOT / "runtime" / "reviews.sqlite3")


def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT, account_id TEXT NOT NULL,
        decision TEXT NOT NULL, analyst TEXT NOT NULL, note TEXT NOT NULL,
        reviewed_at TEXT NOT NULL, dataset_id TEXT NOT NULL)""")
    return conn


def save_review(account_id: str, decision: str, analyst: str, note: str,
                dataset_id: str, path: Path | None = None) -> None:
    if decision not in ["Confirmed", "Dismissed", "Needs investigation"]:
        raise ValueError("Unsupported decision")
    if not account_id.strip() or not analyst.strip() or not note.strip() or not dataset_id.strip():
        raise ValueError("Account, analyst, note and dataset version are required")
    if len(analyst) > 80 or len(note) > 1000:
        raise ValueError("Analyst or note exceeds length limit")
    with _connect(path) as conn:
        conn.execute("INSERT INTO reviews (account_id,decision,analyst,note,reviewed_at,dataset_id) VALUES (?,?,?,?,?,?)",
            (account_id, decision, analyst.strip(), note.strip(), datetime.now(timezone.utc).isoformat(), dataset_id))


def review_history(path: Path | None = None, dataset_id: str | None = None) -> pd.DataFrame:
    with _connect(path) as conn:
        if dataset_id:
            return pd.read_sql_query("SELECT * FROM reviews WHERE dataset_id = ? ORDER BY id", conn, params=(dataset_id,))
        return pd.read_sql_query("SELECT * FROM reviews ORDER BY id", conn)


def latest_reviews(path: Path | None = None, dataset_id: str | None = None) -> pd.DataFrame:
    return review_history(path, dataset_id).drop_duplicates(["dataset_id", "account_id"], keep="last")
