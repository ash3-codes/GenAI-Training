# app/memory_store.py
import json
import os
from datetime import datetime
from typing import List, Dict


MEMORY_FILE = "data/weekly_memory.json"
SUMMARY_FILE = "data/summary_memory.txt"


def _ensure_files():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            f.write("")


def load_weekly_memory() -> List[Dict]:
    _ensure_files()
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_weekly_entry(week: str, topics: List[str], summary: str):
    _ensure_files()
    memory = load_weekly_memory()
    memory.append(
        {
            "week": week,
            "topics": topics,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    )
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


def load_summary_memory() -> str:
    _ensure_files()
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_summary_memory(text: str):
    _ensure_files()
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(text.strip())


def get_recent_entries(n: int = 6) -> List[Dict]:
    memory = load_weekly_memory()
    return memory[-n:]
