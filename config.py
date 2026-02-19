"""
config.py - Persistent JSON configuration for HiDock GUI.

Stores config in hidock_config.json next to this script (avoids OneDrive
sync issues that happen with ~/AppData paths on Windows).
"""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "hidock_config.json"

_DEFAULTS = {
    "download_dir": "",
    "transcript_output_dir": "",
    "diarize_enabled": False,
    "show_timecodes": False,
}


def load() -> dict:
    """Load config from disk, merging with defaults. Returns defaults on missing/corrupt file."""
    cfg = dict(_DEFAULTS)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cfg.update(data)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return cfg


def save(cfg: dict):
    """Write config dict to disk as pretty-printed JSON."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
