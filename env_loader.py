"""
Oracle V2 — Environment Variable Loader
Loads API keys from .env file (no python-dotenv dependency needed).
"""

import os
from pathlib import Path

_BASE = Path(os.path.dirname(os.path.abspath(__file__)))


def load_env(env_path=None):
    """Load .env file into os.environ. No external deps."""
    if env_path is None:
        env_path = _BASE / ".env"
    if not Path(env_path).exists():
        return {}
    loaded = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if value:  # Only set if non-empty
                os.environ[key] = value
                loaded[key] = value
    return loaded


def get_key(name, default=""):
    """Get an API key from env."""
    return os.environ.get(name, default)


def has_key(name):
    """Check if an API key is set and non-empty."""
    return bool(os.environ.get(name, "").strip())


# Auto-load on import
_loaded = load_env()

# Convenience
ODDS_API_KEY = get_key("ODDS_API_KEY")
FOOTBALL_DATA_KEY = get_key("FOOTBALL_DATA_KEY")
NEWSDATA_KEY = get_key("NEWSDATA_KEY")
