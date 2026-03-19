# Oracle V2 — Development Guide

## Project Overview
Universal Sports Prediction Engine covering 9 sports: Cricket, Football (6 leagues), NBA, NHL, MLB, NFL, UFC, F1.

## Quick Start
```bash
# Install all dependencies
pip install -e ".[dev]"

# Run all predictions
python run.py

# Run specific sport
python run.py --cricket
python run.py --football
python run.py --multi
```

## Architecture
- **run.py** — Entry point and orchestrator
- **core.py** — Database (SQLite), rating systems (Elo/Glicko-2/TrueSkill), calibration, backtesting
- **engine.py** — Cricket ML engine with 50 features and 12 model ensemble
- **football_pipeline.py** — Football ML with 3-class (W/D/L) prediction + Poisson scoring
- **multi_sport.py** — NBA/NHL/MLB/UFC/F1 via ESPN API + universal Elo
- **sentiment.py** — ESPN DraftKings odds extraction and model-market blending
- **all_apis.py** — 13 API integrations with caching
- **cricsheet_pipeline.py** — Ball-by-ball cricket data parser
- **fixture_fetcher.py** — Live fixture fetching from ESPN
- **env_loader.py** — Environment variable loader

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/test_core.py -v

# Skip slow/integration tests
pytest -m "not slow and not integration"
```

## Code Quality
```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy .
```

## Key Patterns
- **Graceful degradation**: Optional packages (xgboost, lightgbm, glicko2, trueskill) fail silently
- **API caching**: 6-hour TTL file cache in `.cache/` directory
- **Rate limiting**: Token bucket per-API with configurable limits
- **Thread-safe DB**: SQLite with WAL mode and thread-local connections
- **Walk-forward testing**: No lookahead bias in backtesting

## Environment Variables
All optional — system degrades gracefully without them:
- `ODDS_API_KEY` — the-odds-api.com (500 req/month free)
- `FOOTBALL_DATA_KEY` — football-data.org (10 req/min free)
- `NEWSDATA_KEY` — newsdata.io (200 req/day free)
- `ORACLE_LOG_LEVEL` — Logging level (DEBUG, INFO, WARNING, ERROR)

## Data Directories
Auto-created on first run:
- `cricsheet_data/` — T20 ball-by-ball CSV files
- `football_data/` — football-data.co.uk CSVs
- `.cache/` — API response cache

## Output
- `ORACLE_UNIFIED_OUTPUT.json` — Main output with all predictions and ratings
- `oracle.db` — SQLite database with persistent storage
