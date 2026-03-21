# Oracle V2 — Development Guide

## Project Overview
Universal Sports Prediction Engine covering 20+ sports across ESPN, CricSheet, and OpenF1 data sources. Features ML ensembles (up to 12 models), 4 rating systems, advanced analytics, roster tracking, and market sentiment integration.

## Sports Coverage
| Category | Sports | Data Source |
|----------|--------|-------------|
| **Cricket** | T20I, IPL, BBL, CPL, PSL, SA20, The Hundred | CricSheet ball-by-ball |
| **Football** | EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL, MLS | ESPN + football-data.co.uk |
| **US Majors** | NBA, NHL, MLB, NFL, WNBA | ESPN API |
| **College** | NCAAF, NCAAM (Men's BBall), NCAAW (Women's BBall) | ESPN API |
| **Combat** | UFC | ESPN API (fighter-level Elo) |
| **Individual** | ATP Tennis, WTA Tennis, Golf | ESPN API |
| **International** | Rugby Union, Rugby League, AFL, Field Hockey, Lacrosse | ESPN API |
| **Motorsport** | Formula 1 | OpenF1 API |

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

# Start REST API server
pip install fastapi uvicorn
python api_server.py
```

## Architecture
- **run.py** — Entry point and orchestrator
- **core.py** — Database (SQLite), 4 rating systems (Elo/Glicko-2/TrueSkill/composite), calibration, backtesting, Monte Carlo
- **engine.py** — Cricket ML engine with 56 features, 12-model ensemble + stacking meta-learner
- **ml_sports.py** — Universal ML engine for all ESPN sports (35 features, 12 models, roster tracking)
- **football_pipeline.py** — Football ML with 3-class (W/D/L) prediction + Poisson scoring
- **multi_sport.py** — 20+ sports via ESPN API + universal Elo with dynamic K-factors
- **analytics.py** — CLV tracking, EV calculator, Kelly Criterion, season/playoff sims, SHAP, changepoints, market efficiency
- **sentiment.py** — ESPN DraftKings odds extraction and model-market blending
- **all_apis.py** — 13 API integrations with caching
- **cricsheet_pipeline.py** — Ball-by-ball cricket data parser + domestic T20 league support
- **fixture_fetcher.py** — Live fixture fetching from ESPN
- **api_server.py** — FastAPI REST API server with 25+ endpoints
- **logging_config.py** — Structured logging configuration
- **env_loader.py** — Environment variable loader

## Key Features
- **56 ML features** (cricket) / **35 ML features** (ESPN sports) including form velocity, volatility, and interaction terms
- **12-model ensembles** — RF, GBM, LR, XGBoost, LightGBM, MLP, AdaBoost, SVM, Bagging, NaiveBayes, VotingClassifier, StackingMeta
- **Model disagreement confidence** — ensemble std dev penalizes uncertain predictions
- **RosterTracker** — auto-fetches ESPN rosters, detects changes, computes stability signals
- **Dynamic K-factor** — stage-dependent Elo updates (group=20, final=40)
- **Enhanced H2H** — trend detection, streak tracking, momentum signals
- **BiasAuditor** — systematic bias detection in prediction pipeline
- **Analytics suite** — CLV, EV, Kelly, season sims, playoff brackets, SHAP, changepoint detection, market efficiency
- **Domestic cricket leagues** — IPL, BBL, CPL, PSL, SA20, The Hundred via CricSheet

## Testing
```bash
pytest                                    # Run all tests
pytest --cov=. --cov-report=term-missing  # With coverage
pytest tests/test_core.py -v              # Specific file
pytest -m "not slow and not integration"  # Skip slow tests
```

## Code Quality
```bash
ruff check .   # Lint
ruff format .  # Format
mypy .         # Type check
```

## REST API
```bash
python api_server.py  # Start on port 8000
# Core Endpoints:
#   GET  /health                        — Health check
#   GET  /sports                        — List supported sports
#   GET  /ratings/{sport}               — Get team ratings
#   POST /predict/cricket               — Cricket match prediction
#   POST /predict/football              — Football match prediction
#   GET  /predictions/recent            — Recent predictions
#   GET  /h2h/{team_a}/{team_b}         — Head-to-head analysis
#   GET  /explain/{prediction_id}       — SHAP-based explanation
#   GET  /docs                          — Interactive API docs
# Analytics Endpoints:
#   GET  /analytics/evaluation/{sport}  — Brier, log loss, ROC-AUC
#   GET  /analytics/clv/{sport}         — Closing Line Value
#   POST /analytics/ev                  — Expected Value + Kelly
#   POST /analytics/season-simulation   — Monte Carlo season projection
#   POST /analytics/playoff-simulation  — Bracket simulation
#   GET  /analytics/rolling-performance — Rolling accuracy
#   GET  /analytics/degradation         — Model degradation check
#   GET  /analytics/edge-niches         — Profitable niches
```

## Adding a New Sport
```python
# 1. Add to SPORTS dict in multi_sport.py
"NEW_SPORT": {"espn": "sport/league", "K": 25, "home": 40, "type": "team"}

# 2. Add to season detection in get_active_sports()
# 3. That's it — build_team_sport() + SportMLEngine handle everything
```

## Environment Variables
All optional — system degrades gracefully without them:
- `ODDS_API_KEY` — the-odds-api.com (500 req/month free)
- `FOOTBALL_DATA_KEY` — football-data.org (10 req/min free)
- `NEWSDATA_KEY` — newsdata.io (200 req/day free)
- `ORACLE_LOG_LEVEL` — Logging level (DEBUG, INFO, WARNING, ERROR)

## Data Directories
Auto-created on first run:
- `cricsheet_data/` — T20I ball-by-ball CSV files
- `cricsheet_{league}/` — Domestic T20 league data (IPL, BBL, etc.)
- `football_data/` — football-data.co.uk CSVs
- `.cache/` — API response cache

## Output
- `ORACLE_UNIFIED_OUTPUT.json` — Main output with all predictions and ratings
- `oracle.db` — SQLite database with persistent storage
