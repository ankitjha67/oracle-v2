# Oracle V2 — Universal Sports Prediction Engine

> Production-grade multi-sport prediction engine with ML ensembles, 4 rating systems, advanced analytics, and market sentiment integration.
> 20+ sports, 300+ live predictions per run, zero hardcoded matches.

## What It Does

```
python run.py
```

Auto-installs dependencies → downloads data → fetches LIVE fixtures from ESPN →
builds ratings from real results → trains ML ensembles → predicts every upcoming
match → blends with DraftKings market sentiment → outputs JSON.

**One command. Zero setup. 300+ predictions in ~80 seconds.**

## Sports Covered

| Category | Sports | Data Source | Model |
|----------|--------|-------------|-------|
| **Football** | EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL, MLS | ESPN + football-data.co.uk | 7 ML models (3-class H/D/A) + Poisson scores |
| **Cricket** | T20I, IPL, BBL, CPL, PSL, SA20, The Hundred | CricSheet ball-by-ball | 12 ML models + 56 features + 4 rating systems |
| **US Majors** | NBA, NHL, MLB, NFL, WNBA | ESPN API | 12 ML models + Elo + roster tracking |
| **College** | NCAAF, NCAAM, NCAAW | ESPN API | 12 ML models + Elo |
| **Combat** | UFC | ESPN API | Fighter Elo + records + ML ensemble |
| **Tennis** | ATP, WTA | ESPN API | Player Elo + ML ensemble |
| **International** | Rugby Union, Rugby League, AFL, Field Hockey, Lacrosse | ESPN API | Elo + ML ensemble |
| **Motorsport** | Formula 1 | OpenF1 API | Driver grid + team data |
| **Golf** | PGA Tour | ESPN API | Player Elo |

## ML Architecture

### Cricket Engine (`engine.py`)
- **56 features** including form velocity, volatility, interaction terms, and player-level stats
- **12-model ensemble**: RandomForest, GradientBoosting, LogisticRegression, XGBoost, LightGBM, MLP, AdaBoost, SVM, Bagging, NaiveBayes, VotingClassifier, StackingMeta
- **Model disagreement confidence** — ensemble std dev penalizes uncertain predictions
- **4 rating systems**: Elo, Glicko-2, TrueSkill, custom composite

### Multi-Sport ML Engine (`ml_sports.py`)
- **35 features per matchup**: Elo ratings, rolling win rates, margin of victory, streak momentum, rest days, home/away splits, H2H record, strength of schedule, consistency, form velocity, roster stability
- **Up to 12 models** matching the cricket engine coverage
- **RosterTracker**: auto-fetches rosters from ESPN, detects changes, computes stability signals (new player count, avg experience, injury count, turnover rate)

### Football Pipeline (`football_pipeline.py`)
- **7 ML models** with 3-class (W/D/L) prediction
- **Poisson goal scoring** model for correct-score predictions
- **ROI backtesting** against bookmaker closing odds

## Fan Sentiment Integration

Every prediction is blended with **real betting market data**:

```
Final Prediction = 65% Model (Elo/ML) + 35% Market Sentiment (DraftKings)
```

Sentiment is extracted from ESPN's embedded DraftKings odds:
- **Spread** → converted to implied win probability
- **Over/Under** → total scoring expectation
- **Line movement** → smart money direction (HOME_SHIFT / AWAY_SHIFT / STABLE)
- **Favorite at open vs current** → where the market moved

When model and market agree → **HIGH confidence**.
When they disagree → flagged as **CONTRARIAN** pick (potential value bet).

## Quick Start

```bash
git clone https://github.com/ankitjha67/oracle-v2.git
cd oracle-v2
python run.py
```

That's it. The engine auto-installs all dependencies on first run.

### Install via pip

```bash
pip install -e ".[dev]"
```

### CLI Options

```bash
python run.py              # Everything (cricket + football + all ESPN sports)
python run.py --cricket    # Cricket only (~30s)
python run.py --football   # Football (6 leagues) only (~15s)
python run.py --multi      # NBA/NHL/MLB/NFL/UFC/Tennis/College/etc. (~25s)
```

## REST API

```bash
pip install fastapi uvicorn
python api_server.py                    # Start on port 8000
uvicorn api_server:app --reload         # Development mode
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with DB stats |
| `GET` | `/sports` | List supported sports |
| `GET` | `/ratings/{sport}` | Team/player Elo rankings |
| `POST` | `/predict/cricket` | Cricket match prediction (12 models) |
| `POST` | `/predict/football` | Football match prediction (7 models) |
| `GET` | `/predictions/recent` | Recent predictions with filters |
| `GET` | `/h2h/{team_a}/{team_b}` | Head-to-head analysis |
| `GET` | `/audit/{sport}` | Bias audit on predictions |
| `GET` | `/explain/{prediction_id}` | SHAP-based prediction explanation |

### Analytics Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/analytics/evaluation/{sport}` | Brier score, log loss, ROC-AUC, calibration |
| `GET` | `/analytics/clv/{sport}` | Closing Line Value summary |
| `POST` | `/analytics/ev` | Expected Value + Kelly Criterion calculator |
| `GET` | `/analytics/calibration/{sport}` | Reliability diagram data |
| `GET` | `/analytics/ratings/history/{team}` | Rating time series |
| `GET` | `/analytics/changepoint/{team}` | CUSUM changepoint detection |
| `GET` | `/analytics/market-efficiency` | Where Oracle has edge vs. market |
| `POST` | `/analytics/season-simulation` | Monte Carlo season projections |
| `POST` | `/analytics/playoff-simulation` | Seeded bracket simulation |
| `GET` | `/analytics/rolling-performance` | Rolling accuracy & Brier score |
| `GET` | `/analytics/degradation` | Model accuracy degradation check |
| `GET` | `/analytics/edge-niches` | Most profitable sport × confidence niches |
| `GET` | `/analytics/prediction-evolution/{match_id}` | Probability timeline for a match |
| `GET` | `/docs` | Interactive OpenAPI docs |

## Architecture

```
run.py                    Entry point, auto-installer, orchestrator
├── core.py               SQLite DB, 4 rating systems, calibration, backtest, Monte Carlo
├── engine.py             Cricket ML (56 features, 12 models, stacking meta-learner)
├── ml_sports.py          Universal ML engine (35 features, 12 models, roster tracking)
├── cricsheet_pipeline.py CricSheet ball-by-ball parser (T20I + 6 domestic leagues)
├── football_pipeline.py  Football ML (32 features, 7 models, Poisson scores, ROI)
├── fixture_fetcher.py    Live ESPN fixtures for 6 football leagues
├── multi_sport.py        20+ sports via ESPN API (universal Elo + dynamic K-factors)
├── analytics.py          CLV, EV, Kelly, season/playoff sims, SHAP, changepoints
├── sentiment.py          Fan sentiment from ESPN DraftKings odds
├── all_apis.py           13 API integrations (weather, odds, news, geocoding)
├── api_server.py         FastAPI REST API (25+ endpoints)
├── logging_config.py     Structured logging configuration
├── env_loader.py         .env file loader (no external deps)
└── tests/                Pytest suite with unit + integration tests
```

**11,500+ lines of production Python. Zero hardcoded matches. Everything is LIVE.**

## Analytics Engine

Oracle V2 includes a comprehensive analytics module (`analytics.py`):

- **CLVTracker** — Closing Line Value, the gold standard metric for prediction edge
- **EVCalculator** — Expected Value computation for any bet
- **KellyStaker** — Kelly Criterion stake sizing (quarter/half Kelly)
- **BankrollSimulator** — Monte Carlo bankroll growth simulation
- **EvaluationSuite** — Brier score, BSS, log loss, ROC-AUC, calibration curves
- **RatingChangePointDetector** — CUSUM detection of rating trajectory shifts
- **SeasonSimulator** — Monte Carlo season projections with league-specific rules
- **PlayoffCalculator** — Seeded bracket simulation (single elim or best-of-N)
- **MarketEfficiencyMonitor** — Rolling performance tracking, degradation alerts, edge niches
- **PredictionTimeSeries** — Probability evolution tracking and drift detection

## API Keys (Optional)

The engine works without any keys. Add keys for enhanced data:

```bash
cp .env.example .env
# Edit .env with your keys
```

| Key | Source | Free Tier | What It Adds |
|-----|--------|-----------|-------------|
| `ODDS_API_KEY` | [the-odds-api.com](https://the-odds-api.com) | 500 req/month | 40+ bookmaker odds |
| `FOOTBALL_DATA_KEY` | [football-data.org](https://www.football-data.org/client/register) | 10 req/min | Extra fixture data |
| `NEWSDATA_KEY` | [newsdata.io](https://newsdata.io/register) | 200 req/day | Injury/team news |

Set `ORACLE_LOG_LEVEL` to `DEBUG`, `INFO`, `WARNING`, or `ERROR` to control logging verbosity.

## Testing

```bash
pytest                                    # Run all tests
pytest --cov=. --cov-report=term-missing  # With coverage
pytest tests/test_core.py -v              # Specific file
pytest -m "not slow and not integration"  # Skip slow/network tests
```

## Code Quality

```bash
ruff check .   # Lint
ruff format .  # Format
mypy .         # Type check
```

## Adding a New Sport

```python
# 1. Add to SPORTS dict in multi_sport.py
"NEW_SPORT": {"espn": "sport/league", "K": 25, "home": 40, "type": "team"}

# 2. Add to season detection in get_active_sports()
# 3. That's it — build_team_sport() + SportMLEngine handle everything
```

The ML engine (`ml_sports.py`) automatically builds features and trains models for any sport registered in the `SPORTS` dict.

## Data Directories

Auto-created on first run:
- `cricsheet_data/` — T20I ball-by-ball CSV files
- `cricsheet_{league}/` — Domestic T20 league data (IPL, BBL, CPL, PSL, SA20, The Hundred)
- `football_data/` — football-data.co.uk CSVs
- `.cache/` — API response cache (ESPN, OpenF1, weather, odds)

## Output

Every run produces `ORACLE_UNIFIED_OUTPUT.json` containing:
- All predictions with probabilities, confidence levels, and sentiment data
- Elo rankings for every team/player in every sport
- Model training metrics and backtest results
- Poisson score predictions for football
- ROI backtest results
- UCL two-leg aggregate tracking
- UFC fighter records
- F1 driver grid

## Responsible Gambling

This tool is for **informational and entertainment purposes only**. Past performance
does not guarantee future results. Never bet more than you can afford to lose.

- USA: 1-800-522-4700 (NCPG)
- UK: 0808-8020-133 (GamCare)
- India: iCall — 9152987821
- International: www.begambleaware.org

## License

MIT
