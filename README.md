# Oracle V2 — Universal Sports Prediction Engine

> Production-grade multi-sport prediction engine with fan sentiment integration.
> 9 sports, 300+ live predictions per run, zero hardcoded matches.

## What It Does

```
python run.py
```

Auto-installs dependencies → downloads data → fetches LIVE fixtures from ESPN →
builds Elo ratings from real results → trains ML models → predicts every upcoming
match → blends with DraftKings market sentiment → outputs JSON.

**One command. Zero setup. 300+ predictions in ~80 seconds.**

## Sports Covered

| Sport | Source | Matches/Week | Model |
|-------|--------|-------------|-------|
| ⚽ Football (EPL, La Liga, Serie A, Bundesliga, Ligue 1, UCL) | ESPN + football-data.co.uk | 30-40 | 6 ML models (3-class H/D/A) + Poisson scores |
| 🏏 Cricket | CricSheet ball-by-ball | Varies | 11 ML models + 50 features + 4 rating systems |
| 🏀 NBA | ESPN | 60+ | Elo + home advantage + margin-of-victory |
| 🏒 NHL | ESPN | 60+ | Elo + home advantage |
| ⚾ MLB | ESPN | 100+ | Elo + home advantage |
| 🥊 UFC | ESPN | 10-15 per card | Fighter Elo + records |
| 🏎️ F1 | OpenF1 | Per race weekend | Driver grid + team data |

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

### CLI Options

```bash
python run.py              # Everything (cricket + football + NBA/NHL/MLB/UFC/F1)
python run.py --cricket    # Cricket only (~30s)
python run.py --football   # Football (6 leagues) only (~15s)
python run.py --multi      # NBA/NHL/MLB/UFC/F1 only (~25s)
```

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

## Architecture

```
run.py                    Entry point, auto-installer, orchestrator
├── core.py               SQLite DB, 4 rating systems, calibration, backtest, Monte Carlo
├── engine.py             Cricket ML (50 features, 11 models, player database)
├── cricsheet_pipeline.py CricSheet ball-by-ball parser (2000+ T20 matches)
├── football_pipeline.py  Football ML (32 features, 6 models, Poisson scores, ROI)
├── fixture_fetcher.py    Live ESPN fixtures for 6 football leagues
├── multi_sport.py        NBA/NHL/MLB/UFC/F1 (universal Elo + ESPN)
├── sentiment.py          Fan sentiment from ESPN DraftKings odds
├── all_apis.py           13 API integrations (weather, odds, news, geocoding)
├── env_loader.py         .env file loader (no external deps)
├── .env.example          API key template
└── requirements.txt      Python dependencies
```

**5,700+ lines of production Python. Zero hardcoded matches. Everything is LIVE.**

## Output

Every run produces `ORACLE_UNIFIED_OUTPUT.json` (~700 KB) containing:
- All predictions with probabilities, confidence levels, and sentiment data
- Elo rankings for every team/club in every sport
- Model training metrics and backtest results
- Poisson score predictions for football
- ROI backtest results (+2.1% edge over bookmaker closing odds)
- UCL two-leg aggregate tracking
- UFC fighter records
- F1 driver grid

## Responsible Gambling

This tool is for **informational and entertainment purposes only**. Past performance
does not guarantee future results. Never bet more than you can afford to lose.

- 🇺🇸 USA: 1-800-522-4700 (NCPG)
- 🇬🇧 UK: 0808-8020-133 (GamCare)
- 🇮🇳 India: iCall — 9152987821
- 🌐 International: www.begambleaware.org

## License

MIT
