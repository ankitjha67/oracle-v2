"""
Oracle V2 — REST API Server
Serves predictions via HTTP endpoints using FastAPI.

Usage:
    pip install fastapi uvicorn
    python api_server.py                    # Start on port 8000
    uvicorn api_server:app --reload         # Development mode

Endpoints:
    GET  /health              — Health check
    GET  /sports              — List supported sports
    GET  /ratings/{sport}     — Get team ratings
    POST /predict/cricket     — Cricket match prediction
    POST /predict/football    — Football match prediction
    GET  /predictions/recent  — Recent predictions
    GET  /backtest/{sport}    — Backtest results
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("oracle.api_server")

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError(
        "FastAPI not installed. Run: pip install fastapi uvicorn\n"
        "Then: python api_server.py"
    )

from core import OracleDB, RatingEngine, BiasAuditor
from analytics import (AnalyticsEngine, EVCalculator, KellyStaker,
                       EvaluationSuite, RatingChangePointDetector,
                       SeasonSimulator, PlayoffCalculator,
                       MarketEfficiencyMonitor, PredictionTimeSeries,
                       LEAGUE_RULES)

# ── Pydantic models ────────────────────────────────────────────────────────

class CricketPredictionRequest(BaseModel):
    team_a: str = Field(..., examples=["India"])
    team_b: str = Field(..., examples=["New Zealand"])
    venue: str = Field(default="", examples=["Mumbai"])
    stage: str = Field(default="group", examples=["final"])
    date: str = Field(default="", examples=["2026-03-08"])
    is_day_night: bool = False
    is_neutral: bool = False
    toss_winner: str = ""
    toss_decision: str = ""
    odds_prob_a: float = Field(default=0.5, ge=0, le=1)
    odds_prob_b: float = Field(default=0.5, ge=0, le=1)


class FootballPredictionRequest(BaseModel):
    home: str = Field(..., examples=["Arsenal"])
    away: str = Field(..., examples=["Chelsea"])
    league: str = Field(default="EPL", examples=["EPL"])
    date: str = Field(default="")


class PredictionResponse(BaseModel):
    team_a: str
    team_b: str
    predicted_winner: str
    confidence: str
    prob_a: float
    prob_b: float
    model_agreement: dict = {}
    feature_importance: dict = {}
    weather: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    db_stats: dict
    models_loaded: bool


# ── App setup ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Oracle V2 — Sports Prediction API",
    description="Universal Sports Prediction Engine with ML models, Elo ratings, and market sentiment.",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────────────────────

OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
db = OracleDB(str(OUTPUT_DIR / "oracle.db"))
ratings = RatingEngine(db)
analytics = AnalyticsEngine(db)

# Lazy-load prediction engine
_oracle = None
_football_models = None


def get_oracle():
    """Lazy-load the cricket prediction engine."""
    global _oracle
    if _oracle is None:
        try:
            from engine import OracleV2
            _oracle = OracleV2()
            logger.info("Oracle engine initialized")
        except Exception as e:
            logger.error(f"Failed to load Oracle engine: {e}")
    return _oracle


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check with database stats."""
    stats = db.get_stats()
    oracle = get_oracle()
    return HealthResponse(
        status="healthy",
        version="3.1.0",
        db_stats=stats,
        models_loaded=oracle is not None and oracle.is_trained,
    )


@app.get("/sports")
def list_sports():
    """List all supported sports and their configuration."""
    return {
        "sports": [
            {"key": "cricket", "name": "Cricket (T20)", "models": 12, "features": 56},
            {"key": "football", "name": "Football (6 leagues)", "models": 7, "features": 32},
            {"key": "nba", "name": "NBA", "rating": "Elo"},
            {"key": "nhl", "name": "NHL", "rating": "Elo"},
            {"key": "mlb", "name": "MLB", "rating": "Elo"},
            {"key": "nfl", "name": "NFL", "rating": "Elo"},
            {"key": "ufc", "name": "UFC", "rating": "Fighter Elo"},
            {"key": "f1", "name": "Formula 1", "rating": "Driver Elo"},
        ]
    }


@app.get("/ratings/{sport}")
def get_ratings(sport: str, top_n: int = 30):
    """Get team/player Elo ratings for a sport."""
    conn = db._get_conn()
    rows = conn.execute(
        "SELECT team, rating FROM team_ratings WHERE sport=? AND rating_type='elo' "
        "ORDER BY rating DESC LIMIT ?",
        (sport, top_n)
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No ratings found for sport: {sport}")
    return {
        "sport": sport,
        "rankings": [{"team": r["team"], "rating": round(r["rating"], 1)} for r in rows],
    }


@app.post("/predict/cricket", response_model=PredictionResponse)
def predict_cricket(req: CricketPredictionRequest):
    """Generate a cricket match prediction using 12 ML models."""
    oracle = get_oracle()
    if oracle is None or not oracle.is_trained:
        raise HTTPException(status_code=503, detail="Prediction engine not trained. Run 'python run.py --cricket' first.")

    match = {
        "team_a": req.team_a,
        "team_b": req.team_b,
        "venue": req.venue,
        "stage": req.stage,
        "sport": "cricket",
        "date": req.date,
        "is_day_night": req.is_day_night,
        "is_neutral": req.is_neutral,
        "toss_winner": req.toss_winner,
        "toss_decision": req.toss_decision,
        "odds_prob_a": req.odds_prob_a,
        "odds_prob_b": req.odds_prob_b,
    }

    result = oracle.predict(match)
    return PredictionResponse(
        team_a=result["team_a"],
        team_b=result["team_b"],
        predicted_winner=result["predicted_winner"],
        confidence=result["confidence"],
        prob_a=result["calibrated_prob_a"],
        prob_b=result["calibrated_prob_b"],
        model_agreement=result.get("model_agreement", {}),
        feature_importance=result.get("feature_importance", {}),
        weather=result.get("weather"),
    )


@app.get("/predictions/recent")
def recent_predictions(sport: str = "", limit: int = 20):
    """Get recent predictions from the database."""
    conn = db._get_conn()
    q = "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?"
    params = [limit]
    if sport:
        q = "SELECT * FROM predictions WHERE sport=? ORDER BY created_at DESC LIMIT ?"
        params = [sport, limit]
    rows = conn.execute(q, params).fetchall()
    return {"predictions": [dict(r) for r in rows]}


@app.get("/h2h/{team_a}/{team_b}")
def head_to_head(team_a: str, team_b: str, sport: str = "cricket"):
    """Get head-to-head record with trend analysis."""
    h2h = db.get_h2h(team_a, team_b, sport)
    return {"team_a": team_a, "team_b": team_b, "sport": sport, **h2h}


@app.get("/audit/{sport}")
def bias_audit(sport: str = "cricket"):
    """Run bias audit on predictions for a sport."""
    acc = db.get_prediction_accuracy(sport, n=200)
    return {
        "sport": sport,
        "accuracy": acc,
    }


# ── Analytics Endpoints ────────────────────────────────────────────────────

@app.get("/analytics/evaluation/{sport}")
def analytics_evaluation(sport: str):
    """Comprehensive evaluation report: Brier, BSS, log loss, ROC-AUC, calibration."""
    report = analytics.evaluation_report(sport=sport)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report


@app.get("/analytics/evaluation")
def analytics_evaluation_all():
    """Evaluation report across all sports."""
    return analytics.evaluation_report()


@app.get("/analytics/clv/{sport}")
def analytics_clv(sport: str):
    """Closing Line Value summary for a sport."""
    return analytics.clv.summary(sport=sport)


@app.get("/analytics/clv")
def analytics_clv_all():
    """CLV summary across all sports."""
    return analytics.clv.summary()


@app.post("/analytics/ev")
def analytics_ev(model_prob: float, decimal_odds: float):
    """Calculate Expected Value and Kelly fraction for a bet."""
    ev = EVCalculator.calculate_ev(model_prob, decimal_odds)
    kelly = KellyStaker.kelly_fraction(model_prob, decimal_odds)
    return {**ev, "kelly_quarter": kelly, "kelly_half": KellyStaker.kelly_fraction(model_prob, decimal_odds, 0.5)}


@app.get("/analytics/calibration/{sport}")
def analytics_calibration(sport: str, n_bins: int = 10):
    """Reliability diagram data for a sport."""
    conn = db._get_conn()
    q = "SELECT prob_a, is_correct FROM predictions WHERE is_correct >= 0 AND sport=?"
    rows = conn.execute(q, (sport,)).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No scored predictions for {sport}")
    import numpy as np
    predicted = np.array([r["prob_a"] for r in rows])
    actual = np.array([r["is_correct"] for r in rows])
    return EvaluationSuite.calibration_data(predicted, actual, n_bins)


@app.get("/explain/{prediction_id}")
def explain_prediction(prediction_id: str):
    """Get SHAP-based explanation for a specific prediction."""
    conn = db._get_conn()
    row = conn.execute(
        "SELECT * FROM predictions WHERE id=?", (prediction_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    pred = dict(row)
    import json
    features = json.loads(pred.get("features_json", "{}"))
    model_votes = json.loads(pred.get("model_votes_json", "{}"))

    return {
        "prediction_id": prediction_id,
        "team_a": pred["team_a"],
        "team_b": pred["team_b"],
        "prob_a": pred["prob_a"],
        "confidence": pred["confidence"],
        "features": features,
        "model_votes": model_votes,
        "note": "For per-prediction SHAP values, use the engine.predict() output which includes shap_explanation",
    }


@app.get("/analytics/ratings/history/{team}")
def rating_history(team: str, sport: str = "cricket", n: int = 100):
    """Get rating history time series for a team."""
    cpd = RatingChangePointDetector(db)
    history = cpd.get_history(team, sport, n=n)
    if not history:
        raise HTTPException(status_code=404, detail=f"No rating history for {team} in {sport}")
    return {"team": team, "sport": sport, "history": history}


@app.get("/analytics/changepoint/{team}")
def detect_changepoint(team: str, sport: str = "cricket",
                       threshold: float = 2.0):
    """Run CUSUM changepoint detection on a team's rating trajectory."""
    cpd = RatingChangePointDetector(db)
    result = cpd.detect(team, sport, threshold=threshold)
    return result


@app.get("/analytics/market-efficiency")
def analytics_market_efficiency():
    """Market efficiency analysis — where does Oracle have edge?"""
    conn = db._get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE is_correct >= 0 AND odds_json != '{}'"
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions with odds data")
    # Build bet records
    import json
    bets = []
    for r in rows:
        odds = json.loads(r["odds_json"]) if r["odds_json"] else {}
        if not odds:
            continue
        bets.append({
            "sport": r["sport"],
            "confidence": r["confidence"],
            "won": r["is_correct"] == 1,
            "edge_pct": r["prob_a"] - odds.get("implied_prob_a", 0.5),
            "decimal_odds": 1 / odds.get("implied_prob_a", 0.5) if odds.get("implied_prob_a", 0) > 0 else 2.0,
        })
    from analytics import MarketEfficiencyAnalyzer
    return MarketEfficiencyAnalyzer.edge_report(bets)


# ── Phase 3 Endpoints ─────────────────────────────────────────────────────

@app.post("/analytics/season-simulation")
def season_simulation(
    league: str = "football",
    n_sims: int = 1000,
    playoff_spots: int = 4,
    relegation_spots: int = 3,
    standings: dict = None,
    remaining_fixtures: list = None,
):
    """Monte Carlo season simulation — project final standings from current state.

    POST body: {"standings": {"Team": {"points": N, "gd": N}},
                "remaining_fixtures": [{"home": "A", "away": "B"}]}
    """
    if not standings or not remaining_fixtures:
        raise HTTPException(status_code=400,
                            detail="standings and remaining_fixtures required")
    rules = LEAGUE_RULES.get(league, LEAGUE_RULES["football"])
    sim = SeasonSimulator(league_rules=rules)
    result = sim.simulate(standings, remaining_fixtures,
                          n_sims=n_sims, playoff_spots=playoff_spots,
                          relegation_spots=relegation_spots)
    return result


@app.post("/analytics/playoff-simulation")
def playoff_simulation(
    seeds: list[str] = None,
    best_of: int = 1,
    n_sims: int = 5000,
):
    """Simulate a seeded playoff bracket (single elimination or best-of-N).

    POST body: {"seeds": ["Team1", "Team2", ...], "best_of": 7}
    Bracket size must be a power of 2.
    """
    if not seeds or len(seeds) < 2:
        raise HTTPException(status_code=400,
                            detail="seeds list with >= 2 teams required")
    calc = PlayoffCalculator()
    result = calc.bracket_simulation(seeds, n_sims=n_sims, best_of=best_of)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/analytics/rolling-performance")
def rolling_performance(sport: str = None, n: int = 500):
    """Rolling accuracy and Brier score over time windows (20, 50, 100)."""
    monitor = MarketEfficiencyMonitor(db)
    result = monitor.rolling_performance(sport=sport, n=n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/analytics/degradation")
def check_degradation(sport: str = None, n: int = 200):
    """Check if model accuracy is degrading over recent predictions."""
    conn = db._get_conn()
    q = "SELECT prob_a, is_correct FROM predictions WHERE is_correct >= 0"
    params: list = []
    if sport:
        q += " AND sport=?"
        params.append(sport)
    q += " ORDER BY created_at DESC LIMIT ?"
    params.append(n)
    rows = conn.execute(q, params).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No scored predictions")

    accuracies = [
        1 if (r["prob_a"] > 0.5 and r["is_correct"] == 1) or
             (r["prob_a"] <= 0.5 and r["is_correct"] == 0) else 0
        for r in rows
    ]
    return MarketEfficiencyMonitor.detect_degradation(accuracies)


@app.get("/analytics/edge-niches")
def edge_niches(n: int = 500):
    """Find the most profitable sport x confidence niches (Sharpe-ranked)."""
    monitor = MarketEfficiencyMonitor(db)
    result = monitor.edge_by_niche(n=n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/analytics/prediction-evolution/{match_id}")
def prediction_evolution(match_id: str):
    """Get probability evolution timeline for a specific match."""
    ts = PredictionTimeSeries(db)
    evolution = ts.get_evolution(match_id)
    if not evolution:
        raise HTTPException(status_code=404,
                            detail=f"No prediction snapshots for {match_id}")
    drift = ts.probability_drift(match_id)
    return {"evolution": evolution, "drift": drift}


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting Oracle V2 API server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
