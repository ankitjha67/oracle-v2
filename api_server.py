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


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting Oracle V2 API server...")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
