"""
Oracle V2 — Complete Prediction Engine
All ML models, player-level features, walk-forward testing, full pipeline.
"""
from __future__ import annotations
import json, math, hashlib, warnings, os, logging, time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")
logger = logging.getLogger("oracle.engine")

# Import core
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (
    OracleDB, RateLimiter, RatingEngine, ProbabilityCalibrator,
    WalkForwardBacktester, MonteCarloSimulator, BiasAuditor, BacktestResult,
    rate_limited_request
)

# API imports (loaded separately if available)
HAS_APIS = False
try:
    from all_apis import (
        OpenMeteoAPI, OddsAPI, TheSportsDB, ESPNAPI, CricSheetAPI,
        FootballDataAPI, BasketballAPI, NbaApiWrapper, ErgastF1API,
        ICCRankingsAPI, NewsInjuryAPI, GeocodingAPI, HistoricalDataAPI,
        RatingSystems, OracleDataAggregator
    )
    HAS_APIS = True
except Exception:
    try:
        from oracle_engine.api.all_apis import (
            OpenMeteoAPI, OddsAPI, TheSportsDB, ESPNAPI, CricSheetAPI,
            FootballDataAPI, BasketballAPI, NbaApiWrapper, ErgastF1API,
            ICCRankingsAPI, NewsInjuryAPI, GeocodingAPI, HistoricalDataAPI,
            RatingSystems, OracleDataAggregator
        )
        HAS_APIS = True
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. PLAYER-LEVEL FEATURE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class PlayerDatabase:
    """Comprehensive player-level stats for feature engineering."""

    # Cricket player database with venue/phase/vs-team breakdowns
    CRICKET_PLAYERS = {
        # India
        "Sanju Samson": {"team": "India", "role": "bat_wk", "batting_avg": 38.2, "batting_sr": 158.5,
            "t20_runs": 2100, "matches": 85, "impact": 92,
            "phase": {"pp": {"sr": 152, "avg": 35}, "middle": {"sr": 145, "avg": 38}, "death": {"sr": 195, "avg": 42}},
            "vs": {"NZ": {"avg": 45, "sr": 165}, "ENG": {"avg": 32, "sr": 148}},
            "venue": {"Ahmedabad": {"avg": 55, "sr": 172}, "Mumbai": {"avg": 42, "sr": 160}},
            "form_last5": [89, 97, 12, 45, 0], "fitness": 95},
        "Jasprit Bumrah": {"team": "India", "role": "bowl_pace", "bowling_avg": 18.2, "bowling_econ": 6.5,
            "bowling_sr": 16.8, "wickets": 95, "matches": 72, "impact": 97,
            "phase": {"pp": {"econ": 5.8, "sr": 14}, "middle": {"econ": 6.2, "sr": 18}, "death": {"econ": 7.5, "sr": 15}},
            "vs": {"NZ": {"econ": 6.0, "wkts": 8}, "ENG": {"econ": 7.1, "wkts": 6}},
            "venue": {"Ahmedabad": {"econ": 6.8, "wkts": 4}, "Mumbai": {"econ": 6.2, "wkts": 5}},
            "form_last5": [2, 3, 1, 2, 4], "fitness": 90},
        "Hardik Pandya": {"team": "India", "role": "allrounder", "batting_avg": 28.5, "batting_sr": 148.0,
            "bowling_econ": 8.2, "wickets": 55, "matches": 95, "impact": 88,
            "phase": {"pp": {"sr": 135}, "death": {"sr": 170, "econ": 9.0}},
            "vs": {"NZ": {"avg": 30, "sr": 155}}, "venue": {"Ahmedabad": {"avg": 35}},
            "form_last5": [43, 30, 15, 22, 38], "fitness": 82},
        "Suryakumar Yadav": {"team": "India", "role": "bat", "batting_avg": 34.8, "batting_sr": 168.2,
            "t20_runs": 2600, "matches": 72, "impact": 90,
            "phase": {"pp": {"sr": 145}, "middle": {"sr": 175}, "death": {"sr": 185}},
            "vs": {"NZ": {"avg": 38, "sr": 162}}, "venue": {"Ahmedabad": {"avg": 40, "sr": 170}},
            "form_last5": [34, 28, 52, 18, 45], "fitness": 88},
        "Varun Chakravarthy": {"team": "India", "role": "bowl_spin", "bowling_avg": 22.5, "bowling_econ": 7.0,
            "wickets": 52, "matches": 38, "impact": 78,
            "phase": {"pp": {"econ": 6.5}, "middle": {"econ": 6.8}, "death": {"econ": 8.2}},
            "vs": {"NZ": {"econ": 7.5}}, "form_last5": [1, 0, 2, 1, 0], "fitness": 85},
        "Arshdeep Singh": {"team": "India", "role": "bowl_pace", "bowling_avg": 20.5, "bowling_econ": 7.8,
            "wickets": 65, "matches": 55, "impact": 82,
            "phase": {"pp": {"econ": 7.2}, "death": {"econ": 8.5}},
            "form_last5": [2, 1, 3, 1, 2], "fitness": 90},
        "Abhishek Sharma": {"team": "India", "role": "bat", "batting_avg": 22.0, "batting_sr": 162.0,
            "matches": 30, "impact": 70,
            "phase": {"pp": {"sr": 170, "avg": 20}},
            "form_last5": [0, 0, 0, 5, 12], "fitness": 92},  # 3 ducks in WC
        # New Zealand
        "Finn Allen": {"team": "New Zealand", "role": "bat", "batting_avg": 30.5, "batting_sr": 162.8,
            "matches": 48, "impact": 88,
            "phase": {"pp": {"sr": 172, "avg": 32}},
            "vs": {"India": {"avg": 28, "sr": 155}, "SA": {"avg": 42, "sr": 180}},
            "form_last5": [100, 84, 25, 38, 12], "fitness": 95},  # 100 off 33 in SF!
        "Tim Seifert": {"team": "New Zealand", "role": "bat_wk", "batting_avg": 28.8, "batting_sr": 141.5,
            "matches": 55, "impact": 82,
            "wc_runs": 274, "wc_sr": 161.17,
            "form_last5": [89, 45, 62, 38, 40], "fitness": 90},
        "Mitchell Santner": {"team": "New Zealand", "role": "allrounder", "bowling_avg": 24.0,
            "bowling_econ": 6.8, "wickets": 72, "matches": 75, "impact": 80,
            "phase": {"pp": {"econ": 6.2}, "middle": {"econ": 6.5}},
            "vs": {"India": {"econ": 6.5, "wkts": 5}},
            "form_last5": [2, 1, 0, 2, 1], "fitness": 88},
        "Lockie Ferguson": {"team": "New Zealand", "role": "bowl_pace", "bowling_avg": 19.8,
            "bowling_econ": 7.5, "wickets": 58, "matches": 42, "impact": 85,
            "phase": {"death": {"econ": 8.8, "sr": 12}},
            "form_last5": [3, 2, 1, 2, 0], "fitness": 88},
        "Glenn Phillips": {"team": "New Zealand", "role": "allrounder", "batting_avg": 26.5,
            "batting_sr": 152.0, "bowling_econ": 7.2, "matches": 55, "impact": 78,
            "form_last5": [15, 42, 8, 28, 55], "fitness": 90},
        "Rachin Ravindra": {"team": "New Zealand", "role": "allrounder", "bowling_avg": 22.0,
            "bowling_econ": 7.0, "wickets": 15, "matches": 12, "impact": 75,
            "wc_wickets": 7, "form_last5": [2, 3, 0, 1, 1], "fitness": 92},
    }

    @classmethod
    def get_player(cls, name: str) -> Optional[dict]:
        return cls.CRICKET_PLAYERS.get(name)

    @classmethod
    def get_team_players(cls, team: str) -> list[dict]:
        return [{"name": k, **v} for k, v in cls.CRICKET_PLAYERS.items()
                if v.get("team") == team]

    @classmethod
    def team_aggregate_stats(cls, team: str, opponent: str = "",
                              venue: str = "") -> dict:
        """Aggregate player stats into team-level features."""
        players = cls.get_team_players(team)
        if not players:
            return {"avg_impact": 50, "batting_form_avg": 25, "bowling_form_avg": 1.5,
                    "vs_opponent_bat_avg": 50, "vs_opponent_bowl_econ": 7.5,
                    "venue_batting_avg": 50, "fitness_avg": 85, "powerplay_sr": 135,
                    "death_bowling_econ": 8.5, "player_count": 0,
                    "top_impact_player": 50, "weakest_player_impact": 50}

        batting = [p for p in players if "bat" in p.get("role", "")]
        bowling = [p for p in players if "bowl" in p.get("role", "")]
        allrounders = [p for p in players if "allrounder" in p.get("role", "")]

        # Average impact
        impacts = [p.get("impact", 50) for p in players]
        avg_impact = sum(impacts) / len(impacts) if impacts else 50

        # Form from last 5 (batting = avg runs, bowling = avg wickets)
        bat_form = []
        for p in batting + allrounders:
            f5 = p.get("form_last5", [])
            if f5:
                bat_form.append(sum(f5) / len(f5))

        bowl_form = []
        for p in bowling + allrounders:
            f5 = p.get("form_last5", [])
            if f5:
                bowl_form.append(sum(f5) / len(f5))

        # Vs opponent stats
        vs_batting_avg = 50
        vs_bowling_econ = 7.5
        if opponent:
            vs_avgs = [p.get("vs", {}).get(opponent, {}).get("avg", 0) for p in players if
                       p.get("vs", {}).get(opponent, {}).get("avg")]
            if vs_avgs:
                vs_batting_avg = sum(vs_avgs) / len(vs_avgs)
            vs_econs = [p.get("vs", {}).get(opponent, {}).get("econ", 0) for p in players if
                        p.get("vs", {}).get(opponent, {}).get("econ")]
            if vs_econs:
                vs_bowling_econ = sum(vs_econs) / len(vs_econs)

        # Venue stats
        venue_avg = 50
        if venue:
            v_key = venue.lower()
            v_avgs = []
            for p in players:
                for vk, vs in p.get("venue", {}).items():
                    if v_key in vk.lower():
                        if "avg" in vs:
                            v_avgs.append(vs["avg"])
            if v_avgs:
                venue_avg = sum(v_avgs) / len(v_avgs)

        # Fitness
        fitness = [p.get("fitness", 85) for p in players]
        avg_fitness = sum(fitness) / len(fitness) if fitness else 85

        # Phase-wise
        pp_sr = []
        death_econ = []
        for p in players:
            ph = p.get("phase", {})
            if "pp" in ph and "sr" in ph["pp"]:
                pp_sr.append(ph["pp"]["sr"])
            if "death" in ph and "econ" in ph["death"]:
                death_econ.append(ph["death"]["econ"])

        return {
            "avg_impact": avg_impact,
            "batting_form_avg": sum(bat_form) / len(bat_form) if bat_form else 25,
            "bowling_form_avg": sum(bowl_form) / len(bowl_form) if bowl_form else 1.5,
            "vs_opponent_bat_avg": vs_batting_avg,
            "vs_opponent_bowl_econ": vs_bowling_econ,
            "venue_batting_avg": venue_avg,
            "fitness_avg": avg_fitness,
            "powerplay_sr": sum(pp_sr) / len(pp_sr) if pp_sr else 135,
            "death_bowling_econ": sum(death_econ) / len(death_econ) if death_econ else 8.5,
            "player_count": len(players),
            "top_impact_player": max(impacts) if impacts else 50,
            "weakest_player_impact": min(impacts) if impacts else 50,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. COMPREHENSIVE FEATURE EXTRACTOR (40+ features)
# ═══════════════════════════════════════════════════════════════════════════

STAGE_PRESSURE = {
    "group": 1.0, "league": 1.0, "super_8": 1.15, "super8": 1.15,
    "quarter_final": 1.20, "semi_final": 1.30, "semi": 1.30,
    "final": 1.45, "dead_rubber": 0.85,
}

VENUE_DATA = {
    "ahmedabad": {"bat_first_win": 0.58, "avg_score": 178, "spin": 0.45, "pace": 0.70, "dew": 0.55, "altitude": 55, "boundary_sq": 70, "country": "India"},
    "mumbai":    {"bat_first_win": 0.52, "avg_score": 182, "spin": 0.40, "pace": 0.65, "dew": 0.60, "altitude": 14, "boundary_sq": 65, "country": "India"},
    "kolkata":   {"bat_first_win": 0.51, "avg_score": 172, "spin": 0.55, "pace": 0.55, "dew": 0.65, "altitude": 6, "boundary_sq": 64, "country": "India"},
    "chennai":   {"bat_first_win": 0.56, "avg_score": 165, "spin": 0.70, "pace": 0.50, "dew": 0.65, "altitude": 6, "boundary_sq": 60, "country": "India"},
    "delhi":     {"bat_first_win": 0.54, "avg_score": 170, "spin": 0.50, "pace": 0.60, "dew": 0.50, "altitude": 216, "boundary_sq": 62, "country": "India"},
    "bengaluru": {"bat_first_win": 0.48, "avg_score": 185, "spin": 0.35, "pace": 0.60, "dew": 0.40, "altitude": 920, "boundary_sq": 58, "country": "India"},
    "colombo":   {"bat_first_win": 0.53, "avg_score": 165, "spin": 0.65, "pace": 0.50, "dew": 0.50, "altitude": 7, "boundary_sq": 62, "country": "Sri Lanka"},
    "pallekele": {"bat_first_win": 0.54, "avg_score": 160, "spin": 0.50, "pace": 0.65, "dew": 0.35, "altitude": 500, "boundary_sq": 60, "country": "Sri Lanka"},
}


def get_venue_data(venue: str) -> dict:
    key = venue.lower().strip()
    for name, data in VENUE_DATA.items():
        if name in key or key in name:
            return data
    return {"bat_first_win": 0.52, "avg_score": 170, "spin": 0.50, "pace": 0.55,
            "dew": 0.40, "altitude": 50, "boundary_sq": 65, "country": ""}


def extract_features(match: dict, db: OracleDB = None, weather: dict = None) -> np.ndarray:
    """
    Extract 50+ features for a match prediction.
    Combines team ratings, player stats, venue, weather, odds, H2H, form.
    """
    ta = match["team_a"]
    tb = match["team_b"]
    sport = match.get("sport", "cricket")
    venue = match.get("venue", "")
    stage = match.get("stage", "group")

    # --- Team Elo ratings ---
    elo_a = match.get("elo_a", 1500)
    elo_b = match.get("elo_b", 1500)
    if db:
        ra = db.get_rating(ta, sport, "elo")
        rb = db.get_rating(tb, sport, "elo")
        elo_a = ra["rating"]
        elo_b = rb["rating"]

    elo_diff = elo_a - elo_b
    elo_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))

    # --- Glicko-2 ratings ---
    glicko_diff = 0
    if db:
        ga = db.get_rating(ta, sport, "glicko2")
        gb = db.get_rating(tb, sport, "glicko2")
        glicko_diff = ga["rating"] - gb["rating"]

    # --- Player-level aggregates ---
    pa = PlayerDatabase.team_aggregate_stats(ta, tb, venue)
    pb = PlayerDatabase.team_aggregate_stats(tb, ta, venue)

    # --- H2H ---
    h2h_adv = 0.5
    if db:
        h2h = db.get_h2h(ta, tb, sport)
        if h2h["total"] > 0:
            h2h_adv = h2h["a_wins"] / h2h["total"]
    h2h_adv = match.get("h2h_adv", h2h_adv)

    # --- Recent form ---
    form_a = match.get("form_a", [1, 1, 0, 1, 1])
    form_b = match.get("form_b", [1, 0, 1, 1, 0])

    def form_score(results):
        weights = [math.exp(-0.15 * i) for i in range(len(results))]
        return sum(w * r for w, r in zip(weights, results)) / sum(weights) if weights else 0.5

    def form_velocity(results):
        """Compute slope of recent results — positive = improving."""
        if len(results) < 3:
            return 0.0
        n = len(results)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(results) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, results))
        den = sum((xi - x_mean) ** 2 for xi in x)
        return num / den if den > 0 else 0.0

    def form_volatility(results):
        """Std dev of recent results — high = inconsistent."""
        if len(results) < 2:
            return 0.0
        mean = sum(results) / len(results)
        return math.sqrt(sum((r - mean) ** 2 for r in results) / len(results))

    fa = form_score(form_a)
    fb = form_score(form_b)
    vel_a = form_velocity(form_a)
    vel_b = form_velocity(form_b)
    vol_a = form_volatility(form_a)
    vol_b = form_volatility(form_b)

    # --- Venue ---
    vd = get_venue_data(venue)

    # --- Home advantage ---
    home_adv = 0
    home_countries = {"India": ["India"], "Sri Lanka": ["Sri Lanka"],
                      "England": ["England"], "Australia": ["Australia"]}
    if not match.get("is_neutral", False):
        if vd["country"] in home_countries.get(ta, []):
            home_adv = 1
        elif vd["country"] in home_countries.get(tb, []):
            home_adv = -1

    # --- Toss ---
    toss_adv = 0
    if match.get("toss_winner") == ta: toss_adv = 1
    elif match.get("toss_winner") == tb: toss_adv = -1

    # --- Weather / Dew ---
    dew_risk = vd["dew"]
    temp = 28.0
    humidity = 50.0
    wind = 10.0
    rain_prob = 0.0
    if weather:
        dew_risk = weather.get("dew_risk", dew_risk)
        temp = weather.get("temperature_c", 28) or 28
        humidity = weather.get("humidity_pct", 50) or 50
        wind = weather.get("wind_speed_kmh", 10) or 10
        rain_prob = weather.get("precipitation_prob_pct", 0) or 0

    # --- Betting odds ---
    odds_prob_a = match.get("odds_prob_a", 0.5)
    odds_prob_b = match.get("odds_prob_b", 0.5)

    # --- Pressure ---
    pressure = STAGE_PRESSURE.get(stage.lower().replace(" ", "_"), 1.0)

    # --- Win streaks ---
    streak_a = match.get("streak_a", 0)
    streak_b = match.get("streak_b", 0)
    momentum_diff = min(streak_a * 0.03, 0.15) - min(streak_b * 0.03, 0.15)

    # --- Injury ---
    injury_a = match.get("injury_impact_a", 0.0)
    injury_b = match.get("injury_impact_b", 0.0)

    # --- Fatigue ---
    rest_a = match.get("days_rest_a", 3)
    rest_b = match.get("days_rest_b", 3)

    # --- Day/Night ---
    is_day_night = float(match.get("is_day_night", False))
    is_neutral = float(match.get("is_neutral", False))

    # ═══ Build feature vector (56 features) ═══
    features = np.array([
        # Rating systems (4)
        elo_diff, elo_prob, glicko_diff,
        elo_a - elo_b,  # redundant but different scale helps some models

        # Player aggregates (12)
        pa["avg_impact"] - pb["avg_impact"],
        pa["batting_form_avg"] - pb["batting_form_avg"],
        pa["bowling_form_avg"] - pb["bowling_form_avg"],
        pa["vs_opponent_bat_avg"] - pb["vs_opponent_bat_avg"],
        pa["vs_opponent_bowl_econ"] - pb["vs_opponent_bowl_econ"],
        pa["venue_batting_avg"] - pb["venue_batting_avg"],
        pa["fitness_avg"] - pb["fitness_avg"],
        pa["powerplay_sr"] - pb["powerplay_sr"],
        pa["death_bowling_econ"] - pb["death_bowling_econ"],
        pa["top_impact_player"] - pb["top_impact_player"],
        pa["weakest_player_impact"] - pb["weakest_player_impact"],
        pa["avg_impact"],  # absolute for team A

        # H2H (2)
        h2h_adv, h2h_adv - 0.5,

        # Form (3)
        fa, fb, fa - fb,

        # Form velocity & volatility (4) — NEW
        vel_a, vel_b,             # momentum direction
        vel_a - vel_b,            # relative momentum
        vol_a - vol_b,            # consistency difference

        # Momentum (1)
        momentum_diff,

        # Venue (7)
        vd["bat_first_win"], vd["avg_score"] / 200,
        vd["spin"], vd["pace"], vd["boundary_sq"] / 80,
        vd["altitude"] / 1000, dew_risk,

        # Home/Toss (2)
        home_adv, toss_adv,

        # Weather (4)
        temp / 40, humidity / 100, wind / 50, rain_prob / 100,

        # Odds (3)
        odds_prob_a, odds_prob_b, odds_prob_a - odds_prob_b,

        # Context (5)
        pressure, is_day_night, is_neutral,
        injury_a - injury_b, rest_a - rest_b,

        # Absolute ratings (4)
        elo_a / 2200, elo_b / 2200,
        pa["avg_impact"] / 100, pb["avg_impact"] / 100,

        # Interaction features (5) — expanded
        dew_risk * is_day_night,  # dew only matters at night
        home_adv * pressure,      # home advantage amplified in knockouts
        elo_prob * odds_prob_a,   # elo-market agreement
        vel_a * pressure,         # momentum amplified in knockouts
        fa * (1 - vol_a),         # form weighted by consistency
    ])

    return features


FEATURE_NAMES = [
    "elo_diff", "elo_prob", "glicko_diff", "rating_gap",
    "player_impact_diff", "bat_form_diff", "bowl_form_diff",
    "vs_opp_bat_diff", "vs_opp_bowl_diff", "venue_bat_diff",
    "fitness_diff", "pp_sr_diff", "death_econ_diff",
    "top_impact_diff", "weakest_impact_diff", "impact_a_abs",
    "h2h_advantage", "h2h_centered",
    "form_a", "form_b", "form_diff",
    "form_velocity_a", "form_velocity_b", "velocity_diff", "volatility_diff",
    "momentum_diff",
    "venue_bat_first", "venue_avg_norm", "venue_spin", "venue_pace",
    "venue_boundary", "venue_altitude", "dew_risk",
    "home_advantage", "toss_advantage",
    "temperature", "humidity", "wind", "rain_prob",
    "odds_prob_a", "odds_prob_b", "odds_diff",
    "pressure", "is_day_night", "is_neutral",
    "injury_diff", "rest_diff",
    "elo_a_norm", "elo_b_norm", "impact_a_norm", "impact_b_norm",
    "dew_x_night", "home_x_pressure", "elo_x_odds",
    "momentum_x_pressure", "form_x_consistency",
]


# ═══════════════════════════════════════════════════════════════════════════
# 3. ML MODEL SUITE (12 models + Super Ensemble)
# ═══════════════════════════════════════════════════════════════════════════

def build_models() -> dict[str, Any]:
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=2,
            random_state=42, class_weight="balanced", ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            random_state=42, subsample=0.8),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=5000, random_state=42, class_weight="balanced"),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=3000,
            random_state=42, learning_rate="adaptive"),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42),
        "NaiveBayes": GaussianNB(),
        "Bagging": BaggingClassifier(
            n_estimators=100, max_samples=0.8, random_state=42, ),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", use_label_encoder=False)
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    return models


# ═══════════════════════════════════════════════════════════════════════════
# 4. COMPLETE PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class OracleV2:
    """
    Production-grade prediction engine.

    Pipeline:
    1. Gather data (APIs + DB)
    2. Extract 50 features (player-level, venue, weather, odds, H2H)
    3. Predict with 12 models
    4. Calibrate probabilities (Platt scaling)
    5. Audit for bias
    6. Store prediction + features for backtesting
    7. Update ratings after result
    """

    def __init__(self):
        self.db = OracleDB()
        self.ratings = RatingEngine(self.db)
        self.calibrator = ProbabilityCalibrator()
        self.scaler = StandardScaler()
        self.models: dict[str, Any] = {}
        self.meta_learner = None
        self.is_trained = False
        self.training_stats = {}

    def load_historical_data(self, matches: list[dict]):
        """Load historical matches into DB and update ratings."""
        for m in matches:
            self.db.insert_match(m)
            if m.get("winner") and m["winner"] not in ("", "NO RESULT"):
                loser = m["team_b"] if m["winner"] == m["team_a"] else m["team_a"]
                margin = float(m.get("margin_numeric", 0))
                self.ratings.update_all(m["winner"], loser,
                                        m.get("sport", "cricket"), margin)

    def train(self, matches: list[dict]) -> dict:
        """Train all models on historical data."""
        X, y = [], []
        for m in matches:
            if not m.get("winner") or m["winner"] in ("", "NO RESULT"):
                continue
            features = extract_features(m, self.db)
            label = 1 if m["winner"] == m["team_a"] else 0
            X.append(features)
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        if len(X) < 5:
            return {"error": "Need at least 5 matches to train"}

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.models = build_models()
        cv_scores = {}
        n_cv = min(5, len(y))

        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                scores = cross_val_score(model, X_scaled, y, cv=n_cv,
                                         scoring="accuracy")
                cv_scores[name] = round(scores.mean(), 3)
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                cv_scores[name] = 0

        # Super ensemble
        estimators = [(n.replace(" ","_")[:20], m) for n, m in self.models.items()
                      if hasattr(m, "predict_proba")]
        if len(estimators) >= 2:
            voting = VotingClassifier(estimators=estimators, voting="soft")
            try:
                voting.fit(X_scaled, y)
                self.models["SuperEnsemble"] = voting
                scores = cross_val_score(voting, X_scaled, y, cv=n_cv, scoring="accuracy")
                cv_scores["SuperEnsemble"] = round(scores.mean(), 3)
            except Exception as e:
                logger.warning(f"Super ensemble failed: {e}")

        # Stacking meta-learner: train a LogisticRegression on base model outputs
        self.meta_learner = None
        if len(X_scaled) >= 10:
            meta_features = []
            for xi in X_scaled:
                row = []
                for m in self.models.values():
                    if hasattr(m, "predict_proba"):
                        try:
                            p = m.predict_proba(xi.reshape(1, -1))[0]
                            row.append(p[1] if len(p) > 1 else p[0])
                        except Exception:
                            row.append(0.5)
                meta_features.append(row)
            if meta_features and len(meta_features[0]) >= 2:
                meta_X = np.array(meta_features)
                # Replace NaN/Inf with 0.5 (neutral probability)
                meta_X = np.nan_to_num(meta_X, nan=0.5, posinf=0.5, neginf=0.5)
                try:
                    self.meta_learner = LogisticRegression(max_iter=2000, random_state=42)
                    self.meta_learner.fit(meta_X, y)
                    meta_cv = cross_val_score(self.meta_learner, meta_X, y,
                                              cv=min(5, len(y)), scoring="accuracy")
                    cv_scores["StackedMeta"] = round(meta_cv.mean(), 3)
                except Exception as e:
                    logger.warning(f"Meta-learner failed: {e}")

        # Calibration — use last 30% as held-out to avoid leakage
        all_probs = []
        for xi in X_scaled:
            preds = []
            for m in self.models.values():
                if hasattr(m, "predict_proba"):
                    try:
                        p = m.predict_proba(xi.reshape(1, -1))[0]
                        preds.append(p[1] if len(p) > 1 else p[0])
                    except Exception:
                        pass
            if preds:
                all_probs.append(np.mean(preds))
        if all_probs:
            all_probs_arr = np.array(all_probs)
            # Split: fit calibrator on held-out portion to prevent leakage
            cal_split = max(int(len(all_probs_arr) * 0.7), 1)
            cal_probs = all_probs_arr[cal_split:]
            cal_labels = y[cal_split:]
            if len(cal_probs) >= 5:
                self.calibrator.fit_platt(cal_probs, cal_labels)
                self.calibrator.fit_isotonic(cal_probs, cal_labels, n_bins=15)
            else:
                # Not enough held-out data, fit on all (small dataset fallback)
                self.calibrator.fit_platt(all_probs_arr, y)
                self.calibrator.fit_isotonic(all_probs_arr, y, n_bins=15)

        self.is_trained = True
        self.training_stats = {
            "matches": len(X), "features": X.shape[1],
            "models": len(self.models), "cv_scores": cv_scores,
            "has_meta_learner": self.meta_learner is not None,
        }
        return self.training_stats

    def predict(self, match: dict) -> dict:
        """Generate a full prediction with all models + calibration."""
        if not self.is_trained:
            return {"error": "Model not trained"}

        # Gather weather if API available
        weather = None
        if HAS_APIS and match.get("venue"):
            weather = None
            if HAS_APIS:
                try:
                    weather = OpenMeteoAPI.get_weather(match["venue"], match.get("date"))
                except Exception:
                    weather = None

        features = extract_features(match, self.db, weather)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        model_votes = {}
        probs_a = []
        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    p = model.predict_proba(features_scaled)[0]
                    prob_a = p[1] if len(p) > 1 else p[0]
                else:
                    pred = model.predict(features_scaled)[0]
                    prob_a = 0.85 if pred == 1 else 0.15
                model_votes[name] = {
                    "prob_a": round(prob_a, 4),
                    "winner": match["team_a"] if prob_a > 0.5 else match["team_b"],
                }
                probs_a.append(prob_a)
            except Exception:
                continue

        # Use stacking meta-learner if available, else average
        raw_prob_a = np.mean(probs_a) if probs_a else 0.5
        if self.meta_learner is not None and len(probs_a) >= 2:
            try:
                meta_input = np.array(probs_a).reshape(1, -1)
                meta_prob = self.meta_learner.predict_proba(meta_input)[0]
                raw_prob_a = meta_prob[1] if len(meta_prob) > 1 else meta_prob[0]
            except Exception:
                pass  # fallback to ensemble average
        calibrated_prob_a = self.calibrator.calibrate(raw_prob_a)
        prob_b = 1 - calibrated_prob_a

        # Model disagreement: high std dev = models disagree = lower confidence
        model_std = float(np.std(probs_a)) if len(probs_a) >= 2 else 0.0
        disagreement_penalty = min(model_std / 0.3, 1.0)  # 0-1 scale

        winner = match["team_a"] if calibrated_prob_a > 0.5 else match["team_b"]
        diff = abs(calibrated_prob_a - 0.5) * 2
        # Penalize confidence when models disagree significantly
        effective_diff = diff * (1 - disagreement_penalty * 0.4)
        confidence = ("VERY HIGH" if effective_diff > 0.45 else "HIGH" if effective_diff > 0.30 else
                      "MODERATE" if effective_diff > 0.15 else "LOW" if effective_diff > 0.05 else "TOSS-UP")

        # Feature importance
        feat_imp = {}
        for mname in ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]:
            if mname in self.models and hasattr(self.models[mname], "feature_importances_"):
                for fn, imp in zip(FEATURE_NAMES, self.models[mname].feature_importances_):
                    feat_imp[fn] = feat_imp.get(fn, 0) + imp
        if feat_imp:
            total = sum(feat_imp.values())
            feat_imp = {k: round(v/total, 4) for k, v in
                        sorted(feat_imp.items(), key=lambda x: -x[1])[:15]}

        # Confidence interval from bootstrap over model outputs
        ci_data = {}
        if len(probs_a) >= 3:
            rng = np.random.default_rng(42)
            boot_means = [
                float(np.mean(rng.choice(probs_a, size=len(probs_a), replace=True)))
                for _ in range(500)
            ]
            ci_data = {
                "mean": round(float(np.mean(boot_means)) * 100, 1),
                "ci_lower": round(float(np.percentile(boot_means, 5)) * 100, 1),
                "ci_upper": round(float(np.percentile(boot_means, 95)) * 100, 1),
                "ci_width": round(float(np.percentile(boot_means, 95) - np.percentile(boot_means, 5)) * 100, 1),
            }

        # EV / Kelly analytics (when market odds available)
        analytics_data = {}
        odds_prob_a = match.get("odds_prob_a", 0)
        odds_prob_b = match.get("odds_prob_b", 0)
        if odds_prob_a > 0 and odds_prob_b > 0:
            try:
                from analytics import EVCalculator, KellyStaker
                odds_a_decimal = 1.0 / odds_prob_a if odds_prob_a > 0 else 0
                odds_b_decimal = 1.0 / odds_prob_b if odds_prob_b > 0 else 0
                ev_a = EVCalculator.calculate_ev(calibrated_prob_a, odds_a_decimal)
                ev_b = EVCalculator.calculate_ev(prob_b, odds_b_decimal)
                kelly_a = KellyStaker.kelly_fraction(calibrated_prob_a, odds_a_decimal)
                kelly_b = KellyStaker.kelly_fraction(prob_b, odds_b_decimal)
                analytics_data = {
                    "ev_a": ev_a,
                    "ev_b": ev_b,
                    "kelly_a_pct": round(kelly_a * 100, 2),
                    "kelly_b_pct": round(kelly_b * 100, 2),
                    "value_side": match["team_a"] if ev_a["ev"] > ev_b["ev"] and ev_a["ev"] > 0
                                  else (match["team_b"] if ev_b["ev"] > 0 else "NO VALUE"),
                }
            except ImportError:
                pass

        prediction = {
            "team_a": match["team_a"],
            "team_b": match["team_b"],
            "venue": match.get("venue", ""),
            "stage": match.get("stage", ""),
            "raw_prob_a": round(raw_prob_a * 100, 1),
            "calibrated_prob_a": round(calibrated_prob_a * 100, 1),
            "calibrated_prob_b": round(prob_b * 100, 1),
            "predicted_winner": winner,
            "confidence": confidence,
            "confidence_interval": ci_data,
            "model_agreement": {
                "std_dev": round(model_std, 4),
                "disagreement_pct": round(disagreement_penalty * 100, 1),
                "consensus": "STRONG" if model_std < 0.08 else "MODERATE" if model_std < 0.15 else "WEAK",
            },
            "model_votes": model_votes,
            "models_for_a": sum(1 for v in model_votes.values()
                                if v["winner"] == match["team_a"]),
            "models_for_b": sum(1 for v in model_votes.values()
                                if v["winner"] == match["team_b"]),
            "feature_importance": feat_imp,
            "analytics": analytics_data,
            "weather": weather,
            "odds_market": {"prob_a": match.get("odds_prob_a", 0),
                           "prob_b": match.get("odds_prob_b", 0)},
        }

        # Store prediction
        self.db.insert_prediction({
            "sport": match.get("sport", "cricket"),
            "team_a": match["team_a"], "team_b": match["team_b"],
            "prob_a": calibrated_prob_a, "prob_b": prob_b,
            "predicted_winner": winner, "confidence": confidence,
            "model_votes": model_votes,
            "features": {fn: float(fv) for fn, fv in zip(FEATURE_NAMES, features)},
        })

        self.db.log_event("prediction", {
            "match": f"{match['team_a']} vs {match['team_b']}",
            "winner": winner, "prob": round(calibrated_prob_a * 100, 1),
        })

        return prediction

    def backtest(self, matches: list[dict], min_train: int = 10) -> BacktestResult:
        """Run walk-forward backtest."""
        # Prepare features for all matches
        for m in matches:
            if "features" not in m:
                m["features"] = extract_features(m, self.db).tolist()

        from sklearn.linear_model import LogisticRegression
        def model_factory():
            return LogisticRegression(max_iter=2000, random_state=42)

        backtester = WalkForwardBacktester(
            model_factory=model_factory,
            feature_extractor=lambda m: m["features"],
            min_training_size=min_train
        )
        return backtester.run(matches)

    def monte_carlo(self, groups: dict[str, list[str]],
                    n_sims: int = 10000) -> dict:
        """Run tournament simulation."""
        def predict_fn(a, b):
            match = {"team_a": a, "team_b": b, "sport": "cricket"}
            features = extract_features(match, self.db)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            probs = []
            for m in self.models.values():
                if hasattr(m, "predict_proba"):
                    try:
                        p = m.predict_proba(features_scaled)[0]
                        probs.append(p[1] if len(p) > 1 else p[0])
                    except Exception:
                        pass
            pa = np.mean(probs) if probs else 0.5
            return pa, 1 - pa

        mc = MonteCarloSimulator(predict_fn, n_sims)
        return mc.simulate_tournament(groups)

    def audit(self) -> dict:
        """Run bias audit on all stored predictions."""
        conn = self.db._get_conn()
        rows = conn.execute(
            "SELECT * FROM predictions WHERE is_correct >= 0"
        ).fetchall()
        preds = [dict(r) for r in rows]
        return BiasAuditor.audit(preds)


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN — RUN EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 88 + "╗")
    print("║   ORACLE V2 — COMPLETE PRODUCTION PIPELINE                                        ║")
    print("║   50 Features | 12 Models | Player-Level | Calibrated | Walk-Forward Tested        ║")
    print("╚" + "═" * 88 + "╝")

    oracle = OracleV2()

    # ── Load T20 WC 2026 data ──
    matches = [
        {"team_a": "Pakistan", "team_b": "Netherlands", "winner": "Pakistan", "venue": "Colombo", "stage": "group", "sport": "cricket", "toss_winner": "Netherlands", "toss_decision": "field", "form_a": [0,1,0,1,1], "form_b": [0,1,0,0,1], "elo_a": 1920, "elo_b": 1720},
        {"team_a": "India", "team_b": "United States of America", "winner": "India", "venue": "Mumbai", "stage": "group", "sport": "cricket", "toss_winner": "India", "toss_decision": "bat", "form_a": [1,1,1,1,1], "form_b": [0,0,1,0,0], "elo_a": 2150, "elo_b": 1560},
        {"team_a": "West Indies", "team_b": "Scotland", "winner": "West Indies", "venue": "Kolkata", "stage": "group", "sport": "cricket", "form_a": [1,0,1,1,0], "form_b": [0,1,0,0,1], "elo_a": 1950, "elo_b": 1680},
        {"team_a": "New Zealand", "team_b": "Afghanistan", "winner": "New Zealand", "venue": "Chennai", "stage": "group", "sport": "cricket", "form_a": [1,1,1,0,1], "form_b": [1,1,0,1,0], "elo_a": 2020, "elo_b": 1850},
        {"team_a": "England", "team_b": "Nepal", "winner": "England", "venue": "Mumbai", "stage": "group", "sport": "cricket", "form_a": [1,1,1,0,1], "form_b": [0,1,0,0,1], "elo_a": 2080, "elo_b": 1540},
        {"team_a": "Sri Lanka", "team_b": "Ireland", "winner": "Sri Lanka", "venue": "Colombo", "stage": "group", "sport": "cricket", "elo_a": 1880, "elo_b": 1700},
        {"team_a": "Zimbabwe", "team_b": "Oman", "winner": "Zimbabwe", "venue": "Colombo", "stage": "group", "sport": "cricket", "elo_a": 1660, "elo_b": 1480},
        {"team_a": "South Africa", "team_b": "Canada", "winner": "South Africa", "venue": "Ahmedabad", "stage": "group", "sport": "cricket", "elo_a": 2040, "elo_b": 1520},
        {"team_a": "Australia", "team_b": "Ireland", "winner": "Australia", "venue": "Colombo", "stage": "group", "sport": "cricket", "elo_a": 2050, "elo_b": 1700},
        {"team_a": "Zimbabwe", "team_b": "Australia", "winner": "Zimbabwe", "venue": "Colombo", "stage": "group", "sport": "cricket", "elo_a": 1680, "elo_b": 2050},
        {"team_a": "South Africa", "team_b": "Afghanistan", "winner": "South Africa", "venue": "Ahmedabad", "stage": "group", "sport": "cricket", "elo_a": 2040, "elo_b": 1850},
        {"team_a": "West Indies", "team_b": "England", "winner": "West Indies", "venue": "Mumbai", "stage": "group", "sport": "cricket", "elo_a": 1960, "elo_b": 2085},
        {"team_a": "Australia", "team_b": "Sri Lanka", "winner": "Sri Lanka", "venue": "Pallekele", "stage": "group", "sport": "cricket", "elo_a": 2030, "elo_b": 1900},
        {"team_a": "India", "team_b": "Namibia", "winner": "India", "venue": "Delhi", "stage": "group", "sport": "cricket", "elo_a": 2165, "elo_b": 1580, "margin_numeric": 93},
        {"team_a": "India", "team_b": "Pakistan", "winner": "India", "venue": "Colombo", "stage": "group", "sport": "cricket", "elo_a": 2175, "elo_b": 1935, "margin_numeric": 61},
        {"team_a": "New Zealand", "team_b": "South Africa", "winner": "South Africa", "venue": "Ahmedabad", "stage": "group", "sport": "cricket", "elo_a": 2040, "elo_b": 2060},
        {"team_a": "India", "team_b": "Netherlands", "winner": "India", "venue": "Ahmedabad", "stage": "group", "sport": "cricket", "elo_a": 2190, "elo_b": 1700},
        {"team_a": "India", "team_b": "South Africa", "winner": "South Africa", "venue": "Ahmedabad", "stage": "super8", "sport": "cricket", "elo_a": 2195, "elo_b": 2070, "margin_numeric": 76},
        {"team_a": "West Indies", "team_b": "Zimbabwe", "winner": "West Indies", "venue": "Mumbai", "stage": "super8", "sport": "cricket", "elo_a": 1970, "elo_b": 1700},
        {"team_a": "England", "team_b": "Sri Lanka", "winner": "England", "venue": "Pallekele", "stage": "super8", "sport": "cricket", "elo_a": 2070, "elo_b": 1870},
        {"team_a": "England", "team_b": "Pakistan", "winner": "England", "venue": "Pallekele", "stage": "super8", "sport": "cricket", "elo_a": 2085, "elo_b": 1920},
        {"team_a": "New Zealand", "team_b": "Sri Lanka", "winner": "New Zealand", "venue": "Colombo", "stage": "super8", "sport": "cricket", "elo_a": 2025, "elo_b": 1860},
        {"team_a": "South Africa", "team_b": "West Indies", "winner": "South Africa", "venue": "Ahmedabad", "stage": "super8", "sport": "cricket", "elo_a": 2090, "elo_b": 1975},
        {"team_a": "India", "team_b": "Zimbabwe", "winner": "India", "venue": "Chennai", "stage": "super8", "sport": "cricket", "elo_a": 2170, "elo_b": 1690, "margin_numeric": 72},
        {"team_a": "England", "team_b": "New Zealand", "winner": "England", "venue": "Colombo", "stage": "super8", "sport": "cricket", "elo_a": 2095, "elo_b": 2035},
        {"team_a": "India", "team_b": "West Indies", "winner": "India", "venue": "Kolkata", "stage": "super8", "sport": "cricket", "elo_a": 2185, "elo_b": 1960},
        {"team_a": "South Africa", "team_b": "Zimbabwe", "winner": "South Africa", "venue": "Delhi", "stage": "super8", "sport": "cricket", "elo_a": 2100, "elo_b": 1680},
        {"team_a": "New Zealand", "team_b": "South Africa", "winner": "New Zealand", "venue": "Kolkata", "stage": "semi", "sport": "cricket", "elo_a": 2030, "elo_b": 2110, "toss_winner": "New Zealand", "toss_decision": "field"},
        {"team_a": "India", "team_b": "England", "winner": "India", "venue": "Mumbai", "stage": "semi", "sport": "cricket", "elo_a": 2195, "elo_b": 2100, "toss_winner": "England", "toss_decision": "field", "margin_numeric": 7},
    ]

    print(f"\n  Loading {len(matches)} historical matches...")
    oracle.load_historical_data(matches)

    # ── Train ──
    print(f"  Training 12 ML models on {len(matches)} matches × 50 features...")
    stats = oracle.train(matches)
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  MODEL CROSS-VALIDATION ACCURACY                │")
    print(f"  └─────────────────────────────────────────────────┘")
    for name, score in sorted(stats.get("cv_scores", {}).items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(score * 30)
        print(f"  {name:<22} {bar} {score:.3f}")

    # ── Walk-Forward Backtest ──
    print(f"\n  Running walk-forward backtest...")
    bt = oracle.backtest(matches, min_train=8)
    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  WALK-FORWARD BACKTEST RESULTS                  │")
    print(f"  │  Accuracy: {bt.accuracy:.1%} ({bt.correct}/{bt.total_matches}){' ' * 20}│")
    print(f"  │  Brier Score: {bt.brier_score:.4f} (lower=better){' ' * 13}│")
    print(f"  │  Log Loss: {bt.log_loss:.4f}{' ' * 27}│")
    print(f"  │  Calibration Error: {bt.calibration_error:.4f}{' ' * 18}│")
    print(f"  └─────────────────────────────────────────────────┘")
    if bt.by_stage:
        for stage, data in bt.by_stage.items():
            acc = data["correct"] / data["total"] if data["total"] else 0
            print(f"    {stage}: {acc:.0%} ({data['correct']}/{data['total']})")

    # ── Predict the Final ──
    print(f"\n  {'═' * 70}")
    print(f"  🏆 PREDICTING: INDIA vs NEW ZEALAND — T20 WC FINAL")
    print(f"  {'═' * 70}")

    final_pred = oracle.predict({
        "team_a": "India", "team_b": "New Zealand",
        "venue": "Ahmedabad", "stage": "final", "sport": "cricket",
        "date": "2026-03-08", "is_day_night": True,
        "toss_winner": "New Zealand", "toss_decision": "field",
        "form_a": [1, 1, 1, 0, 1, 1, 1], "form_b": [1, 1, 0, 1, 0, 1, 1],
        "streak_a": 3, "streak_b": 2,
        "odds_prob_a": 0.70, "odds_prob_b": 0.30,  # Market consensus
        "elo_a": 2195, "elo_b": 2040,
    })

    print(f"\n  India:        {final_pred['calibrated_prob_a']}% (raw: {final_pred['raw_prob_a']}%)")
    print(f"  New Zealand:  {final_pred['calibrated_prob_b']}%")
    print(f"  Winner:       {final_pred['predicted_winner']} [{final_pred['confidence']}]")
    print(f"  Models:       {final_pred['models_for_a']} for India | "
          f"{final_pred['models_for_b']} for NZ")
    if final_pred.get("weather"):
        w = final_pred["weather"]
        print(f"  Weather:      {w.get('conditions_summary', 'N/A')}")
        print(f"  Dew Risk:     {w.get('dew_risk', 'N/A')}")
    print(f"  Market Odds:  India {final_pred['odds_market']['prob_a']*100:.0f}% | "
          f"NZ {final_pred['odds_market']['prob_b']*100:.0f}%")

    if final_pred.get("feature_importance"):
        print(f"\n  Top Features:")
        for fn, imp in list(final_pred["feature_importance"].items())[:8]:
            bar = "█" * int(imp * 200)
            print(f"    {fn:<25} {bar} {imp:.4f}")

    # ── Monte Carlo ──
    print(f"\n  Running Monte Carlo simulation (5,000 tournaments)...")
    # Use fast Elo-based prediction for MC speed
    team_elos = {}
    for t in ["India", "South Africa", "West Indies", "Zimbabwe",
              "England", "New Zealand", "Pakistan", "Sri Lanka"]:
        r = oracle.db.get_rating(t, "cricket", "elo")
        team_elos[t] = r["rating"]

    def fast_predict(a, b):
        ea = team_elos.get(a, 1500)
        eb = team_elos.get(b, 1500)
        pa = 1 / (1 + 10**((eb - ea) / 400))
        return pa, 1 - pa

    mc = MonteCarloSimulator(fast_predict, n_simulations=5000)
    mc_results = mc.simulate_tournament({
        "Group1": ["India", "South Africa", "West Indies", "Zimbabwe"],
        "Group2": ["England", "New Zealand", "Pakistan", "Sri Lanka"],
    })
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  MONTE CARLO: TOURNAMENT WIN PROBABILITY        │")
    print(f"  └─────────────────────────────────────────────────┘")
    for team, probs in list(mc_results.items())[:6]:
        bar = "█" * int(probs["champion_pct"] / 2)
        print(f"  {team:<20} {bar} {probs['champion_pct']:.1f}%")

    # ── Reliability Diagram ──
    print(f"\n  Calibration (Reliability Diagram):")
    rel = oracle.calibrator.reliability_diagram()
    if rel.get("bins"):
        for b in rel["bins"]:
            if b["count"] > 0:
                pred_bar = "▓" * int(b["predicted"] * 20)
                act_bar = "█" * int(b["actual"] * 20)
                print(f"    Pred {b['predicted']:.2f} {pred_bar}")
                print(f"    Real {b['actual']:.2f} {act_bar}  (n={b['count']})")

    # ── DB Stats ──
    print(f"\n  Database: {oracle.db.get_stats()}")

    # ── Save outputs ──
    output_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "final_prediction.json", "w") as f:
        json.dump(final_pred, f, indent=2, default=str)

    with open(output_dir / "backtest_results.json", "w") as f:
        json.dump(asdict(bt), f, indent=2, default=str)

    with open(output_dir / "monte_carlo.json", "w") as f:
        json.dump(mc_results, f, indent=2, default=str)

    print(f"\n  📁 Saved: oracle_v2/final_prediction.json")
    print(f"  📁 Saved: oracle_v2/backtest_results.json")
    print(f"  📁 Saved: oracle_v2/monte_carlo.json")
    print(f"\n{'═' * 80}")


if __name__ == "__main__":
    main()
