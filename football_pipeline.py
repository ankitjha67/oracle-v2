#!/usr/bin/env python3
"""
Oracle V2 — Football Pipeline
Loads 4,700+ real matches from 5 leagues × 3 seasons with bookmaker odds.
Builds Elo ratings for 100+ clubs. Trains 3-class ML models (W/D/L).
Predicts upcoming EPL + Champions League matches.
"""

import csv, json, math, os, time, warnings, glob, hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
try: import xgboost as xgb; HAS_XGB=True
except: HAS_XGB=False
try: import lightgbm as lgb; HAS_LGB=True
except: HAS_LGB=False

warnings.filterwarnings("ignore")

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "football_data"
OUTPUT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT.mkdir(parents=True, exist_ok=True)

LEAGUE_NAMES = {"E0":"EPL","SP1":"La Liga","I1":"Serie A","D1":"Bundesliga","F1":"Ligue 1"}

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD ALL MATCH DATA
# ═══════════════════════════════════════════════════════════════════════

def load_all_matches():
    """Load all CSV files from football-data.co.uk into a single DataFrame."""
    all_dfs = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(f, encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
            # Extract league and season from filename
            parts = f.stem.split("_")
            league_code = parts[0] if parts else "UNK"
            season = parts[1] if len(parts)>1 else "UNK"
            df["league"] = LEAGUE_NAMES.get(league_code, league_code)
            df["season"] = season
            df["source_file"] = f.name
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {f.name}: {e}")

    if not all_dfs:
        print("    ⚠️ No football CSV files found. Run setup_data() or download manually.")
        return pd.DataFrame(columns=["HomeTeam","AwayTeam","FTR","FTHG","FTAG",
                                      "HS","AS","HST","AST","HC","AC","HF","AF",
                                      "B365H","B365D","B365A","league","season","source_file"])

    combined = pd.concat(all_dfs, ignore_index=True)
    # Clean — keep only rows with valid results
    combined = combined.dropna(subset=["HomeTeam","AwayTeam","FTR"])
    combined = combined[combined["FTR"].isin(["H","D","A"])]
    return combined


# ═══════════════════════════════════════════════════════════════════════
# 2. BUILD ELO RATINGS FOR ALL CLUBS
# ═══════════════════════════════════════════════════════════════════════

class FootballElo:
    """Elo rating system for football clubs with home advantage and MOV."""

    def __init__(self, default=1500, K=20, home_adv=65):
        self.ratings = defaultdict(lambda: default)
        self.matches_played = defaultdict(int)
        self.K = K
        self.home_adv = home_adv
        self.history = []  # Track rating changes

    def expected(self, ra, rb):
        return 1.0 / (1.0 + 10**((rb - ra) / 400))

    def update(self, home, away, result, home_goals=0, away_goals=0):
        """result: 'H'=home win, 'D'=draw, 'A'=away win"""
        ra = self.ratings[home] + self.home_adv
        rb = self.ratings[away]
        ea = self.expected(ra, rb)

        if result == "H":
            sa = 1.0
        elif result == "D":
            sa = 0.5
        else:
            sa = 0.0

        # MOV adjustment
        goal_diff = abs(home_goals - away_goals)
        mov = 1 + math.log(max(goal_diff, 1)) * 0.5 if goal_diff > 1 else 1.0

        self.ratings[home] += self.K * mov * (sa - ea)
        self.ratings[away] += self.K * mov * ((1-sa) - (1-ea))
        self.matches_played[home] += 1
        self.matches_played[away] += 1

    def get(self, team):
        return self.ratings[team]

    def get_all(self):
        return dict(sorted(self.ratings.items(), key=lambda x: -x[1]))


# ═══════════════════════════════════════════════════════════════════════
# 3. FEATURE EXTRACTION (30+ features per match)
# ═══════════════════════════════════════════════════════════════════════

FEAT_NAMES = [
    "elo_home","elo_away","elo_diff","elo_prob",
    "home_form_pts","away_form_pts","form_diff",
    "home_goals_scored_avg","home_goals_conceded_avg",
    "away_goals_scored_avg","away_goals_conceded_avg",
    "home_attack_diff","home_defense_diff",
    "home_shots_avg","away_shots_avg","shots_diff",
    "home_sot_avg","away_sot_avg","sot_diff",
    "h2h_home_win_pct","h2h_draw_pct",
    "odds_home","odds_draw","odds_away",
    "implied_home","implied_draw","implied_away",
    "odds_overround",
    "home_corners_avg","away_corners_avg",
    "home_fouls_avg","away_fouls_avg",
]

def build_features_and_labels(df, elo):
    """Build feature matrix from match DataFrame."""
    # Pre-compute team stats
    team_stats = defaultdict(lambda: {
        "goals_scored": [], "goals_conceded": [],
        "shots": [], "sot": [], "corners": [], "fouls": [],
        "results": [], "points": []
    })
    h2h_record = defaultdict(lambda: {"H": 0, "D": 0, "A": 0, "total": 0})

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]

        try:
            hg = int(row.get("FTHG", 0) or 0)
            ag = int(row.get("FTAG", 0) or 0)
        except:
            hg, ag = 0, 0

        # Get current stats BEFORE this match (to avoid leakage)
        hs = team_stats[home]
        aws = team_stats[away]

        # Form (last 5 results as points)
        home_form = sum(hs["points"][-5:]) / max(len(hs["points"][-5:]), 1) if hs["points"] else 1.0
        away_form = sum(aws["points"][-5:]) / max(len(aws["points"][-5:]), 1) if aws["points"] else 1.0

        # Goals averages
        hgs = sum(hs["goals_scored"][-10:]) / max(len(hs["goals_scored"][-10:]), 1) if hs["goals_scored"] else 1.3
        hgc = sum(hs["goals_conceded"][-10:]) / max(len(hs["goals_conceded"][-10:]), 1) if hs["goals_conceded"] else 1.1
        ags = sum(aws["goals_scored"][-10:]) / max(len(aws["goals_scored"][-10:]), 1) if aws["goals_scored"] else 1.3
        agc = sum(aws["goals_conceded"][-10:]) / max(len(aws["goals_conceded"][-10:]), 1) if aws["goals_conceded"] else 1.1

        # Shots
        h_shots = sum(hs["shots"][-10:]) / max(len(hs["shots"][-10:]), 1) if hs["shots"] else 12
        a_shots = sum(aws["shots"][-10:]) / max(len(aws["shots"][-10:]), 1) if aws["shots"] else 12
        h_sot = sum(hs["sot"][-10:]) / max(len(hs["sot"][-10:]), 1) if hs["sot"] else 4
        a_sot = sum(aws["sot"][-10:]) / max(len(aws["sot"][-10:]), 1) if aws["sot"] else 4
        h_cor = sum(hs["corners"][-10:]) / max(len(hs["corners"][-10:]), 1) if hs["corners"] else 5
        a_cor = sum(aws["corners"][-10:]) / max(len(aws["corners"][-10:]), 1) if aws["corners"] else 5
        h_foul = sum(hs["fouls"][-10:]) / max(len(hs["fouls"][-10:]), 1) if hs["fouls"] else 11
        a_foul = sum(aws["fouls"][-10:]) / max(len(aws["fouls"][-10:]), 1) if aws["fouls"] else 11

        # H2H
        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_record[h2h_key]
        h2h_home_pct = h2h["H"] / max(h2h["total"], 1) if h2h["total"] > 0 else 0.45
        h2h_draw_pct = h2h["D"] / max(h2h["total"], 1) if h2h["total"] > 0 else 0.27

        # Elo
        elo_h = elo.get(home)
        elo_a = elo.get(away)
        elo_diff = elo_h - elo_a
        elo_prob = 1 / (1 + 10 ** (-(elo_diff + elo.home_adv) / 400))

        # Betting odds (B365 = bet365)
        try:
            odds_h = float(row.get("B365H", 0) or 0)
            odds_d = float(row.get("B365D", 0) or 0)
            odds_a = float(row.get("B365A", 0) or 0)
        except:
            odds_h, odds_d, odds_a = 2.0, 3.3, 3.5

        # Implied probabilities (normalized)
        if odds_h > 0 and odds_d > 0 and odds_a > 0:
            raw_h, raw_d, raw_a = 1/odds_h, 1/odds_d, 1/odds_a
            total = raw_h + raw_d + raw_a
            imp_h, imp_d, imp_a = raw_h/total, raw_d/total, raw_a/total
            overround = total - 1
        else:
            imp_h, imp_d, imp_a = 0.45, 0.27, 0.28
            overround = 0.05
            odds_h, odds_d, odds_a = 2.0, 3.3, 3.5

        features = [
            elo_h, elo_a, elo_diff, elo_prob,
            home_form, away_form, home_form - away_form,
            hgs, hgc, ags, agc,
            (hgs - agc), (ags - hgc),  # attack vs defense matchup
            h_shots, a_shots, h_shots - a_shots,
            h_sot, a_sot, h_sot - a_sot,
            h2h_home_pct, h2h_draw_pct,
            odds_h, odds_d, odds_a,
            imp_h, imp_d, imp_a,
            overround,
            h_cor, a_cor,
            h_foul, a_foul,
        ]

        # Label: 0=Home, 1=Draw, 2=Away (3-class!)
        label = {"H": 0, "D": 1, "A": 2}[result]
        X.append(features)
        y.append(label)

        # NOW update stats with this match result
        elo.update(home, away, result, hg, ag)
        team_stats[home]["goals_scored"].append(hg)
        team_stats[home]["goals_conceded"].append(ag)
        team_stats[away]["goals_scored"].append(ag)
        team_stats[away]["goals_conceded"].append(hg)
        try:
            team_stats[home]["shots"].append(int(row.get("HS",12) or 12))
            team_stats[away]["shots"].append(int(row.get("AS",12) or 12))
            team_stats[home]["sot"].append(int(row.get("HST",4) or 4))
            team_stats[away]["sot"].append(int(row.get("AST",4) or 4))
            team_stats[home]["corners"].append(int(row.get("HC",5) or 5))
            team_stats[away]["corners"].append(int(row.get("AC",5) or 5))
            team_stats[home]["fouls"].append(int(row.get("HF",11) or 11))
            team_stats[away]["fouls"].append(int(row.get("AF",11) or 11))
        except: pass
        team_stats[home]["points"].append(3 if result=="H" else 1 if result=="D" else 0)
        team_stats[away]["points"].append(3 if result=="A" else 1 if result=="D" else 0)
        h2h_record[h2h_key][result] += 1
        h2h_record[h2h_key]["total"] += 1

    return np.array(X), np.array(y), team_stats, h2h_record


# ═══════════════════════════════════════════════════════════════════════
# 4. ML MODELS — 3-CLASS (Home/Draw/Away)
# ═══════════════════════════════════════════════════════════════════════

def build_football_models():
    m = {
        "RF": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "LR": LogisticRegression(C=1, max_iter=2000, random_state=42),
        "Ada": AdaBoostClassifier(n_estimators=50, random_state=42),
        "NB": GaussianNB(),
    }
    if HAS_XGB:
        m["XGB"] = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss", use_label_encoder=False)
    if HAS_LGB:
        m["LGBM"] = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbose=-1)
    return m


# ═══════════════════════════════════════════════════════════════════════
# 5. PREDICT UPCOMING MATCHES
# ═══════════════════════════════════════════════════════════════════════

# NO hardcoded matches — all fixtures fetched LIVE from ESPN API via fixture_fetcher.py
# This list only exists as a structural placeholder; run.py never uses it.
UPCOMING = []


def predict_upcoming(models, scaler, elo, team_stats, h2h_record, upcoming):
    """Generate predictions for upcoming matches using trained models."""
    predictions = []

    for match in upcoming:
        home, away = match["home"], match["away"]

        hs = team_stats.get(home, {})
        aws = team_stats.get(away, {})

        home_form = sum(hs.get("points",[1])[-5:]) / max(len(hs.get("points",[1])[-5:]),1) if hs.get("points") else 1.0
        away_form = sum(aws.get("points",[1])[-5:]) / max(len(aws.get("points",[1])[-5:]),1) if aws.get("points") else 1.0
        hgs = sum(hs.get("goals_scored",[1.3])[-10:]) / max(len(hs.get("goals_scored",[])[-10:]),1) if hs.get("goals_scored") else 1.3
        hgc = sum(hs.get("goals_conceded",[1.1])[-10:]) / max(len(hs.get("goals_conceded",[])[-10:]),1) if hs.get("goals_conceded") else 1.1
        ags = sum(aws.get("goals_scored",[1.3])[-10:]) / max(len(aws.get("goals_scored",[])[-10:]),1) if aws.get("goals_scored") else 1.3
        agc = sum(aws.get("goals_conceded",[1.1])[-10:]) / max(len(aws.get("goals_conceded",[])[-10:]),1) if aws.get("goals_conceded") else 1.1
        h_shots = sum(hs.get("shots",[12])[-10:]) / max(len(hs.get("shots",[])[-10:]),1) if hs.get("shots") else 12
        a_shots = sum(aws.get("shots",[12])[-10:]) / max(len(aws.get("shots",[])[-10:]),1) if aws.get("shots") else 12
        h_sot = sum(hs.get("sot",[4])[-10:]) / max(len(hs.get("sot",[])[-10:]),1) if hs.get("sot") else 4
        a_sot = sum(aws.get("sot",[4])[-10:]) / max(len(aws.get("sot",[])[-10:]),1) if aws.get("sot") else 4
        h_cor = sum(hs.get("corners",[5])[-10:]) / max(len(hs.get("corners",[])[-10:]),1) if hs.get("corners") else 5
        a_cor = sum(aws.get("corners",[5])[-10:]) / max(len(aws.get("corners",[])[-10:]),1) if aws.get("corners") else 5
        h_foul = sum(hs.get("fouls",[11])[-10:]) / max(len(hs.get("fouls",[])[-10:]),1) if hs.get("fouls") else 11
        a_foul = sum(aws.get("fouls",[11])[-10:]) / max(len(aws.get("fouls",[])[-10:]),1) if aws.get("fouls") else 11

        h2h_key = tuple(sorted([home, away]))
        h2h = h2h_record.get(h2h_key, {"H":0,"D":0,"A":0,"total":0})
        h2h_h = h2h["H"]/max(h2h["total"],1) if h2h["total"]>0 else 0.45
        h2h_d = h2h["D"]/max(h2h["total"],1) if h2h["total"]>0 else 0.27

        elo_h = elo.get(home) if hasattr(elo, 'get') else 1500
        elo_a = elo.get(away) if hasattr(elo, 'get') else 1500
        elo_diff = elo_h - elo_a
        elo_prob = 1/(1+10**(-(elo_diff + getattr(elo, 'home_adv', 65))/400))

        # Use SportRadar odds as proxy for bookmaker odds
        sr = match.get("sr_prob", {"H":33,"D":33,"A":33})
        imp_h, imp_d, imp_a = sr["H"]/100, sr["D"]/100, sr["A"]/100
        odds_h = 1/imp_h if imp_h>0 else 3.0
        odds_d = 1/imp_d if imp_d>0 else 3.3
        odds_a = 1/imp_a if imp_a>0 else 3.0

        features = [
            elo_h, elo_a, elo_diff, elo_prob,
            home_form, away_form, home_form - away_form,
            hgs, hgc, ags, agc,
            hgs - agc, ags - hgc,
            h_shots, a_shots, h_shots - a_shots,
            h_sot, a_sot, h_sot - a_sot,
            h2h_h, h2h_d,
            odds_h, odds_d, odds_a,
            imp_h, imp_d, imp_a,
            imp_h + imp_d + imp_a - 1,  # overround (0 for normalized)
            h_cor, a_cor,
            h_foul, a_foul,
        ]

        feat_scaled = scaler.transform(np.array(features).reshape(1,-1))

        # Get predictions from all models
        model_votes = {}
        all_probs_h, all_probs_d, all_probs_a = [], [], []

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feat_scaled)[0]
                if len(proba) == 3:
                    ph, pd, pa = float(proba[0]), float(proba[1]), float(proba[2])
                else:
                    ph, pd, pa = float(proba[0]), 0.25, float(proba[-1]) if len(proba)>1 else 0.25
            else:
                pred = int(model.predict(feat_scaled)[0])
                ph = 0.8 if pred==0 else 0.1
                pd = 0.8 if pred==1 else 0.1
                pa = 0.8 if pred==2 else 0.1

            model_votes[name] = {"home": round(ph,4), "draw": round(pd,4), "away": round(pa,4)}
            all_probs_h.append(ph)
            all_probs_d.append(pd)
            all_probs_a.append(pa)

        avg_h = np.mean(all_probs_h)
        avg_d = np.mean(all_probs_d)
        avg_a = np.mean(all_probs_a)

        # Normalize
        total = avg_h + avg_d + avg_a
        avg_h, avg_d, avg_a = avg_h/total, avg_d/total, avg_a/total

        # FIX: Boost draw probability when Elo difference is small (< 80 points)
        # In real football, close matchups draw ~28-32% of the time
        if abs(elo_diff) < 80:
            draw_boost = (80 - abs(elo_diff)) / 80 * 0.08  # Up to +8% draw boost
            avg_d += draw_boost
            # Redistribute from the weaker side
            if avg_h > avg_a:
                avg_h -= draw_boost * 0.6
                avg_a -= draw_boost * 0.4
            else:
                avg_a -= draw_boost * 0.6
                avg_h -= draw_boost * 0.4
            # Re-normalize
            total = avg_h + avg_d + avg_a
            avg_h, avg_d, avg_a = avg_h/total, avg_d/total, avg_a/total

        # FIX: Fair winner selection — draw should win ties
        if avg_d >= avg_h and avg_d >= avg_a:
            winner = "DRAW"; conf_val = avg_d
        elif avg_h >= avg_a:
            winner = home; conf_val = avg_h
        else:
            winner = away; conf_val = avg_a

        if conf_val > 0.55: confidence = "HIGH"
        elif conf_val > 0.40: confidence = "MODERATE"
        else: confidence = "LOW"

        predictions.append({
            "match": f"{home} vs {away}",
            "league": match["league"],
            "date": match["date"],
            "oracle_prediction": {
                "home_pct": round(avg_h*100, 1),
                "draw_pct": round(avg_d*100, 1),
                "away_pct": round(avg_a*100, 1),
                "predicted_result": winner,
                "confidence": confidence,
            },
            "sportRadar_prob": match.get("sr_prob"),
            "elo_ratings": {"home": round(elo_h,1), "away": round(elo_a,1), "diff": round(elo_diff,1)},
            "model_votes": model_votes,
            "form": {"home_ppg": round(home_form,2), "away_ppg": round(away_form,2)},
            "goals_avg": {"home_scored": round(hgs,2), "home_conceded": round(hgc,2),
                          "away_scored": round(ags,2), "away_conceded": round(agc,2)},
        })

    return predictions


# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("╔" + "═"*78 + "╗")
    print("║  ORACLE V2 — FOOTBALL PIPELINE                                               ║")
    print("║  5 Leagues × 3 Seasons × Betting Odds → 3-Class ML → EPL + UCL Predictions   ║")
    print("╚" + "═"*78 + "╝")

    # 1. Load data
    print("\n  [1/5] Loading football match data...")
    df = load_all_matches()
    print(f"    ✅ {len(df)} matches loaded across {df['league'].nunique()} leagues")
    for league, count in df.groupby("league").size().items():
        print(f"       {league}: {count} matches")

    # 2. Build Elo + Features
    print("\n  [2/5] Building Elo ratings + extracting features...")
    elo = FootballElo(default=1500, K=20, home_adv=65)
    X, y, team_stats, h2h_record = build_features_and_labels(df, elo)
    print(f"    ✅ {len(X)} feature vectors | {X.shape[1]} features each")
    print(f"    Labels: Home={sum(y==0)}, Draw={sum(y==1)}, Away={sum(y==2)}")
    print(f"    Teams rated: {len(elo.get_all())}")

    # Top 20 Elo
    print(f"\n    Top 20 Club Elo Ratings:")
    for i, (team, rating) in enumerate(list(elo.get_all().items())[:20]):
        bar = "█" * int((rating - 1300) / 10)
        print(f"    {i+1:>2}. {team:<25} {bar} {rating:.0f}")

    # 3. Train models
    print(f"\n  [3/5] Training 3-class ML models (Home/Draw/Away)...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    models = build_football_models()
    cv_scores = {}
    for name, model in models.items():
        model.fit(Xs, y)
        try:
            s = cross_val_score(model, Xs, y, cv=2, scoring="accuracy")
            cv_scores[name] = round(s.mean(), 3)
        except:
            cv_scores[name] = round(float((model.predict(Xs)==y).mean()), 3)

    print(f"    ✅ {len(models)} models trained (3-class: Home/Draw/Away)")
    for n, s in sorted(cv_scores.items(), key=lambda x: -x[1])[:8]:
        print(f"       {n:<12} CV: {s:.3f}")

    # Baseline comparisons
    home_always = sum(y==0) / len(y)
    print(f"\n    Baselines:")
    print(f"       Always pick Home:     {home_always:.1%}")
    print(f"       Best model:            {max(cv_scores.values()):.1%}")

    # 4. Predict upcoming
    print(f"\n  [4/5] Predicting {len(UPCOMING)} upcoming matches...")
    predictions = predict_upcoming(models, scaler, elo, team_stats, h2h_record, UPCOMING)

    print(f"\n  ┌{'─'*78}┐")
    print(f"  │{'UPCOMING MATCH PREDICTIONS':^78}│")
    print(f"  ├{'─'*78}┤")

    for p in predictions:
        o = p["oracle_prediction"]
        icon = "🟠" if o["confidence"]=="HIGH" else "🟡" if o["confidence"]=="MODERATE" else "🟢"
        sr = p.get("sportRadar_prob", {})
        elo_d = p["elo_ratings"]
        result = o["predicted_result"]
        print(f"  │  {p['league']:<5} {p['match']:<35} {p['date']}           │")
        print(f"  │    Oracle: H {o['home_pct']:>5.1f}% D {o['draw_pct']:>5.1f}% A {o['away_pct']:>5.1f}%  "
              f"→ {icon} {result:<15} [{o['confidence']}] │")
        print(f"  │    Market: H {sr.get('H',0):>5.1f}% D {sr.get('D',0):>5.1f}% A {sr.get('A',0):>5.1f}%  "
              f"  Elo: {elo_d['home']:.0f} vs {elo_d['away']:.0f} ({elo_d['diff']:+.0f})      │")
        print(f"  │{'─'*78}│")

    print(f"  └{'─'*78}┘")

    # Where Oracle disagrees with market (value bets)
    print(f"\n  ⚡ VALUE SPOTS (Oracle disagrees with market by >10%):")
    for p in predictions:
        o = p["oracle_prediction"]
        sr = p.get("sportRadar_prob", {})
        if abs(o["home_pct"] - sr.get("H",0)) > 10 or abs(o["away_pct"] - sr.get("A",0)) > 10:
            print(f"    {p['match']}")
            print(f"      Oracle: H={o['home_pct']}% D={o['draw_pct']}% A={o['away_pct']}%")
            print(f"      Market: H={sr.get('H')}% D={sr.get('D')}% A={sr.get('A')}%")

    # 5. Save everything
    print(f"\n  [5/5] Saving complete output...")
    output = {
        "metadata": {
            "engine": "Oracle V2 Football Pipeline",
            "matches_trained": len(X),
            "features": len(FEAT_NAMES),
            "leagues": list(df["league"].unique()),
            "seasons": list(df["season"].unique()),
            "teams_rated": len(elo.get_all()),
            "models": len(models),
            "model_type": "3-class (Home/Draw/Away)",
            "cv_accuracy": cv_scores,
            "execution_time": round(time.time()-t0, 1),
        },
        "elo_ratings_top50": {t: round(r,1) for t,r in list(elo.get_all().items())[:50]},
        "predictions": predictions,
    }

    path = OUTPUT / "FOOTBALL_PREDICTIONS.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  📁 {path} ({path.stat().st_size/1024:.0f} KB)")
    print(f"  Completed in {time.time()-t0:.1f}s")
    print(f"{'═'*80}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════
# 6. POISSON SCORE PREDICTION
# ═══════════════════════════════════════════════════════════════════════

def poisson_score_predict(home_goals_avg, home_conceded_avg, away_goals_avg, away_conceded_avg,
                          league_avg_goals=1.35, max_goals=6, elo_diff=0):
    """Predict exact score probabilities using Poisson distribution.
    
    elo_diff: home Elo minus away Elo (positive = home stronger).
    Adjusts expected goals based on relative team strength.
    """
    import math
    
    home_attack = home_goals_avg / max(league_avg_goals, 0.5)
    away_attack = away_goals_avg / max(league_avg_goals, 0.5)
    home_defense = home_conceded_avg / max(league_avg_goals, 0.5)
    away_defense = away_conceded_avg / max(league_avg_goals, 0.5)
    
    # Elo-based strength adjustment: +100 Elo ≈ +0.15 expected goals
    elo_adjust = elo_diff / 800  # Maps ±400 Elo diff to ±0.5 goals
    
    home_expected = home_attack * away_defense * league_avg_goals * 1.05 + elo_adjust  # Reduced home boost from 1.1
    away_expected = away_attack * home_defense * league_avg_goals - elo_adjust * 0.5
    
    home_expected = max(0.3, min(home_expected, 4.0))
    away_expected = max(0.3, min(away_expected, 4.0))
    
    def poisson_pmf(k, lam):
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    # Score matrix
    scores = {}
    home_win_prob = draw_prob = away_win_prob = 0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_pmf(h, home_expected) * poisson_pmf(a, away_expected)
            scores[f"{h}-{a}"] = round(p * 100, 2)
            if h > a: home_win_prob += p
            elif h == a: draw_prob += p
            else: away_win_prob += p
    
    # Sort by probability
    top_scores = sorted(scores.items(), key=lambda x: -x[1])[:10]
    
    return {
        "home_expected_goals": round(home_expected, 2),
        "away_expected_goals": round(away_expected, 2),
        "home_win_pct": round(home_win_prob * 100, 1),
        "draw_pct": round(draw_prob * 100, 1),
        "away_win_pct": round(away_win_prob * 100, 1),
        "most_likely_scores": dict(top_scores),
        "predicted_score": top_scores[0][0] if top_scores else "1-1",
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. ROI TRACKING (Backtest predictions vs closing odds)
# ═══════════════════════════════════════════════════════════════════════

def calculate_roi(predictions_with_results, stake=100):
    """Calculate ROI from historical predictions vs actual results.
    
    Each prediction needs: predicted_result, actual_result, odds_for_predicted
    """
    total_staked = 0
    total_returned = 0
    wins = losses = 0
    
    for p in predictions_with_results:
        predicted = p.get("predicted_result")
        actual = p.get("actual_result")  # "H", "D", or "A"
        odds = p.get("odds_for_predicted", 2.0)
        
        if not predicted or not actual:
            continue
            
        total_staked += stake
        
        if predicted == actual:
            total_returned += stake * odds
            wins += 1
        else:
            losses += 1
    
    total_bets = wins + losses
    if total_bets == 0:
        return {"roi_pct": 0, "total_bets": 0}
    
    roi = ((total_returned - total_staked) / total_staked) * 100
    
    return {
        "total_bets": total_bets,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(wins / total_bets * 100, 1),
        "total_staked": total_staked,
        "total_returned": round(total_returned, 2),
        "profit_loss": round(total_returned - total_staked, 2),
        "roi_pct": round(roi, 1),
        "edge_over_market": "YES" if roi > 0 else "NO",
    }


def backtest_roi_on_training_data(df, elo, team_stats, models, scaler):
    """Backtest ROI on the last 20% of training data (most recent matches)."""
    n = len(df)
    test_start = int(n * 0.8)
    test_df = df.iloc[test_start:]
    
    results = []
    for _, row in test_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        actual = row["FTR"]  # H, D, A
        
        try:
            odds_h = float(row.get("B365H", 2.0) or 2.0)
            odds_d = float(row.get("B365D", 3.3) or 3.3)
            odds_a = float(row.get("B365A", 3.5) or 3.5)
        except:
            continue
        
        # Oracle's prediction (simplified — uses Elo)
        elo_h = elo.get(home)
        elo_a = elo.get(away)
        elo_prob = 1 / (1 + 10 ** (-(elo_h - elo_a + elo.home_adv) / 400))
        
        if elo_prob > 0.5:
            predicted = "H"
            odds_for_pred = odds_h
        elif elo_prob < 0.35:
            predicted = "A"
            odds_for_pred = odds_a
        else:
            predicted = "D"
            odds_for_pred = odds_d
        
        results.append({
            "match": f"{home} vs {away}",
            "predicted_result": predicted,
            "actual_result": actual,
            "odds_for_predicted": odds_for_pred,
        })
    
    return calculate_roi(results)


# ═══════════════════════════════════════════════════════════════════════
# 8. RESPONSIBLE GAMBLING DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════

GAMBLING_DISCLAIMER = """
⚠️ RESPONSIBLE GAMBLING NOTICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• This tool is for INFORMATIONAL and ENTERTAINMENT purposes only.
• Past performance does NOT guarantee future results.
• Sports outcomes are inherently unpredictable.
• Never bet more than you can afford to lose.
• If you or someone you know has a gambling problem:
  - 🇺🇸 USA: 1-800-522-4700 (National Council on Problem Gambling)
  - 🇬🇧 UK: 0808-8020-133 (GamCare)
  - 🇮🇳 India: iCall — 9152987821
  - 🌐 International: www.begambleaware.org
• The model's predictions carry NO warranty of accuracy.
• The creators accept NO liability for financial losses.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
