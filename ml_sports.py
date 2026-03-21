"""
Oracle V2 — Universal ML Pipeline for ESPN Sports
Adds ML-based predictions to NBA, NHL, MLB, NFL, UFC, MLS, College, Rugby, etc.

Features are derived from match history (scores, dates, home/away) collected
by multi_sport.py. No external data sources required beyond ESPN.

Feature set (~20 features per matchup):
- Rolling win rate (last 5, 10, 20 games)
- Rolling margin of victory
- Win/loss streak momentum
- Rest days since last game
- Home/away performance splits
- Strength of schedule (opponent Elo average)
- Head-to-head record
- Elo ratings + Elo probability
- Recent form velocity (trend direction)
- Consistency (variance in margins)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

logger = logging.getLogger("oracle.ml_sports")

FEATURE_NAMES = [
    "elo_a",
    "elo_b",
    "elo_diff",
    "elo_prob",
    "win_rate_5_a",
    "win_rate_5_b",
    "win_rate_10_a",
    "win_rate_10_b",
    "win_rate_diff_5",
    "win_rate_diff_10",
    "avg_margin_5_a",
    "avg_margin_5_b",
    "margin_diff",
    "streak_a",
    "streak_b",
    "streak_diff",
    "rest_days_a",
    "rest_days_b",
    "rest_diff",
    "home_win_rate_a",
    "away_win_rate_b",
    "h2h_win_rate_a",
    "sos_a",
    "sos_b",
    "sos_diff",
    "consistency_a",
    "consistency_b",
    "form_velocity_a",
    "form_velocity_b",
]


class TeamStats:
    """Tracks rolling statistics for a team from match history."""

    __slots__ = ("away_results", "dates", "home_results", "margins", "opponents", "results")

    def __init__(self):
        self.results: list[int] = []  # 1=win, 0=loss
        self.margins: list[float] = []
        self.dates: list[str] = []
        self.home_results: list[int] = []
        self.away_results: list[int] = []
        self.opponents: list[tuple[str, float]] = []  # (opponent, opponent_elo)

    def add(self, won: bool, margin: float, date: str, is_home: bool, opponent: str, opp_elo: float):
        self.results.append(1 if won else 0)
        self.margins.append(margin if won else -margin)
        self.dates.append(date)
        if is_home:
            self.home_results.append(1 if won else 0)
        else:
            self.away_results.append(1 if won else 0)
        self.opponents.append((opponent, opp_elo))

    def win_rate(self, n: int) -> float:
        recent = self.results[-n:]
        return sum(recent) / len(recent) if recent else 0.5

    def avg_margin(self, n: int) -> float:
        recent = self.margins[-n:]
        return sum(recent) / len(recent) if recent else 0.0

    def streak(self) -> int:
        """Positive = win streak, negative = loss streak."""
        if not self.results:
            return 0
        s = 0
        last = self.results[-1]
        for r in reversed(self.results):
            if r == last:
                s += 1
            else:
                break
        return s if last == 1 else -s

    def rest_days(self, current_date: str) -> float:
        """Days since last game."""
        if not self.dates:
            return 7.0
        try:
            last = datetime.strptime(self.dates[-1], "%Y-%m-%d")
            now = datetime.strptime(current_date, "%Y-%m-%d")
            return max((now - last).days, 0)
        except (ValueError, TypeError):
            return 7.0

    def home_win_rate(self) -> float:
        return sum(self.home_results) / len(self.home_results) if self.home_results else 0.5

    def away_win_rate(self) -> float:
        return sum(self.away_results) / len(self.away_results) if self.away_results else 0.5

    def strength_of_schedule(self, n: int = 10) -> float:
        """Average Elo of recent opponents."""
        recent = self.opponents[-n:]
        if not recent:
            return 1500.0
        return sum(e for _, e in recent) / len(recent)

    def consistency(self, n: int = 10) -> float:
        """Std dev of recent margins — lower = more consistent."""
        recent = self.margins[-n:]
        if len(recent) < 2:
            return 0.0
        return float(np.std(recent))

    def form_velocity(self, n: int = 5) -> float:
        """Slope of recent win rate — positive = improving."""
        recent = self.results[-n:]
        if len(recent) < 3:
            return 0.0
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


def extract_features(
    team_a: str,
    team_b: str,
    elo_a: float,
    elo_b: float,
    home_team: str | None,
    match_date: str,
    stats: dict[str, TeamStats],
    h2h: dict[tuple[str, str], list[int]],
    home_adv: float,
) -> np.ndarray:
    """Extract ~29 ML features for a matchup from accumulated stats."""
    sa = stats.get(team_a, TeamStats())
    sb = stats.get(team_b, TeamStats())

    # Elo features
    ha = home_adv if home_team == team_a else (-home_adv if home_team == team_b else 0)
    elo_diff = (elo_a + ha) - elo_b
    elo_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    # Win rates
    wr5a, wr5b = sa.win_rate(5), sb.win_rate(5)
    wr10a, wr10b = sa.win_rate(10), sb.win_rate(10)

    # Margins
    am5a, am5b = sa.avg_margin(5), sb.avg_margin(5)

    # Streaks
    stk_a, stk_b = sa.streak(), sb.streak()

    # Rest
    rest_a = sa.rest_days(match_date)
    rest_b = sb.rest_days(match_date)

    # Home/away splits
    hwr_a = sa.home_win_rate()
    awr_b = sb.away_win_rate()

    # H2H
    h2h_key = (team_a, team_b)
    h2h_results = h2h.get(h2h_key, [])
    h2h_wr = sum(h2h_results) / len(h2h_results) if h2h_results else 0.5

    # Strength of schedule
    sos_a = sa.strength_of_schedule()
    sos_b = sb.strength_of_schedule()

    # Consistency & velocity
    cons_a = sa.consistency()
    cons_b = sb.consistency()
    vel_a = sa.form_velocity()
    vel_b = sb.form_velocity()

    return np.array(
        [
            elo_a / 2000,
            elo_b / 2000,
            elo_diff / 400,
            elo_prob,
            wr5a,
            wr5b,
            wr10a,
            wr10b,
            wr5a - wr5b,
            wr10a - wr10b,
            am5a / 20,
            am5b / 20,
            (am5a - am5b) / 20,
            stk_a / 10,
            stk_b / 10,
            (stk_a - stk_b) / 10,
            min(rest_a, 14) / 14,
            min(rest_b, 14) / 14,
            (rest_a - rest_b) / 14,
            hwr_a,
            awr_b,
            h2h_wr,
            sos_a / 2000,
            sos_b / 2000,
            (sos_a - sos_b) / 400,
            cons_a / 20,
            cons_b / 20,
            vel_a,
            vel_b,
        ]
    )


def _build_models() -> dict:
    """Build ML model ensemble for sport prediction."""
    models: dict = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    return models


class SportMLEngine:
    """Universal ML prediction engine for any ESPN-based sport.

    Collects match history, extracts features, trains ensemble models,
    and blends ML predictions with Elo ratings.
    """

    def __init__(self, sport_key: str, K: int = 25, home_adv: float = 50):
        self.sport_key = sport_key
        self.K = K
        self.home_adv = home_adv

        # Elo state
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.elo_matches: dict[str, int] = defaultdict(int)

        # ML state
        self.scaler = StandardScaler()
        self.models: dict = {}
        self.is_trained = False

        # History tracking
        self.stats: dict[str, TeamStats] = defaultdict(TeamStats)
        self.h2h: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.match_history: list[dict] = []

    def _elo_update(self, winner: str, loser: str, home_team: str | None, margin: float):
        """Update Elo ratings after a match."""
        ha = self.home_adv if home_team == winner else (-self.home_adv if home_team == loser else 0)
        ea = 1 / (1 + 10 ** ((self.elo[loser] - (self.elo[winner] + ha)) / 400))
        mov = 1 + math.log(max(margin, 1)) * 0.3 if margin > 0 else 1.0
        delta = self.K * mov * (1 - ea)
        self.elo[winner] += delta
        self.elo[loser] -= delta
        self.elo_matches[winner] += 1
        self.elo_matches[loser] += 1

    def add_result(self, team_a: str, team_b: str, score_a: int, score_b: int, home_team: str | None, date: str):
        """Process a completed match result."""
        if score_a == score_b:
            return

        winner = team_a if score_a > score_b else team_b
        loser = team_b if winner == team_a else team_a
        margin = abs(score_a - score_b)

        # Record Elo before update for feature extraction
        elo_a_pre = self.elo[team_a]
        elo_b_pre = self.elo[team_b]

        # Update Elo
        self._elo_update(winner, loser, home_team, margin)

        # Update team stats
        is_a_home = home_team == team_a
        is_b_home = home_team == team_b
        a_won = winner == team_a
        self.stats[team_a].add(a_won, margin, date, is_a_home, team_b, elo_b_pre)
        self.stats[team_b].add(not a_won, margin, date, is_b_home, team_a, elo_a_pre)

        # H2H tracking
        self.h2h[(team_a, team_b)].append(1 if a_won else 0)
        self.h2h[(team_b, team_a)].append(0 if a_won else 1)

        # Store for training
        self.match_history.append(
            {
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "home": home_team,
                "date": date,
                "margin": margin,
                "elo_a": elo_a_pre,
                "elo_b": elo_b_pre,
            }
        )

    def train(self, min_matches: int = 30) -> dict:
        """Train ML models on accumulated match history.

        Uses walk-forward: only features available before each match are used.
        Returns training stats.
        """
        if len(self.match_history) < min_matches:
            return {"error": f"Need {min_matches}+ matches, have {len(self.match_history)}"}

        # Rebuild features using walk-forward approach
        # We need to re-accumulate stats chronologically
        wf_stats: dict[str, TeamStats] = defaultdict(TeamStats)
        wf_h2h: dict[tuple[str, str], list[int]] = defaultdict(list)
        wf_elo: dict[str, float] = defaultdict(lambda: 1500.0)

        X, y = [], []
        skip_first = max(min_matches // 3, 10)  # Skip early matches with insufficient history

        for i, m in enumerate(self.match_history):
            ta, tb = m["team_a"], m["team_b"]
            a_won = m["winner"] == ta

            if i >= skip_first:
                features = extract_features(
                    ta,
                    tb,
                    wf_elo[ta],
                    wf_elo[tb],
                    m["home"],
                    m["date"],
                    wf_stats,
                    wf_h2h,
                    self.home_adv,
                )
                X.append(features)
                y.append(1 if a_won else 0)

            # Update walk-forward state
            margin = m["margin"]
            winner, loser = (ta, tb) if a_won else (tb, ta)
            ha = self.home_adv if m["home"] == winner else (-self.home_adv if m["home"] == loser else 0)
            ea = 1 / (1 + 10 ** ((wf_elo[loser] - (wf_elo[winner] + ha)) / 400))
            mov = 1 + math.log(max(margin, 1)) * 0.3 if margin > 0 else 1.0
            delta = self.K * mov * (1 - ea)
            wf_elo[winner] += delta
            wf_elo[loser] -= delta

            is_a_home = m["home"] == ta
            is_b_home = m["home"] == tb
            wf_stats[ta].add(a_won, margin, m["date"], is_a_home, tb, wf_elo[tb])
            wf_stats[tb].add(not a_won, margin, m["date"], is_b_home, ta, wf_elo[ta])
            wf_h2h[(ta, tb)].append(1 if a_won else 0)
            wf_h2h[(tb, ta)].append(0 if a_won else 1)

        X = np.array(X)
        y = np.array(y)

        if len(X) < 20:
            return {"error": f"Not enough valid training samples: {len(X)}"}

        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train models
        self.models = _build_models()
        cv_scores = {}
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                cv_scores[name] = name
            except Exception as e:
                logger.warning(f"Model {name} failed for {self.sport_key}: {e}")

        self.is_trained = len(self.models) > 0
        return {
            "sport": self.sport_key,
            "training_samples": len(X),
            "models_trained": len(self.models),
            "model_names": list(self.models.keys()),
        }

    def predict(self, team_a: str, team_b: str, home_team: str | None, match_date: str) -> dict:
        """Predict match outcome using ML ensemble + Elo blend.

        Returns prediction dict with probabilities and confidence.
        """
        elo_a = self.elo[team_a]
        elo_b = self.elo[team_b]

        # Elo-only prediction
        ha = self.home_adv if home_team == team_a else (-self.home_adv if home_team == team_b else 0)
        elo_prob = 1 / (1 + 10 ** ((elo_b - (elo_a + ha)) / 400))

        if not self.is_trained:
            # Fallback to pure Elo
            pa = round(elo_prob * 100, 1)
            pb = round((1 - elo_prob) * 100, 1)
            return self._format_prediction(team_a, team_b, pa, pb, method="elo")

        # ML prediction
        features = extract_features(
            team_a,
            team_b,
            elo_a,
            elo_b,
            home_team,
            match_date,
            self.stats,
            self.h2h,
            self.home_adv,
        )
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Ensemble prediction
        probs = []
        model_votes = {}
        for name, model in self.models.items():
            try:
                p = model.predict_proba(features_scaled)[0]
                prob_a = p[1] if len(p) > 1 else p[0]
                probs.append(prob_a)
                model_votes[name] = round(prob_a, 3)
            except Exception:
                continue

        if not probs:
            pa = round(elo_prob * 100, 1)
            pb = round((1 - elo_prob) * 100, 1)
            return self._format_prediction(team_a, team_b, pa, pb, method="elo")

        ml_prob = float(np.mean(probs))
        ml_std = float(np.std(probs))

        # Blend ML (70%) with Elo (30%) — Elo stabilizes when ML data is thin
        blended = 0.7 * ml_prob + 0.3 * elo_prob
        pa = round(blended * 100, 1)
        pb = round((1 - blended) * 100, 1)

        # Confidence based on agreement + distance from 50%
        diff = abs(pa - 50)
        agreement_penalty = max(0, ml_std - 0.15) * 30  # penalize when models disagree
        adjusted_diff = max(diff - agreement_penalty, 0)
        conf = "HIGH" if adjusted_diff > 15 else "MODERATE" if adjusted_diff > 7 else "LOW"

        return self._format_prediction(
            team_a,
            team_b,
            pa,
            pb,
            method="ml_ensemble",
            confidence=conf,
            model_votes=model_votes,
            ml_prob=round(ml_prob, 3),
            elo_prob=round(elo_prob, 3),
            model_std=round(ml_std, 3),
        )

    def _format_prediction(
        self,
        team_a: str,
        team_b: str,
        pa: float,
        pb: float,
        method: str = "elo",
        confidence: str | None = None,
        **extra,
    ) -> dict:
        if confidence is None:
            diff = abs(pa - 50)
            confidence = "HIGH" if diff > 15 else "MODERATE" if diff > 7 else "LOW"
        winner = team_a if pa > pb else team_b
        result = {
            "winner": winner,
            "prob_a": pa,
            "prob_b": pb,
            "confidence": confidence,
            "method": method,
        }
        result.update(extra)
        return result

    def rankings(self, n: int = 30) -> dict:
        return dict(sorted(self.elo.items(), key=lambda x: -x[1])[:n])
