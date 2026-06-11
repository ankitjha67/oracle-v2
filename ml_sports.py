"""
Oracle V2 — Universal ML Pipeline for ESPN Sports
Adds ML-based predictions to NBA, NHL, MLB, NFL, UFC, MLS, College, Rugby, etc.

Features are derived from match history (scores, dates, home/away) collected
by multi_sport.py, plus roster data from ESPN teams API.

Feature set (35 features per matchup):
- Elo ratings + probability (4)
- Rolling win rate at 5/10 game windows (6)
- Rolling margin of victory (3)
- Win/loss streak momentum (3)
- Rest days since last game (3)
- Home/away performance splits (2)
- Head-to-head record (1)
- Strength of schedule (3)
- Consistency & form velocity (4)
- Roster stability signals (6): new player count, avg experience, injury count,
  roster turnover rate, experience differential, injury differential

Models (up to 12):
- RandomForest, GradientBoosting, LogisticRegression, AdaBoost, SVM, Bagging,
  NaiveBayes, XGBoost*, LightGBM*, VotingClassifier, StackingMeta
  (* = if installed)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    # Elo (4)
    "elo_a",
    "elo_b",
    "elo_diff",
    "elo_prob",
    # Win rates (6)
    "win_rate_5_a",
    "win_rate_5_b",
    "win_rate_10_a",
    "win_rate_10_b",
    "win_rate_diff_5",
    "win_rate_diff_10",
    # Margins (3)
    "avg_margin_5_a",
    "avg_margin_5_b",
    "margin_diff",
    # Streaks (3)
    "streak_a",
    "streak_b",
    "streak_diff",
    # Rest (3)
    "rest_days_a",
    "rest_days_b",
    "rest_diff",
    # Home/Away (2)
    "home_win_rate_a",
    "away_win_rate_b",
    # H2H (1)
    "h2h_win_rate_a",
    # SOS (3)
    "sos_a",
    "sos_b",
    "sos_diff",
    # Form (4)
    "consistency_a",
    "consistency_b",
    "form_velocity_a",
    "form_velocity_b",
    # Roster (6)
    "roster_new_players_a",
    "roster_new_players_b",
    "roster_avg_exp_a",
    "roster_avg_exp_b",
    "roster_injured_a",
    "roster_injured_b",
]


# ═══════════════════════════════════════════════════════════════════════════
# ROSTER TRACKER — Detects player changes before every match
# ═══════════════════════════════════════════════════════════════════════════

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".cache"


class RosterTracker:
    """Tracks team rosters and detects new/changed players.

    Fetches current roster from ESPN teams API, compares against
    previous snapshot, and computes roster stability signals.
    """

    def __init__(self, espn_path: str):
        self.espn_path = espn_path
        self._rosters: dict[str, dict] = {}
        self._previous_rosters: dict[str, set[str]] = {}
        self._team_ids: dict[str, str] = {}
        self._loaded = False

    def _fetch_cached(self, url: str, cache_key: str, ttl_hours: float = 6) -> dict | None:
        CACHE_DIR.mkdir(exist_ok=True)
        cf = CACHE_DIR / f"{cache_key}.json"
        if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < ttl_hours:
            with open(cf) as f:
                return json.load(f)
        try:
            import requests

            r = requests.get(url, timeout=15, headers={"User-Agent": "OracleV2/3.1"})
            if r.status_code == 200:
                data = r.json()
                with open(cf, "w") as f:
                    json.dump(data, f)
                return data
        except Exception:
            pass
        return None

    def _load_team_ids(self):
        """Fetch team list from ESPN to map display names to IDs."""
        if self._loaded:
            return
        data = self._fetch_cached(
            f"{ESPN_BASE}/{self.espn_path}/teams",
            f"teams_{self.espn_path.replace('/', '_')}",
            ttl_hours=24,
        )
        if not data:
            self._loaded = True
            return
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for t in league.get("teams", []):
                    team = t.get("team", {})
                    name = team.get("displayName", "")
                    tid = team.get("id", "")
                    if name and tid:
                        self._team_ids[name] = tid
        self._loaded = True

    def fetch_roster(self, team_name: str) -> dict:
        """Fetch current roster for a team. Returns roster info dict.

        Returns:
            {
                "players": set of player names,
                "avg_experience": float (years),
                "injured_count": int,
                "total": int,
            }
        """
        self._load_team_ids()
        tid = self._team_ids.get(team_name, "")
        if not tid:
            return {"players": set(), "avg_experience": 0.0, "injured_count": 0, "total": 0}

        data = self._fetch_cached(
            f"{ESPN_BASE}/{self.espn_path}/teams/{tid}/roster",
            f"roster_{self.espn_path.replace('/', '_')}_{tid}",
            ttl_hours=6,
        )
        if not data:
            return {"players": set(), "avg_experience": 0.0, "injured_count": 0, "total": 0}

        players = set()
        experiences = []
        injured = 0

        athletes = data.get("athletes", [])
        for entry in athletes:
            # ESPN may return athletes as flat list or grouped by position
            if isinstance(entry, dict):
                items = entry.get("items", [])
                if items:
                    for a in items:
                        self._parse_athlete(a, players, experiences)
                        if a.get("injuries"):
                            injured += 1
                elif "fullName" in entry:
                    self._parse_athlete(entry, players, experiences)
                    if entry.get("injuries"):
                        injured += 1

        avg_exp = sum(experiences) / len(experiences) if experiences else 0.0
        return {
            "players": players,
            "avg_experience": round(avg_exp, 1),
            "injured_count": injured,
            "total": len(players),
        }

    def _parse_athlete(self, athlete: dict, players: set, experiences: list):
        name = athlete.get("fullName") or athlete.get("displayName", "")
        if name:
            players.add(name)
        exp = athlete.get("experience", {})
        if isinstance(exp, dict):
            years = exp.get("years", 0)
        else:
            years = 0
        experiences.append(years)

    def check_roster(self, team_name: str) -> dict:
        """Check roster and compute change signals vs previous snapshot.

        Returns:
            {
                "new_players": int,
                "departed_players": int,
                "turnover_rate": float,
                "avg_experience": float,
                "injured_count": int,
                "total": int,
            }
        """
        current = self.fetch_roster(team_name)
        current_players = current["players"]
        previous = self._previous_rosters.get(team_name, set())

        new_players = len(current_players - previous) if previous else 0
        departed = len(previous - current_players) if previous else 0
        turnover = (new_players + departed) / max(len(previous), 1) if previous else 0.0

        # Update snapshot
        self._previous_rosters[team_name] = current_players
        self._rosters[team_name] = current

        return {
            "new_players": new_players,
            "departed_players": departed,
            "turnover_rate": round(turnover, 3),
            "avg_experience": current["avg_experience"],
            "injured_count": current["injured_count"],
            "total": current["total"],
        }

    def get_cached_info(self, team_name: str) -> dict:
        """Get roster info without re-fetching (for feature extraction during training)."""
        if team_name in self._rosters:
            r = self._rosters[team_name]
            return {
                "new_players": 0,
                "avg_experience": r["avg_experience"],
                "injured_count": r["injured_count"],
                "total": r["total"],
            }
        return {"new_players": 0, "avg_experience": 0.0, "injured_count": 0, "total": 0}


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
    roster_a: dict | None = None,
    roster_b: dict | None = None,
) -> np.ndarray:
    """Extract 35 ML features for a matchup from accumulated stats + roster."""
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

    # Roster features (default to neutral if unavailable)
    ra = roster_a or {}
    rb = roster_b or {}
    new_a = ra.get("new_players", 0)
    new_b = rb.get("new_players", 0)
    exp_a = ra.get("avg_experience", 0.0)
    exp_b = rb.get("avg_experience", 0.0)
    inj_a = ra.get("injured_count", 0)
    inj_b = rb.get("injured_count", 0)

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
            # Roster features
            min(new_a, 5) / 5,
            min(new_b, 5) / 5,
            min(exp_a, 15) / 15,
            min(exp_b, 15) / 15,
            min(inj_a, 10) / 10,
            min(inj_b, 10) / 10,
        ]
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL ENSEMBLE — 12 models matching cricket engine coverage
# ═══════════════════════════════════════════════════════════════════════════


def _build_models() -> dict:
    """Build full ML model ensemble for sport prediction.

    Up to 12 models: RF, GBM, LR, AdaBoost, SVM, Bagging, NaiveBayes,
    XGBoost*, LightGBM*, plus VotingClassifier and Stacking meta-learner.
    """
    models: dict = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42
        ),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42),
        "Bagging": BaggingClassifier(n_estimators=100, max_samples=0.8, random_state=42),
        "NaiveBayes": GaussianNB(),
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


def _build_super_ensemble(base_models: dict) -> VotingClassifier | None:
    """Build a VotingClassifier from trained base models."""
    estimators = [
        (name.replace(" ", "_")[:20], model) for name, model in base_models.items() if hasattr(model, "predict_proba")
    ]
    if len(estimators) < 2:
        return None
    return VotingClassifier(estimators=estimators, voting="soft")


class SportMLEngine:
    """Universal ML prediction engine for any ESPN-based sport.

    Collects match history, extracts features, trains ensemble models,
    and blends ML predictions with Elo ratings. Includes roster change
    detection for prediction-time adjustments.
    """

    def __init__(self, sport_key: str, K: int = 25, home_adv: float = 50, espn_path: str = ""):
        self.sport_key = sport_key
        self.K = K
        self.home_adv = home_adv

        # Elo state
        self.elo: dict[str, float] = defaultdict(lambda: 1500.0)
        self.elo_matches: dict[str, int] = defaultdict(int)

        # ML state
        self.scaler = StandardScaler()
        self.models: dict = {}
        self.meta_learner: LogisticRegression | None = None
        self.is_trained = False

        # History tracking
        self.stats: dict[str, TeamStats] = defaultdict(TeamStats)
        self.h2h: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.match_history: list[dict] = []

        # Roster tracker
        self.roster_tracker: RosterTracker | None = None
        if espn_path:
            self.roster_tracker = RosterTracker(espn_path)

        # Self-improvement weights from outcome tracking (lazy-loaded)
        self._dampen: float | None = None

    def _learning_dampen(self) -> float:
        """Confidence dampening factor learned from past scored predictions.

        outcome_tracker.learn_from_mistakes() writes per-sport weights to
        oracle_learning.json; sports with sub-55% measured accuracy get
        their probabilities shrunk toward 50%.
        """
        if self._dampen is None:
            self._dampen = 1.0
            try:
                from outcome_tracker import load_learning

                weights = load_learning().get("sport_weights", {})
                w = weights.get(self.sport_key, {})
                self._dampen = float(w.get("confidence_dampen", 1.0))
            except Exception:
                pass
        return self._dampen

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
        wf_stats: dict[str, TeamStats] = defaultdict(TeamStats)
        wf_h2h: dict[tuple[str, str], list[int]] = defaultdict(list)
        wf_elo: dict[str, float] = defaultdict(lambda: 1500.0)

        X, y = [], []
        skip_first = max(min_matches // 3, 10)

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

        # Train base models
        self.models = _build_models()
        trained_names = []
        failed = []
        for name, model in list(self.models.items()):
            try:
                model.fit(X_scaled, y)
                trained_names.append(name)
            except Exception as e:
                logger.warning(f"Model {name} failed for {self.sport_key}: {e}")
                del self.models[name]
                failed.append(name)

        # Super ensemble (VotingClassifier over all base models)
        super_ens = _build_super_ensemble(self.models)
        if super_ens:
            try:
                super_ens.fit(X_scaled, y)
                self.models["SuperEnsemble"] = super_ens
                trained_names.append("SuperEnsemble")
            except Exception as e:
                logger.debug(f"SuperEnsemble failed: {e}")

        # Stacking meta-learner: LogisticRegression on base model outputs
        self.meta_learner = None
        if len(X_scaled) >= 20 and len(self.models) >= 3:
            meta_features = []
            for xi in X_scaled:
                row = []
                for model in self.models.values():
                    if hasattr(model, "predict_proba"):
                        try:
                            p = model.predict_proba(xi.reshape(1, -1))[0]
                            row.append(p[1] if len(p) > 1 else p[0])
                        except Exception:
                            row.append(0.5)
                meta_features.append(row)

            if meta_features and len(meta_features[0]) >= 2:
                meta_X = np.array(meta_features)
                meta_X = np.nan_to_num(meta_X, nan=0.5, posinf=0.5, neginf=0.5)
                try:
                    self.meta_learner = LogisticRegression(max_iter=2000, random_state=42)
                    self.meta_learner.fit(meta_X, y)
                    trained_names.append("StackingMeta")
                except Exception as e:
                    logger.debug(f"Stacking meta-learner failed: {e}")
                    self.meta_learner = None

        self.is_trained = len(self.models) > 0
        return {
            "sport": self.sport_key,
            "training_samples": len(X),
            "models_trained": len(self.models) + (1 if self.meta_learner else 0),
            "model_names": trained_names,
            "failed": failed,
        }

    def predict(
        self,
        team_a: str,
        team_b: str,
        home_team: str | None,
        match_date: str,
        roster_a: dict | None = None,
        roster_b: dict | None = None,
    ) -> dict:
        """Predict match outcome using ML ensemble + Elo blend.

        Optionally accepts roster dicts for roster-aware features.
        If roster_tracker is set and no roster provided, auto-fetches.
        """
        elo_a = self.elo[team_a]
        elo_b = self.elo[team_b]

        # Elo-only prediction
        ha = self.home_adv if home_team == team_a else (-self.home_adv if home_team == team_b else 0)
        elo_prob = 1 / (1 + 10 ** ((elo_b - (elo_a + ha)) / 400))

        if not self.is_trained:
            pa = round(elo_prob * 100, 1)
            pb = round((1 - elo_prob) * 100, 1)
            return self._format_prediction(team_a, team_b, pa, pb, method="elo")

        # Auto-fetch roster if tracker available and not provided
        if self.roster_tracker and roster_a is None:
            roster_a = self.roster_tracker.check_roster(team_a)
        if self.roster_tracker and roster_b is None:
            roster_b = self.roster_tracker.check_roster(team_b)

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
            roster_a,
            roster_b,
        )
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Ensemble prediction
        probs = []
        model_votes = {}
        for name, model in self.models.items():
            try:
                p = model.predict_proba(features_scaled)[0]
                prob_a = float(p[1] if len(p) > 1 else p[0])
                if not np.isfinite(prob_a):
                    continue
                probs.append(prob_a)
                model_votes[name] = round(prob_a, 3)
            except Exception:
                continue

        # Stacking meta-learner prediction
        if self.meta_learner and probs:
            meta_input = np.array([list(model_votes.values())])
            meta_input = np.nan_to_num(meta_input, nan=0.5, posinf=0.5, neginf=0.5)
            try:
                meta_p = self.meta_learner.predict_proba(meta_input)[0]
                meta_prob = float(meta_p[1] if len(meta_p) > 1 else meta_p[0])
                if np.isfinite(meta_prob):
                    probs.append(meta_prob)
                    model_votes["StackingMeta"] = round(meta_prob, 3)
            except Exception:
                pass

        if not probs:
            pa = round(elo_prob * 100, 1)
            pb = round((1 - elo_prob) * 100, 1)
            return self._format_prediction(team_a, team_b, pa, pb, method="elo")

        ml_prob = float(np.mean(probs))
        ml_std = float(np.std(probs))

        # Adaptive blend: ML features need match history to be informative.
        # With no recent games for either side they collapse to defaults, so
        # lean on Elo (which carries record-based priors) instead.
        n_a = len(self.stats[team_a].results) if team_a in self.stats else 0
        n_b = len(self.stats[team_b].results) if team_b in self.stats else 0
        ml_weight = 0.7 if min(n_a, n_b) >= 3 else 0.3
        blended = ml_weight * ml_prob + (1 - ml_weight) * elo_prob
        if not math.isfinite(blended):
            blended = elo_prob

        # Self-improvement: shrink toward 50% if outcome tracking found this
        # sport's model to be overconfident (weights from oracle_learning.json)
        dampen = self._learning_dampen()
        if dampen < 1.0:
            blended = 0.5 + (blended - 0.5) * dampen

        pa = round(blended * 100, 1)
        pb = round((1 - blended) * 100, 1)

        # Confidence based on agreement + distance from 50%
        diff = abs(pa - 50)
        agreement_penalty = max(0, ml_std - 0.15) * 30
        adjusted_diff = max(diff - agreement_penalty, 0)
        conf = "HIGH" if adjusted_diff > 15 else "MODERATE" if adjusted_diff > 7 else "LOW"

        extra = {}
        if dampen < 1.0:
            extra["learning_dampen"] = round(dampen, 3)
        result = self._format_prediction(
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
            **extra,
        )

        # Attach roster change alerts
        if roster_a and roster_a.get("new_players", 0) > 0:
            result["roster_alert_a"] = f"{roster_a['new_players']} new player(s)"
        if roster_b and roster_b.get("new_players", 0) > 0:
            result["roster_alert_b"] = f"{roster_b['new_players']} new player(s)"

        return result

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
