"""
Oracle V2 — Advanced Analytics Module
Phase 1: CLV Tracking, Expected Value, Kelly Criterion, Bankroll Simulation,
          Evaluation Suite, Market Efficiency Analysis.
Phase 2: SHAP Feature Importance, Rating Changepoint Detection.
"""
from __future__ import annotations
import math
import logging
import json
import hashlib
import warnings
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logger = logging.getLogger("oracle.analytics")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CLV TRACKER — Closing Line Value is the gold standard
# ═══════════════════════════════════════════════════════════════════════════

class CLVTracker:
    """Track Closing Line Value — the single best metric for prediction edge.

    CLV = model_implied_prob - closing_line_prob
    Positive average CLV over 100+ bets proves real edge.
    """

    def __init__(self, db=None):
        self.db = db
        if db:
            self._ensure_schema()

    def _ensure_schema(self):
        from core import OracleDB
        with self.db.transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analytics_clv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id TEXT,
                    sport TEXT,
                    team_a TEXT,
                    team_b TEXT,
                    model_prob_a REAL,
                    opening_odds_a REAL DEFAULT 0,
                    closing_odds_a REAL DEFAULT 0,
                    opening_implied_a REAL DEFAULT 0,
                    closing_implied_a REAL DEFAULT 0,
                    clv REAL DEFAULT 0,
                    actual_outcome INTEGER DEFAULT -1,
                    ev REAL DEFAULT 0,
                    kelly_fraction REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_clv_sport ON analytics_clv(sport);
                CREATE INDEX IF NOT EXISTS idx_clv_pred ON analytics_clv(prediction_id);
            """)

    def compute_clv(self, model_prob: float, closing_implied_prob: float) -> float:
        """CLV = model probability - closing line probability.
        Positive = model saw value the market eventually agreed with."""
        return model_prob - closing_implied_prob

    def track(self, prediction_id: str, sport: str, team_a: str, team_b: str,
              model_prob_a: float, opening_odds_a: float = 0,
              closing_odds_a: float = 0) -> dict:
        """Record a CLV observation."""
        opening_implied = (1 / opening_odds_a) if opening_odds_a > 1 else 0.5
        closing_implied = (1 / closing_odds_a) if closing_odds_a > 1 else 0.5
        clv = self.compute_clv(model_prob_a, closing_implied)

        record = {
            "prediction_id": prediction_id,
            "sport": sport,
            "team_a": team_a,
            "team_b": team_b,
            "model_prob_a": model_prob_a,
            "opening_odds_a": opening_odds_a,
            "closing_odds_a": closing_odds_a,
            "opening_implied_a": round(opening_implied, 4),
            "closing_implied_a": round(closing_implied, 4),
            "clv": round(clv, 4),
        }

        if self.db:
            with self.db.transaction() as conn:
                conn.execute("""
                    INSERT INTO analytics_clv
                    (prediction_id, sport, team_a, team_b, model_prob_a,
                     opening_odds_a, closing_odds_a, opening_implied_a,
                     closing_implied_a, clv)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (prediction_id, sport, team_a, team_b, model_prob_a,
                      opening_odds_a, closing_odds_a,
                      record["opening_implied_a"], record["closing_implied_a"], clv))

        return record

    def summary(self, sport: str = None, n: int = 200) -> dict:
        """Aggregate CLV stats. Positive avg_clv = real edge."""
        if not self.db:
            return {"error": "no database"}

        conn = self.db._get_conn()
        q = "SELECT * FROM analytics_clv WHERE closing_odds_a > 0"
        params = []
        if sport:
            q += " AND sport=?"
            params.append(sport)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        rows = conn.execute(q, params).fetchall()
        records = [dict(r) for r in rows]

        if not records:
            return {"n": 0, "avg_clv": 0, "pct_positive": 0}

        clvs = [r["clv"] for r in records]
        return {
            "n": len(records),
            "avg_clv": round(np.mean(clvs), 4),
            "median_clv": round(float(np.median(clvs)), 4),
            "pct_positive": round(sum(1 for c in clvs if c > 0) / len(clvs), 3),
            "std_clv": round(float(np.std(clvs)), 4),
            "best_clv": round(max(clvs), 4),
            "worst_clv": round(min(clvs), 4),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. EV CALCULATOR — Expected Value for every prediction
# ═══════════════════════════════════════════════════════════════════════════

class EVCalculator:
    """Compute Expected Value: EV = (model_prob * decimal_odds) - 1.
    Positive EV = profitable bet in the long run."""

    @staticmethod
    def calculate_ev(model_prob: float, decimal_odds: float) -> dict:
        """Calculate EV, edge, and implied market probability.

        Args:
            model_prob: Our model's win probability (0-1)
            decimal_odds: Market decimal odds (e.g. 2.50)

        Returns:
            Dict with ev, edge_pct, implied_market_prob
        """
        if decimal_odds <= 1:
            return {"ev": 0, "edge_pct": 0, "implied_market_prob": 0.5}

        implied_market_prob = 1.0 / decimal_odds
        ev = (model_prob * decimal_odds) - 1.0
        edge_pct = model_prob - implied_market_prob

        return {
            "ev": round(ev, 4),
            "edge_pct": round(edge_pct, 4),
            "implied_market_prob": round(implied_market_prob, 4),
            "model_prob": round(model_prob, 4),
            "decimal_odds": decimal_odds,
            "is_value": ev > 0,
        }

    @staticmethod
    def find_value_bets(predictions: list[dict], min_edge: float = 0.03) -> list[dict]:
        """Filter predictions where model finds positive EV.

        Args:
            predictions: List of dicts with 'model_prob' and 'decimal_odds'
            min_edge: Minimum edge to qualify (default 3% avoids vig noise)
        """
        value_bets = []
        for pred in predictions:
            model_prob = pred.get("model_prob", 0.5)
            odds = pred.get("decimal_odds", 0)
            if odds <= 1:
                continue
            ev_data = EVCalculator.calculate_ev(model_prob, odds)
            if ev_data["edge_pct"] >= min_edge:
                value_bets.append({**pred, **ev_data})
        return sorted(value_bets, key=lambda x: -x["ev"])

    @staticmethod
    def roi_by_edge_bucket(bets: list[dict]) -> dict:
        """Bin historical bets by edge size and show ROI in each bucket.

        Each bet needs: edge_pct, decimal_odds, won (bool)
        """
        buckets = {"0-3%": [], "3-5%": [], "5-10%": [], "10%+": []}
        for b in bets:
            edge = abs(b.get("edge_pct", 0)) * 100
            if edge < 3:
                buckets["0-3%"].append(b)
            elif edge < 5:
                buckets["3-5%"].append(b)
            elif edge < 10:
                buckets["5-10%"].append(b)
            else:
                buckets["10%+"].append(b)

        result = {}
        for bucket_name, bucket_bets in buckets.items():
            if not bucket_bets:
                result[bucket_name] = {"n": 0, "roi_pct": 0, "win_rate": 0}
                continue
            wins = sum(1 for b in bucket_bets if b.get("won"))
            total_staked = len(bucket_bets)
            total_returned = sum(
                b.get("decimal_odds", 2.0) for b in bucket_bets if b.get("won")
            )
            roi = ((total_returned - total_staked) / total_staked) * 100 if total_staked else 0
            result[bucket_name] = {
                "n": total_staked,
                "wins": wins,
                "win_rate": round(wins / total_staked, 3),
                "roi_pct": round(roi, 1),
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. KELLY STAKER — Fractional Kelly Criterion for bet sizing
# ═══════════════════════════════════════════════════════════════════════════

class KellyStaker:
    """Kelly Criterion bet sizing with fractional Kelly for variance reduction."""

    @staticmethod
    def kelly_fraction(model_prob: float, decimal_odds: float,
                       fraction: float = 0.25) -> float:
        """Compute fractional Kelly stake as fraction of bankroll.

        Full Kelly is optimal but volatile. Quarter Kelly (default) is standard.
        f = fraction * (p * (odds-1) - (1-p)) / (odds-1)

        Returns 0 if no edge (negative EV).
        """
        if decimal_odds <= 1 or model_prob <= 0 or model_prob >= 1:
            return 0.0
        b = decimal_odds - 1
        q = 1 - model_prob
        full_kelly = (model_prob * b - q) / b
        if full_kelly <= 0:
            return 0.0
        return round(fraction * full_kelly, 4)

    @staticmethod
    def risk_of_ruin(win_rate: float, avg_odds: float,
                     bankroll_units: int = 100) -> float:
        """Estimate probability of losing entire bankroll.

        Uses gambler's ruin approximation for fixed-fraction betting.
        """
        if win_rate <= 0 or win_rate >= 1 or avg_odds <= 1:
            return 1.0
        # Expected gain per unit bet
        ev_per_bet = win_rate * (avg_odds - 1) - (1 - win_rate)
        if ev_per_bet <= 0:
            return 1.0
        # Variance per bet
        var_per_bet = win_rate * (avg_odds - 1) ** 2 + (1 - win_rate) * 1
        if var_per_bet <= 0:
            return 0.0
        # Risk of ruin ≈ exp(-2 * edge * bankroll / variance)
        ror = math.exp(-2 * ev_per_bet * bankroll_units / var_per_bet)
        return round(min(ror, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 4. BANKROLL SIMULATOR — Monte Carlo bankroll evolution
# ═══════════════════════════════════════════════════════════════════════════

class BankrollSimulator:
    """Simulate bankroll evolution over historical or hypothetical bets."""

    def __init__(self, initial_bankroll: float = 10000):
        self.initial = initial_bankroll

    def simulate(self, bets: list[dict],
                 strategy: str = "kelly_quarter") -> dict:
        """Simulate bankroll through a sequence of bets.

        Each bet dict needs: model_prob, decimal_odds, won (bool)
        strategy: 'kelly_quarter', 'kelly_half', 'flat_1pct', 'flat_2pct'
        """
        bankroll = self.initial
        peak = bankroll
        max_drawdown_pct = 0.0
        history = [bankroll]
        bets_placed = 0

        for bet in bets:
            prob = bet.get("model_prob", 0.5)
            odds = bet.get("decimal_odds", 2.0)
            won = bet.get("won", False)

            if odds <= 1:
                continue

            # Determine stake
            if strategy == "kelly_quarter":
                frac = KellyStaker.kelly_fraction(prob, odds, 0.25)
            elif strategy == "kelly_half":
                frac = KellyStaker.kelly_fraction(prob, odds, 0.5)
            elif strategy == "flat_2pct":
                frac = 0.02
            else:  # flat_1pct
                frac = 0.01

            if frac <= 0:
                history.append(bankroll)
                continue

            stake = bankroll * frac
            bets_placed += 1

            if won:
                bankroll += stake * (odds - 1)
            else:
                bankroll -= stake

            bankroll = max(bankroll, 0)  # Can't go negative
            peak = max(peak, bankroll)
            drawdown = (peak - bankroll) / peak if peak > 0 else 0
            max_drawdown_pct = max(max_drawdown_pct, drawdown)
            history.append(bankroll)

            if bankroll <= 0:
                break

        total_return = (bankroll - self.initial) / self.initial if self.initial > 0 else 0
        # Sharpe-like ratio: mean return / std of returns
        if len(history) > 2:
            returns = [
                (history[i] - history[i - 1]) / history[i - 1]
                for i in range(1, len(history))
                if history[i - 1] > 0
            ]
            sharpe = (np.mean(returns) / np.std(returns) *
                      np.sqrt(252)) if returns and np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            "initial_bankroll": self.initial,
            "final_bankroll": round(bankroll, 2),
            "total_return_pct": round(total_return * 100, 1),
            "max_drawdown_pct": round(max_drawdown_pct * 100, 1),
            "peak_bankroll": round(peak, 2),
            "bets_placed": bets_placed,
            "strategy": strategy,
            "sharpe_ratio": round(float(sharpe), 2),
            "busted": bankroll <= 0,
        }

    def monte_carlo(self, bets: list[dict], n_sims: int = 5000,
                    strategy: str = "kelly_quarter",
                    seed: int = 42) -> dict:
        """Run N simulations with randomized bet ordering.

        Returns percentile distribution of final bankrolls.
        """
        rng = np.random.default_rng(seed)
        finals = []

        for _ in range(n_sims):
            shuffled = list(bets)
            rng.shuffle(shuffled)
            result = self.simulate(shuffled, strategy)
            finals.append(result["final_bankroll"])

        finals_arr = np.array(finals)
        return {
            "n_sims": n_sims,
            "strategy": strategy,
            "initial": self.initial,
            "mean_final": round(float(np.mean(finals_arr)), 2),
            "median_final": round(float(np.median(finals_arr)), 2),
            "percentiles": {
                "p5": round(float(np.percentile(finals_arr, 5)), 2),
                "p25": round(float(np.percentile(finals_arr, 25)), 2),
                "p50": round(float(np.percentile(finals_arr, 50)), 2),
                "p75": round(float(np.percentile(finals_arr, 75)), 2),
                "p95": round(float(np.percentile(finals_arr, 95)), 2),
            },
            "bust_rate": round(float(np.mean(finals_arr <= 0)), 3),
            "profit_rate": round(float(np.mean(finals_arr > self.initial)), 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. EVALUATION SUITE — Proper scoring rules & calibration
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationSuite:
    """Comprehensive prediction evaluation with proper scoring rules.

    Fixes:
    - Brier Skill Score (baseline-relative)
    - Per-sport, per-confidence-tier metrics
    - Structured calibration data (10-bin reliability, resolution, sharpness)
    """

    @staticmethod
    def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Brier Score: mean squared error of probability forecasts. Lower = better."""
        return float(np.mean((predicted - actual) ** 2))

    @staticmethod
    def brier_skill_score(predicted: np.ndarray, actual: np.ndarray) -> float:
        """BSS = 1 - (BS_model / BS_naive).
        BS_naive uses the base rate as prediction for all.
        BSS > 0 = better than naive, BSS = 1 = perfect."""
        bs_model = np.mean((predicted - actual) ** 2)
        base_rate = np.mean(actual)
        bs_naive = np.mean((base_rate - actual) ** 2)
        if bs_naive == 0:
            return 0.0
        return round(float(1 - bs_model / bs_naive), 4)

    @staticmethod
    def log_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Log loss (cross-entropy). Lower = better."""
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        return float(-np.mean(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        ))

    @staticmethod
    def roc_auc(predicted: np.ndarray, actual: np.ndarray) -> float:
        """Area under ROC curve. 0.5 = random, 1.0 = perfect."""
        if len(np.unique(actual)) < 2:
            return 0.5
        # Wilcoxon-Mann-Whitney statistic
        pos = predicted[actual == 1]
        neg = predicted[actual == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        # Count concordant pairs
        auc = 0.0
        for p in pos:
            auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        auc /= (len(pos) * len(neg))
        return round(float(auc), 4)

    @staticmethod
    def calibration_data(predicted: np.ndarray, actual: np.ndarray,
                         n_bins: int = 10) -> dict:
        """Structured calibration output: per-bin stats + aggregate metrics.

        Returns ECE (Expected Calibration Error), MCE (Max Calibration Error),
        resolution, sharpness, and per-bin data for reliability diagrams.
        """
        bins_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted, bins_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_data = []
        ece = 0.0
        mce = 0.0
        base_rate = float(np.mean(actual))

        for i in range(n_bins):
            mask = bin_indices == i
            count = int(mask.sum())
            if count > 0:
                avg_pred = float(predicted[mask].mean())
                avg_actual = float(actual[mask].mean())
                gap = abs(avg_pred - avg_actual)
                ece += (count / len(predicted)) * gap
                mce = max(mce, gap)
            else:
                avg_pred = (bins_edges[i] + bins_edges[i + 1]) / 2
                avg_actual = avg_pred
                gap = 0.0

            bin_data.append({
                "bin_lower": round(float(bins_edges[i]), 2),
                "bin_upper": round(float(bins_edges[i + 1]), 2),
                "avg_predicted": round(avg_pred, 4),
                "avg_actual": round(avg_actual, 4),
                "count": count,
                "gap": round(gap, 4),
            })

        # Resolution: how much do predicted probabilities vary from the base rate?
        sharpness = float(np.var(predicted))
        # Resolution: variance of actual outcomes across bins
        bin_actuals = [b["avg_actual"] for b in bin_data if b["count"] > 0]
        resolution = float(np.var(bin_actuals)) if bin_actuals else 0

        return {
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "sharpness": round(sharpness, 4),
            "resolution": round(resolution, 4),
            "base_rate": round(base_rate, 4),
            "n_predictions": len(predicted),
            "bins": bin_data,
        }

    @staticmethod
    def full_evaluation(predicted: np.ndarray, actual: np.ndarray,
                        sport: str = "", tier: str = "") -> dict:
        """Run all metrics on a set of predictions."""
        if len(predicted) == 0:
            return {"error": "no predictions", "sport": sport, "tier": tier}

        return {
            "sport": sport,
            "tier": tier,
            "n": len(predicted),
            "accuracy": round(float(np.mean(
                (predicted > 0.5).astype(int) == actual
            )), 4),
            "brier_score": EvaluationSuite.brier_score(predicted, actual),
            "brier_skill_score": EvaluationSuite.brier_skill_score(predicted, actual),
            "log_loss": EvaluationSuite.log_loss(predicted, actual),
            "roc_auc": EvaluationSuite.roc_auc(predicted, actual),
            "calibration": EvaluationSuite.calibration_data(predicted, actual),
        }

    @staticmethod
    def evaluate_by_confidence_tier(predictions: list[dict]) -> dict:
        """Break down metrics by confidence tier.

        Each prediction needs: prob_a, actual_outcome (0 or 1), confidence
        """
        tiers = defaultdict(lambda: {"predicted": [], "actual": []})
        for p in predictions:
            prob = p.get("prob_a", 0.5)
            actual = p.get("actual_outcome", -1)
            if actual < 0:
                continue
            tier = p.get("confidence", "UNKNOWN")
            tiers[tier]["predicted"].append(prob)
            tiers[tier]["actual"].append(actual)

        results = {}
        for tier, data in tiers.items():
            pred = np.array(data["predicted"])
            act = np.array(data["actual"])
            results[tier] = EvaluationSuite.full_evaluation(pred, act, tier=tier)
        return results

    @staticmethod
    def evaluate_by_sport(predictions: list[dict]) -> dict:
        """Break down metrics by sport."""
        sports = defaultdict(lambda: {"predicted": [], "actual": []})
        for p in predictions:
            prob = p.get("prob_a", 0.5)
            actual = p.get("actual_outcome", -1)
            if actual < 0:
                continue
            sport = p.get("sport", "unknown")
            sports[sport]["predicted"].append(prob)
            sports[sport]["actual"].append(actual)

        results = {}
        for sport, data in sports.items():
            pred = np.array(data["predicted"])
            act = np.array(data["actual"])
            results[sport] = EvaluationSuite.full_evaluation(pred, act, sport=sport)
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. MARKET EFFICIENCY ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class MarketEfficiencyAnalyzer:
    """Systematic analysis of where the model has edge vs the market.

    Compares Oracle calibration vs market calibration head-to-head.
    Identifies the model's most profitable niches by sport, league,
    confidence tier, and outcome type.
    """

    @staticmethod
    def compare_calibrations(model_probs: np.ndarray, market_probs: np.ndarray,
                             actual: np.ndarray, n_bins: int = 10) -> dict:
        """Head-to-head calibration: Oracle vs Market."""
        model_cal = EvaluationSuite.calibration_data(model_probs, actual, n_bins)
        market_cal = EvaluationSuite.calibration_data(market_probs, actual, n_bins)

        model_brier = EvaluationSuite.brier_score(model_probs, actual)
        market_brier = EvaluationSuite.brier_score(market_probs, actual)

        return {
            "model_ece": model_cal["ece"],
            "market_ece": market_cal["ece"],
            "model_brier": round(model_brier, 4),
            "market_brier": round(market_brier, 4),
            "model_better": model_brier < market_brier,
            "brier_advantage": round(market_brier - model_brier, 4),
            "model_calibration": model_cal,
            "market_calibration": market_cal,
        }

    @staticmethod
    def edge_report(bets: list[dict]) -> dict:
        """Identify most profitable niches.

        Each bet needs: sport, league, confidence, edge_pct, won, decimal_odds
        """
        dimensions = {
            "by_sport": defaultdict(list),
            "by_league": defaultdict(list),
            "by_confidence": defaultdict(list),
        }

        for b in bets:
            dimensions["by_sport"][b.get("sport", "unknown")].append(b)
            dimensions["by_league"][b.get("league", "unknown")].append(b)
            dimensions["by_confidence"][b.get("confidence", "unknown")].append(b)

        report = {}
        for dim_name, groups in dimensions.items():
            dim_report = {}
            for group_name, group_bets in groups.items():
                if not group_bets:
                    continue
                wins = sum(1 for b in group_bets if b.get("won"))
                total = len(group_bets)
                returned = sum(
                    b.get("decimal_odds", 2.0) for b in group_bets if b.get("won")
                )
                roi = ((returned - total) / total) * 100 if total else 0
                avg_edge = np.mean([b.get("edge_pct", 0) for b in group_bets])
                dim_report[group_name] = {
                    "n": total,
                    "wins": wins,
                    "win_rate": round(wins / total, 3) if total else 0,
                    "roi_pct": round(roi, 1),
                    "avg_edge": round(float(avg_edge), 4),
                    "profitable": roi > 0,
                }
            report[dim_name] = dict(sorted(
                dim_report.items(), key=lambda x: -x[1].get("roi_pct", 0)
            ))
        return report


# ═══════════════════════════════════════════════════════════════════════════
# 7. SHAP FEATURE IMPORTANCE (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class SHAPExplainer:
    """SHAP-based feature importance — replaces biased sklearn feature_importances_.

    Uses TreeExplainer for tree-based models (XGBoost, LGBM, RF, GBM) which is
    exact and fast. Falls back to permutation importance if SHAP unavailable.
    """

    def __init__(self, feature_names: list[str] = None):
        self.feature_names = feature_names or []
        self._explainers: dict[str, Any] = {}
        self._global_shap_values = None

    def fit(self, models: dict, X_train: np.ndarray,
            feature_names: list[str] = None):
        """Build SHAP explainers for all tree-based models.

        Call this once after training.
        """
        if feature_names:
            self.feature_names = feature_names

        if not HAS_SHAP:
            logger.info("SHAP not installed — using permutation importance fallback")
            return

        self._explainers = {}
        tree_models = ["RandomForest", "XGBoost", "LightGBM",
                        "GradientBoosting", "Bagging"]

        for name in tree_models:
            model = models.get(name)
            if model is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    explainer = shap.TreeExplainer(model)
                self._explainers[name] = explainer
            except Exception as e:
                logger.debug(f"SHAP TreeExplainer failed for {name}: {e}")

        # Compute global SHAP values on a background sample
        if self._explainers and len(X_train) > 0:
            # Use up to 200 samples for speed
            n_sample = min(200, len(X_train))
            sample_idx = np.random.default_rng(42).choice(
                len(X_train), n_sample, replace=False
            )
            X_sample = X_train[sample_idx]
            self._compute_global_shap(X_sample)

    def _compute_global_shap(self, X_sample: np.ndarray):
        """Compute mean |SHAP| values across models for global ranking."""
        all_abs_shap = []
        for name, explainer in self._explainers.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sv = explainer.shap_values(X_sample)
                # Handle multi-output (binary classifiers return list of 2 arrays)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                all_abs_shap.append(np.abs(sv))
            except Exception as e:
                logger.debug(f"Global SHAP failed for {name}: {e}")

        if all_abs_shap:
            # Average absolute SHAP across models
            # Each element is shape (n_samples, n_features)
            # Average over models and samples to get per-feature importance
            per_model = [np.mean(sv, axis=0) for sv in all_abs_shap]
            stacked = np.stack(per_model, axis=0)  # (n_models, n_features)
            mean_vals = np.mean(stacked, axis=0)  # (n_features,)
            # Ensure 1D
            self._global_shap_values = np.atleast_1d(mean_vals).flatten()

    def global_importance(self, top_n: int = 20) -> dict:
        """Return canonical global feature ranking by mean |SHAP|."""
        if self._global_shap_values is None:
            return {}

        importance = {}
        names = self.feature_names or [f"f{i}" for i in range(len(self._global_shap_values))]
        for i, val in enumerate(self._global_shap_values):
            if i < len(names):
                importance[names[i]] = round(float(val), 6)

        return dict(sorted(importance.items(), key=lambda x: -x[1])[:top_n])

    def explain_prediction(self, models: dict, features_scaled: np.ndarray,
                           top_n: int = 5) -> dict:
        """Per-prediction SHAP: top positive and negative contributors.

        Returns the top_n features pushing toward team_a and top_n toward team_b.
        """
        if not HAS_SHAP or not self._explainers:
            return self._fallback_explain(models, features_scaled, top_n)

        all_shap = []
        for name, explainer in self._explainers.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sv = explainer.shap_values(features_scaled)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                all_shap.append(sv.flatten())
            except Exception:
                continue

        if not all_shap:
            return self._fallback_explain(models, features_scaled, top_n)

        # Average SHAP values across models
        avg_shap = np.mean(all_shap, axis=0)
        names = self.feature_names or [f"f{i}" for i in range(len(avg_shap))]

        # Build sorted list
        shap_pairs = [(names[i], float(avg_shap[i]))
                      for i in range(min(len(names), len(avg_shap)))]

        # Positive SHAP = pushes toward team_a winning
        positive = sorted([p for p in shap_pairs if p[1] > 0],
                          key=lambda x: -x[1])[:top_n]
        # Negative SHAP = pushes toward team_b winning
        negative = sorted([p for p in shap_pairs if p[1] < 0],
                          key=lambda x: x[1])[:top_n]

        return {
            "top_for_a": [{"feature": f, "shap_value": round(v, 4)} for f, v in positive],
            "top_for_b": [{"feature": f, "shap_value": round(abs(v), 4)} for f, v in negative],
            "method": "shap_tree",
        }

    def _fallback_explain(self, models: dict, features_scaled: np.ndarray,
                          top_n: int = 5) -> dict:
        """Fallback: use sklearn feature_importances_ when SHAP unavailable."""
        feat_imp = {}
        for mname in ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]:
            model = models.get(mname)
            if model is not None and hasattr(model, "feature_importances_"):
                names = self.feature_names or [f"f{i}" for i in range(len(model.feature_importances_))]
                for fn, imp in zip(names, model.feature_importances_):
                    feat_imp[fn] = feat_imp.get(fn, 0) + imp

        if not feat_imp:
            return {"top_for_a": [], "top_for_b": [], "method": "none"}

        total = sum(feat_imp.values())
        if total > 0:
            feat_imp = {k: v / total for k, v in feat_imp.items()}

        top = sorted(feat_imp.items(), key=lambda x: -x[1])[:top_n]
        return {
            "top_for_a": [{"feature": f, "shap_value": round(v, 4)} for f, v in top],
            "top_for_b": [],
            "method": "sklearn_importance_fallback",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 8. RATING CHANGEPOINT DETECTOR (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════

class RatingChangePointDetector:
    """Detect abrupt shifts in team rating trajectories using CUSUM.

    Flags coaching changes, transfer windows, injury impacts.
    After detected changepoint: boost K-factor 50% for next 5 matches.
    """

    def __init__(self, db=None):
        self.db = db
        if db:
            self._ensure_schema()

    def _ensure_schema(self):
        with self.db.transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS rating_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    rating_type TEXT DEFAULT 'elo',
                    rating REAL,
                    match_date TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_rh_team_sport
                    ON rating_history(team, sport);
                CREATE INDEX IF NOT EXISTS idx_rh_date
                    ON rating_history(match_date);

                CREATE TABLE IF NOT EXISTS changepoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    detected_at TEXT,
                    rating_before REAL,
                    rating_after REAL,
                    direction TEXT DEFAULT 'unknown',
                    k_boost_remaining INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_cp_team
                    ON changepoints(team, sport);
            """)

    def log_rating(self, team: str, sport: str, rating: float,
                   rating_type: str = "elo", match_date: str = ""):
        """Append a rating snapshot to history."""
        if not self.db:
            return
        with self.db.transaction() as conn:
            conn.execute("""
                INSERT INTO rating_history (team, sport, rating_type, rating, match_date)
                VALUES (?,?,?,?,?)
            """, (team, sport, rating_type, rating, match_date))

    def get_history(self, team: str, sport: str,
                    rating_type: str = "elo", n: int = 100) -> list[dict]:
        """Get rating time series for a team."""
        if not self.db:
            return []
        conn = self.db._get_conn()
        rows = conn.execute("""
            SELECT rating, match_date, created_at FROM rating_history
            WHERE team=? AND sport=? AND rating_type=?
            ORDER BY created_at DESC LIMIT ?
        """, (team, sport, rating_type, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    @staticmethod
    def cusum(values: list[float], threshold: float = 2.0,
              drift: float = 0.5) -> list[dict]:
        """CUSUM (Cumulative Sum) changepoint detection.

        Detects both upward and downward shifts in the time series.
        threshold: sensitivity (lower = more sensitive, typical 2-5)
        drift: allowable drift before alarm (typical 0.5-1.0)

        Returns list of detected changepoints with index and direction.
        """
        if len(values) < 5:
            return []

        # Compute differences from running mean
        arr = np.array(values, dtype=float)
        diffs = np.diff(arr)
        if len(diffs) == 0:
            return []

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        if std_diff == 0:
            return []

        # Normalize
        z = (diffs - mean_diff) / std_diff

        # CUSUM accumulators
        s_pos = 0.0
        s_neg = 0.0
        changepoints = []

        for i, zi in enumerate(z):
            s_pos = max(0, s_pos + zi - drift)
            s_neg = max(0, s_neg - zi - drift)

            if s_pos > threshold:
                changepoints.append({
                    "index": i + 1,  # +1 because diff shifts by 1
                    "direction": "UP",
                    "magnitude": round(s_pos, 2),
                    "value": float(arr[i + 1]),
                })
                s_pos = 0  # Reset after detection

            if s_neg > threshold:
                changepoints.append({
                    "index": i + 1,
                    "direction": "DOWN",
                    "magnitude": round(s_neg, 2),
                    "value": float(arr[i + 1]),
                })
                s_neg = 0

        return changepoints

    def detect(self, team: str, sport: str, threshold: float = 2.0,
               drift: float = 0.5) -> dict:
        """Run CUSUM on a team's rating history and record changepoints."""
        history = self.get_history(team, sport, n=200)
        if len(history) < 10:
            return {"team": team, "sport": sport, "changepoints": [],
                    "n_history": len(history)}

        ratings = [h["rating"] for h in history]
        cps = self.cusum(ratings, threshold, drift)

        # Record new changepoints to DB
        if cps and self.db:
            latest = cps[-1]
            idx = latest["index"]
            rating_before = ratings[max(0, idx - 3):idx]
            rating_after = ratings[idx:min(len(ratings), idx + 3)]
            with self.db.transaction() as conn:
                conn.execute("""
                    INSERT INTO changepoints
                    (team, sport, detected_at, rating_before, rating_after,
                     direction, k_boost_remaining)
                    VALUES (?,?,?,?,?,?,?)
                """, (team, sport, datetime.now().isoformat(),
                      np.mean(rating_before) if rating_before else 0,
                      np.mean(rating_after) if rating_after else 0,
                      latest["direction"], 5))

        return {
            "team": team,
            "sport": sport,
            "n_history": len(ratings),
            "current_rating": round(ratings[-1], 1) if ratings else 0,
            "changepoints": cps,
            "trend": self._compute_trend(ratings),
        }

    @staticmethod
    def _compute_trend(ratings: list[float], window: int = 10) -> dict:
        """Linear regression slope over recent ratings."""
        recent = ratings[-window:] if len(ratings) >= window else ratings
        if len(recent) < 3:
            return {"direction": "STABLE", "slope": 0}

        x = np.arange(len(recent))
        slope = float(np.polyfit(x, recent, 1)[0])

        if slope > 1.0:
            direction = "RISING"
        elif slope < -1.0:
            direction = "FALLING"
        else:
            direction = "STABLE"

        return {
            "direction": direction,
            "slope": round(slope, 2),
            "current": round(recent[-1], 1),
            "min_recent": round(min(recent), 1),
            "max_recent": round(max(recent), 1),
        }

    def get_k_boost(self, team: str, sport: str) -> float:
        """Check if team has an active K-factor boost from recent changepoint.

        Returns multiplier: 1.0 (no boost) or 1.5 (boost active).
        """
        if not self.db:
            return 1.0
        conn = self.db._get_conn()
        row = conn.execute("""
            SELECT k_boost_remaining FROM changepoints
            WHERE team=? AND sport=? AND k_boost_remaining > 0
            ORDER BY created_at DESC LIMIT 1
        """, (team, sport)).fetchone()

        if row and row["k_boost_remaining"] > 0:
            return 1.5
        return 1.0

    def decrement_k_boost(self, team: str, sport: str):
        """Decrement the K-boost counter after a match is processed."""
        if not self.db:
            return
        conn = self.db._get_conn()
        row = conn.execute("""
            SELECT id, k_boost_remaining FROM changepoints
            WHERE team=? AND sport=? AND k_boost_remaining > 0
            ORDER BY created_at DESC LIMIT 1
        """, (team, sport)).fetchone()

        if row:
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE changepoints SET k_boost_remaining=? WHERE id=?",
                    (max(0, row["k_boost_remaining"] - 1), row["id"])
                )


# ═══════════════════════════════════════════════════════════════════════════
# 9. ANALYTICS ENGINE — Coordinator that ties everything together
# ═══════════════════════════════════════════════════════════════════════════

class AnalyticsEngine:
    """Top-level coordinator for all analytics (Phase 1 + Phase 2)."""

    def __init__(self, db=None):
        self.db = db
        # Phase 1
        self.clv = CLVTracker(db)
        self.ev = EVCalculator()
        self.kelly = KellyStaker()
        self.bankroll = BankrollSimulator()
        self.evaluation = EvaluationSuite()
        self.market = MarketEfficiencyAnalyzer()
        # Phase 2
        self.shap = SHAPExplainer()
        self.changepoint = RatingChangePointDetector(db)

    def analyze_prediction(self, prediction: dict,
                           market_odds: dict = None) -> dict:
        """Run Phase 1 analytics on a single prediction.

        Called after each predict(). Adds EV, Kelly, and CLV data.
        """
        result = {}
        model_prob = prediction.get("prob_a", prediction.get("calibrated_prob_a", 50))
        # Normalize to 0-1 if given as percentage
        if model_prob > 1:
            model_prob = model_prob / 100.0

        if market_odds:
            odds_a = market_odds.get("odds_a", 0)
            odds_b = market_odds.get("odds_b", 0)

            if odds_a > 1:
                ev_a = self.ev.calculate_ev(model_prob, odds_a)
                kelly_a = self.kelly.kelly_fraction(model_prob, odds_a)
                result["ev_team_a"] = ev_a
                result["kelly_team_a"] = kelly_a

            if odds_b > 1:
                ev_b = self.ev.calculate_ev(1 - model_prob, odds_b)
                kelly_b = self.kelly.kelly_fraction(1 - model_prob, odds_b)
                result["ev_team_b"] = ev_b
                result["kelly_team_b"] = kelly_b

            # Best bet
            if result.get("ev_team_a", {}).get("ev", 0) > result.get("ev_team_b", {}).get("ev", 0):
                result["best_bet"] = {
                    "side": prediction.get("team_a", "A"),
                    "ev": result["ev_team_a"]["ev"],
                    "kelly_pct": round(result.get("kelly_team_a", 0) * 100, 2),
                    "edge_pct": round(result["ev_team_a"]["edge_pct"] * 100, 2),
                }
            elif result.get("ev_team_b", {}).get("ev", 0) > 0:
                result["best_bet"] = {
                    "side": prediction.get("team_b", "B"),
                    "ev": result["ev_team_b"]["ev"],
                    "kelly_pct": round(result.get("kelly_team_b", 0) * 100, 2),
                    "edge_pct": round(result["ev_team_b"]["edge_pct"] * 100, 2),
                }
            else:
                result["best_bet"] = {"side": "NO BET", "ev": 0, "kelly_pct": 0}

        return result

    def evaluation_report(self, sport: str = None, n: int = 500) -> dict:
        """Generate comprehensive evaluation report from stored predictions."""
        if not self.db:
            return {"error": "no database"}

        conn = self.db._get_conn()
        q = "SELECT * FROM predictions WHERE is_correct >= 0"
        params = []
        if sport:
            q += " AND sport=?"
            params.append(sport)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        rows = conn.execute(q, params).fetchall()
        preds = [dict(r) for r in rows]

        if not preds:
            return {"error": "no scored predictions"}

        predicted = np.array([p["prob_a"] for p in preds])
        actual = np.array([p["is_correct"] for p in preds])

        # Overall
        report = {
            "overall": self.evaluation.full_evaluation(predicted, actual, sport=sport or "all"),
        }

        # By sport
        pred_dicts = [
            {"prob_a": p["prob_a"], "actual_outcome": p["is_correct"],
             "sport": p["sport"], "confidence": p["confidence"]}
            for p in preds
        ]
        report["by_sport"] = self.evaluation.evaluate_by_sport(pred_dicts)
        report["by_confidence"] = self.evaluation.evaluate_by_confidence_tier(pred_dicts)
        report["clv_summary"] = self.clv.summary(sport)

        return report

    def bankroll_report(self, bets: list[dict],
                        strategy: str = "kelly_quarter") -> dict:
        """Run bankroll simulation + Monte Carlo on historical bets."""
        sim = self.bankroll.simulate(bets, strategy)
        mc = self.bankroll.monte_carlo(bets, n_sims=2000, strategy=strategy)
        roi_by_edge = self.ev.roi_by_edge_bucket(bets)

        return {
            "simulation": sim,
            "monte_carlo": mc,
            "roi_by_edge_bucket": roi_by_edge,
            "risk_of_ruin": self.kelly.risk_of_ruin(
                win_rate=sim["bets_placed"] and
                    sum(1 for b in bets if b.get("won")) / max(len(bets), 1) or 0.5,
                avg_odds=np.mean([b.get("decimal_odds", 2.0) for b in bets]) if bets else 2.0,
            ),
        }
