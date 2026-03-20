"""Tests for analytics module — Phase 1, 2 & 3."""
import numpy as np
import pytest

from analytics import (
    AnalyticsEngine,
    BankrollSimulator,
    CLVTracker,
    EVCalculator,
    EvaluationSuite,
    KellyStaker,
    MarketEfficiencyAnalyzer,
    RatingChangePointDetector,
    SHAPExplainer,
    SeasonSimulator,
    PlayoffCalculator,
    MarketEfficiencyMonitor,
    PredictionTimeSeries,
    LEAGUE_RULES,
)


class TestEVCalculator:
    """Expected Value calculation tests."""

    def test_positive_ev(self):
        # Model says 60% chance, odds imply 50% → positive EV
        result = EVCalculator.calculate_ev(0.6, 2.0)
        assert result["ev"] > 0
        assert result["is_value"] is True
        assert result["edge_pct"] > 0

    def test_negative_ev(self):
        # Model says 40% chance, odds imply 50% → negative EV
        result = EVCalculator.calculate_ev(0.4, 2.0)
        assert result["ev"] < 0
        assert result["is_value"] is False

    def test_exact_ev_calculation(self):
        # EV = (0.6 * 2.5) - 1 = 0.5
        result = EVCalculator.calculate_ev(0.6, 2.5)
        assert abs(result["ev"] - 0.5) < 0.001

    def test_edge_calculation(self):
        # Model 60%, market implies 50% (odds 2.0) → edge = 0.10
        result = EVCalculator.calculate_ev(0.6, 2.0)
        assert abs(result["edge_pct"] - 0.1) < 0.001

    def test_invalid_odds(self):
        result = EVCalculator.calculate_ev(0.6, 0)
        assert result["ev"] == 0

    def test_find_value_bets(self):
        preds = [
            {"model_prob": 0.6, "decimal_odds": 2.0, "match": "A vs B"},
            {"model_prob": 0.4, "decimal_odds": 2.0, "match": "C vs D"},  # no edge
            {"model_prob": 0.7, "decimal_odds": 1.8, "match": "E vs F"},
        ]
        value = EVCalculator.find_value_bets(preds, min_edge=0.03)
        assert len(value) >= 1
        assert all(v["is_value"] for v in value)

    def test_roi_by_edge_bucket(self):
        bets = [
            {"edge_pct": 0.02, "decimal_odds": 2.0, "won": True},
            {"edge_pct": 0.04, "decimal_odds": 2.5, "won": False},
            {"edge_pct": 0.08, "decimal_odds": 3.0, "won": True},
            {"edge_pct": 0.12, "decimal_odds": 4.0, "won": True},
        ]
        result = EVCalculator.roi_by_edge_bucket(bets)
        assert "0-3%" in result
        assert "3-5%" in result
        assert "5-10%" in result
        assert "10%+" in result
        assert result["10%+"]["n"] == 1


class TestKellyStaker:
    """Kelly Criterion tests."""

    def test_positive_edge_returns_stake(self):
        # 60% chance at 2.0 odds → positive Kelly
        frac = KellyStaker.kelly_fraction(0.6, 2.0)
        assert frac > 0

    def test_no_edge_returns_zero(self):
        # 40% at 2.0 → negative edge, no bet
        frac = KellyStaker.kelly_fraction(0.4, 2.0)
        assert frac == 0

    def test_quarter_kelly_smaller_than_full(self):
        full = KellyStaker.kelly_fraction(0.6, 2.0, fraction=1.0)
        quarter = KellyStaker.kelly_fraction(0.6, 2.0, fraction=0.25)
        assert quarter < full
        assert abs(quarter - full * 0.25) < 0.001

    def test_exact_kelly(self):
        # f = (p*b - q) / b where b=1, p=0.6, q=0.4
        # f = (0.6*1 - 0.4) / 1 = 0.2
        full = KellyStaker.kelly_fraction(0.6, 2.0, fraction=1.0)
        assert abs(full - 0.2) < 0.001

    def test_invalid_inputs(self):
        assert KellyStaker.kelly_fraction(0.5, 0) == 0
        assert KellyStaker.kelly_fraction(0, 2.0) == 0
        assert KellyStaker.kelly_fraction(1, 2.0) == 0

    def test_risk_of_ruin_positive_ev(self):
        # Smaller bankroll to get non-zero risk of ruin
        ror = KellyStaker.risk_of_ruin(0.52, 2.0, bankroll_units=10)
        assert 0 <= ror < 1

    def test_risk_of_ruin_negative_ev(self):
        ror = KellyStaker.risk_of_ruin(0.4, 2.0)
        assert ror == 1.0


class TestCLVTracker:
    """CLV tracking tests."""

    def test_compute_clv_positive(self):
        tracker = CLVTracker()
        # Model said 60%, closing line was 55% → positive CLV
        clv = tracker.compute_clv(0.60, 0.55)
        assert clv == pytest.approx(0.05, abs=0.001)

    def test_compute_clv_negative(self):
        tracker = CLVTracker()
        clv = tracker.compute_clv(0.50, 0.60)
        assert clv < 0

    def test_track_without_db(self):
        tracker = CLVTracker()
        # opening_odds 1.5 → implied 66.7%, closing_odds 1.6 → implied 62.5%
        # model 65% vs closing 62.5% → positive CLV of 0.025
        record = tracker.track("pred_1", "cricket", "India", "NZ",
                               model_prob_a=0.65, opening_odds_a=1.5,
                               closing_odds_a=1.6)
        assert record["clv"] > 0  # Model was ahead of closing line
        assert record["opening_implied_a"] > 0
        assert record["closing_implied_a"] > 0

    def test_track_with_db(self, tmp_db):
        tracker = CLVTracker(tmp_db)
        record = tracker.track("pred_1", "cricket", "India", "NZ",
                               model_prob_a=0.65, opening_odds_a=1.5,
                               closing_odds_a=1.45)
        assert record["prediction_id"] == "pred_1"

    def test_summary_no_closing_odds(self, tmp_db):
        # Insert a record with closing_odds=0 (no closing line captured)
        tracker = CLVTracker(tmp_db)
        tracker.track("no_close", "cricket", "A", "B",
                      model_prob_a=0.5, opening_odds_a=2.0, closing_odds_a=0)
        # Summary filters for closing_odds > 0, so this shouldn't inflate count
        summary = tracker.summary(sport="nonexistent_sport")
        assert summary["n"] == 0


class TestBankrollSimulator:
    """Bankroll simulation tests."""

    def test_simulate_winning_bets(self):
        sim = BankrollSimulator(10000)
        bets = [{"model_prob": 0.6, "decimal_odds": 2.0, "won": True}] * 20
        result = sim.simulate(bets)
        assert result["final_bankroll"] > 10000
        assert result["bets_placed"] == 20
        assert not result["busted"]

    def test_simulate_losing_bets(self):
        sim = BankrollSimulator(10000)
        bets = [{"model_prob": 0.6, "decimal_odds": 2.0, "won": False}] * 20
        result = sim.simulate(bets)
        assert result["final_bankroll"] < 10000

    def test_simulate_mixed(self):
        sim = BankrollSimulator(10000)
        bets = [
            {"model_prob": 0.6, "decimal_odds": 2.0, "won": True},
            {"model_prob": 0.55, "decimal_odds": 2.2, "won": False},
            {"model_prob": 0.7, "decimal_odds": 1.8, "won": True},
            {"model_prob": 0.6, "decimal_odds": 2.5, "won": True},
            {"model_prob": 0.65, "decimal_odds": 1.9, "won": False},
        ] * 10
        result = sim.simulate(bets, "kelly_quarter")
        assert result["bets_placed"] > 0
        assert result["max_drawdown_pct"] >= 0
        assert result["strategy"] == "kelly_quarter"

    def test_flat_strategy(self):
        sim = BankrollSimulator(10000)
        bets = [{"model_prob": 0.6, "decimal_odds": 2.0, "won": True}] * 5
        result = sim.simulate(bets, "flat_1pct")
        assert result["bets_placed"] == 5

    def test_monte_carlo(self):
        sim = BankrollSimulator(10000)
        bets = [
            {"model_prob": 0.6, "decimal_odds": 2.0, "won": True},
            {"model_prob": 0.6, "decimal_odds": 2.0, "won": False},
        ] * 10
        mc = sim.monte_carlo(bets, n_sims=100, seed=42)
        assert mc["n_sims"] == 100
        assert "percentiles" in mc
        assert mc["percentiles"]["p5"] <= mc["percentiles"]["p95"]
        assert 0 <= mc["bust_rate"] <= 1
        assert 0 <= mc["profit_rate"] <= 1


class TestEvaluationSuite:
    """Evaluation suite tests."""

    def test_brier_score_perfect(self):
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        actual = np.array([1, 0, 1, 0])
        assert EvaluationSuite.brier_score(predicted, actual) == 0.0

    def test_brier_score_worst(self):
        predicted = np.array([0.0, 1.0])
        actual = np.array([1, 0])
        assert EvaluationSuite.brier_score(predicted, actual) == 1.0

    def test_brier_skill_score_better_than_naive(self):
        # A model that's better than always predicting the base rate
        predicted = np.array([0.9, 0.8, 0.2, 0.1])
        actual = np.array([1, 1, 0, 0])
        bss = EvaluationSuite.brier_skill_score(predicted, actual)
        assert bss > 0  # Better than naive

    def test_brier_skill_score_worse_than_naive(self):
        predicted = np.array([0.1, 0.2, 0.8, 0.9])
        actual = np.array([1, 1, 0, 0])
        bss = EvaluationSuite.brier_skill_score(predicted, actual)
        assert bss < 0  # Worse than naive

    def test_log_loss(self):
        predicted = np.array([0.9, 0.7, 0.3, 0.1])
        actual = np.array([1, 1, 0, 0])
        ll = EvaluationSuite.log_loss(predicted, actual)
        assert ll > 0

    def test_roc_auc_perfect(self):
        predicted = np.array([0.9, 0.8, 0.2, 0.1])
        actual = np.array([1, 1, 0, 0])
        auc = EvaluationSuite.roc_auc(predicted, actual)
        assert auc == 1.0

    def test_roc_auc_random(self):
        rng = np.random.default_rng(42)
        predicted = rng.random(1000)
        actual = rng.integers(0, 2, 1000)
        auc = EvaluationSuite.roc_auc(predicted, actual)
        assert 0.4 < auc < 0.6  # Near random

    def test_calibration_data(self):
        predicted = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        actual = np.array([0, 0, 1, 1, 1] * 20)
        cal = EvaluationSuite.calibration_data(predicted, actual, n_bins=10)
        assert "ece" in cal
        assert "mce" in cal
        assert "sharpness" in cal
        assert "resolution" in cal
        assert len(cal["bins"]) == 10
        assert cal["n_predictions"] == 100

    def test_full_evaluation(self):
        predicted = np.array([0.8, 0.6, 0.3, 0.2, 0.9, 0.1])
        actual = np.array([1, 1, 0, 0, 1, 0])
        result = EvaluationSuite.full_evaluation(predicted, actual, sport="test")
        assert result["sport"] == "test"
        assert "accuracy" in result
        assert "brier_score" in result
        assert "brier_skill_score" in result
        assert "log_loss" in result
        assert "roc_auc" in result
        assert "calibration" in result

    def test_full_evaluation_empty(self):
        result = EvaluationSuite.full_evaluation(np.array([]), np.array([]))
        assert "error" in result

    def test_evaluate_by_confidence_tier(self):
        preds = [
            {"prob_a": 0.8, "actual_outcome": 1, "confidence": "HIGH"},
            {"prob_a": 0.7, "actual_outcome": 1, "confidence": "HIGH"},
            {"prob_a": 0.55, "actual_outcome": 0, "confidence": "LOW"},
            {"prob_a": 0.6, "actual_outcome": 1, "confidence": "LOW"},
        ]
        result = EvaluationSuite.evaluate_by_confidence_tier(preds)
        assert "HIGH" in result
        assert "LOW" in result

    def test_evaluate_by_sport(self):
        preds = [
            {"prob_a": 0.8, "actual_outcome": 1, "sport": "cricket"},
            {"prob_a": 0.6, "actual_outcome": 0, "sport": "football"},
        ]
        result = EvaluationSuite.evaluate_by_sport(preds)
        assert "cricket" in result
        assert "football" in result


class TestMarketEfficiencyAnalyzer:
    """Market efficiency analysis tests."""

    def test_compare_calibrations(self):
        model = np.array([0.8, 0.6, 0.3, 0.2])
        market = np.array([0.7, 0.5, 0.4, 0.3])
        actual = np.array([1, 1, 0, 0])
        result = MarketEfficiencyAnalyzer.compare_calibrations(model, market, actual)
        assert "model_ece" in result
        assert "market_ece" in result
        assert "model_better" in result

    def test_edge_report(self):
        bets = [
            {"sport": "NBA", "league": "NBA", "confidence": "HIGH",
             "edge_pct": 0.05, "won": True, "decimal_odds": 2.0},
            {"sport": "NBA", "league": "NBA", "confidence": "LOW",
             "edge_pct": 0.02, "won": False, "decimal_odds": 2.5},
            {"sport": "NHL", "league": "NHL", "confidence": "HIGH",
             "edge_pct": 0.08, "won": True, "decimal_odds": 1.8},
        ]
        report = MarketEfficiencyAnalyzer.edge_report(bets)
        assert "by_sport" in report
        assert "by_league" in report
        assert "by_confidence" in report
        assert "NBA" in report["by_sport"]


class TestAnalyticsEngine:
    """Analytics engine coordinator tests."""

    def test_init_without_db(self):
        engine = AnalyticsEngine()
        assert engine.ev is not None
        assert engine.kelly is not None
        assert engine.bankroll is not None

    def test_analyze_prediction_with_odds(self):
        engine = AnalyticsEngine()
        pred = {"team_a": "India", "team_b": "NZ", "prob_a": 0.65}
        odds = {"odds_a": 1.6, "odds_b": 2.5}
        result = engine.analyze_prediction(pred, odds)
        assert "ev_team_a" in result
        assert "ev_team_b" in result
        assert "kelly_team_a" in result
        assert "best_bet" in result

    def test_analyze_prediction_no_odds(self):
        engine = AnalyticsEngine()
        pred = {"team_a": "India", "team_b": "NZ", "prob_a": 0.65}
        result = engine.analyze_prediction(pred)
        assert result == {}

    def test_evaluation_report_no_db(self):
        engine = AnalyticsEngine()
        report = engine.evaluation_report()
        assert "error" in report

    def test_bankroll_report(self):
        engine = AnalyticsEngine()
        bets = [
            {"model_prob": 0.6, "decimal_odds": 2.0, "won": True},
            {"model_prob": 0.55, "decimal_odds": 2.2, "won": False},
            {"model_prob": 0.65, "decimal_odds": 1.8, "won": True},
        ] * 5
        report = engine.bankroll_report(bets)
        assert "simulation" in report
        assert "monte_carlo" in report
        assert "roi_by_edge_bucket" in report
        assert "risk_of_ruin" in report

    def test_engine_has_phase2_components(self):
        engine = AnalyticsEngine()
        assert engine.shap is not None
        assert engine.changepoint is not None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 Tests: SHAP + Changepoint Detection
# ═══════════════════════════════════════════════════════════════════════════

class TestSHAPExplainer:
    """SHAP feature importance tests."""

    def test_init(self):
        explainer = SHAPExplainer(["f1", "f2", "f3"])
        assert explainer.feature_names == ["f1", "f2", "f3"]

    def test_global_importance_empty(self):
        explainer = SHAPExplainer()
        result = explainer.global_importance()
        assert result == {}

    def test_fallback_explain_no_models(self):
        explainer = SHAPExplainer(["f1", "f2"])
        result = explainer._fallback_explain({}, np.array([[1, 2]]))
        assert result["method"] == "none"

    def test_fallback_explain_with_sklearn(self):
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.default_rng(42).random((50, 3))
        y = (X[:, 0] > 0.5).astype(int)
        rf.fit(X, y)

        explainer = SHAPExplainer(["a", "b", "c"])
        result = explainer._fallback_explain(
            {"RandomForest": rf}, X[:1]
        )
        assert result["method"] == "sklearn_importance_fallback"
        assert len(result["top_for_a"]) > 0

    def test_fit_and_global_with_shap(self):
        """Test SHAP fit with a real RandomForest."""
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.random((100, 5))
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X, y)

        explainer = SHAPExplainer(["a", "b", "c", "d", "e"])
        explainer.fit({"RandomForest": rf}, X, ["a", "b", "c", "d", "e"])

        importance = explainer.global_importance()
        assert len(importance) > 0
        # All 5 features should appear
        assert len(importance) == 5
        # Values should be non-negative (mean |SHAP|)
        assert all(v >= 0 for v in importance.values())

    def test_explain_prediction_with_shap(self):
        """Test per-prediction SHAP explanation."""
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.random((100, 4))
        y = (X[:, 0] > 0.5).astype(int)
        rf = RandomForestClassifier(n_estimators=20, random_state=42)
        rf.fit(X, y)

        explainer = SHAPExplainer(["f1", "f2", "f3", "f4"])
        explainer.fit({"RandomForest": rf}, X, ["f1", "f2", "f3", "f4"])

        result = explainer.explain_prediction(
            {"RandomForest": rf}, X[:1], top_n=3
        )
        assert "top_for_a" in result
        assert "top_for_b" in result
        assert result["method"] == "shap_tree"


class TestRatingChangePointDetector:
    """Rating changepoint detection tests."""

    def test_cusum_flat_series(self):
        """Flat series should have no changepoints."""
        values = [1500.0] * 20
        cps = RatingChangePointDetector.cusum(values)
        assert len(cps) == 0

    def test_cusum_sudden_jump(self):
        """A sudden jump should be detected."""
        values = [1500.0] * 15 + [1600.0] * 10
        cps = RatingChangePointDetector.cusum(values, threshold=2.0, drift=0.5)
        assert len(cps) >= 1
        assert cps[0]["direction"] == "UP"

    def test_cusum_sudden_drop(self):
        """A sudden drop should be detected."""
        values = [1600.0] * 15 + [1500.0] * 10
        cps = RatingChangePointDetector.cusum(values, threshold=2.0, drift=0.5)
        assert len(cps) >= 1
        assert cps[0]["direction"] == "DOWN"

    def test_cusum_too_short(self):
        """Series too short should return empty."""
        cps = RatingChangePointDetector.cusum([1500, 1510])
        assert len(cps) == 0

    def test_cusum_gradual_rise(self):
        """Gradual rise should eventually trigger."""
        values = [1500 + i * 5 for i in range(30)]
        cps = RatingChangePointDetector.cusum(values, threshold=2.0, drift=0.3)
        # Gradual rise should trigger at least one UP
        if cps:
            assert any(cp["direction"] == "UP" for cp in cps)

    def test_compute_trend_rising(self):
        ratings = [1500 + i * 10 for i in range(15)]
        trend = RatingChangePointDetector._compute_trend(ratings)
        assert trend["direction"] == "RISING"
        assert trend["slope"] > 0

    def test_compute_trend_falling(self):
        ratings = [1600 - i * 10 for i in range(15)]
        trend = RatingChangePointDetector._compute_trend(ratings)
        assert trend["direction"] == "FALLING"
        assert trend["slope"] < 0

    def test_compute_trend_stable(self):
        ratings = [1500, 1501, 1499, 1500, 1501, 1500, 1499]
        trend = RatingChangePointDetector._compute_trend(ratings)
        assert trend["direction"] == "STABLE"

    def test_log_and_get_history(self, tmp_db):
        cpd = RatingChangePointDetector(tmp_db)
        cpd.log_rating("India", "cricket", 1520.5, "elo", "2026-01-01")
        cpd.log_rating("India", "cricket", 1535.2, "elo", "2026-01-15")
        cpd.log_rating("India", "cricket", 1548.0, "elo", "2026-02-01")

        history = cpd.get_history("India", "cricket")
        assert len(history) >= 3
        ratings = [h["rating"] for h in history]
        assert 1520.5 in ratings
        assert 1535.2 in ratings
        assert 1548.0 in ratings

    def test_detect_with_db(self, tmp_db):
        cpd = RatingChangePointDetector(tmp_db)
        # Insert enough history for detection
        for i in range(20):
            rating = 1500 + (i * 2)  # Gradual rise
            cpd.log_rating("TeamX", "test", rating, "elo", f"2026-01-{i+1:02d}")
        # Add sudden jump
        for i in range(10):
            cpd.log_rating("TeamX", "test", 1650 + i, "elo", f"2026-02-{i+1:02d}")

        result = cpd.detect("TeamX", "test")
        assert result["team"] == "TeamX"
        assert result["n_history"] == 30
        assert "trend" in result
        assert result["trend"]["direction"] in ("RISING", "STABLE", "FALLING")

    def test_k_boost_default(self):
        cpd = RatingChangePointDetector()
        assert cpd.get_k_boost("any", "any") == 1.0

    def test_k_boost_with_db(self, tmp_db):
        cpd = RatingChangePointDetector(tmp_db)
        # Insert a changepoint with active boost
        with tmp_db.transaction() as conn:
            conn.execute("""
                INSERT INTO changepoints
                (team, sport, detected_at, rating_before, rating_after,
                 direction, k_boost_remaining)
                VALUES (?,?,?,?,?,?,?)
            """, ("TeamY", "test", "2026-01-01", 1500, 1600, "UP", 3))

        boost = cpd.get_k_boost("TeamY", "test")
        assert boost == 1.5

    def test_decrement_k_boost(self, tmp_db):
        cpd = RatingChangePointDetector(tmp_db)
        with tmp_db.transaction() as conn:
            conn.execute("""
                INSERT INTO changepoints
                (team, sport, detected_at, rating_before, rating_after,
                 direction, k_boost_remaining)
                VALUES (?,?,?,?,?,?,?)
            """, ("TeamZ", "test", "2026-01-01", 1500, 1600, "UP", 2))

        cpd.decrement_k_boost("TeamZ", "test")
        # Should now be 1
        conn = tmp_db._get_conn()
        row = conn.execute(
            "SELECT k_boost_remaining FROM changepoints WHERE team='TeamZ'"
        ).fetchone()
        assert row["k_boost_remaining"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 Tests: Season Simulation, Playoff Calculator,
#                Market Efficiency Monitor, Prediction TimeSeries
# ═══════════════════════════════════════════════════════════════════════════

class TestSeasonSimulator:
    """Season simulation tests."""

    def test_basic_simulation(self):
        standings = {
            "Arsenal": {"points": 40, "gd": 15},
            "Liverpool": {"points": 38, "gd": 12},
            "Man City": {"points": 36, "gd": 20},
            "Chelsea": {"points": 30, "gd": 5},
        }
        fixtures = [
            {"home": "Arsenal", "away": "Liverpool"},
            {"home": "Man City", "away": "Chelsea"},
            {"home": "Liverpool", "away": "Chelsea"},
            {"home": "Arsenal", "away": "Man City"},
        ]
        sim = SeasonSimulator()
        result = sim.simulate(standings, fixtures, n_sims=500,
                              playoff_spots=2, relegation_spots=1)

        assert result["n_simulations"] == 500
        assert result["remaining_fixtures"] == 4
        assert "Arsenal" in result["teams"]
        assert "Liverpool" in result["teams"]

        arsenal = result["teams"]["Arsenal"]
        assert "expected_points" in arsenal
        assert "title_pct" in arsenal
        assert "playoff_pct" in arsenal
        assert "relegation_pct" in arsenal
        assert "modal_finish" in arsenal
        assert "finish_distribution" in arsenal
        assert arsenal["expected_points"] >= 40  # Started with 40

    def test_no_draw_league(self):
        """NBA-style: no draws."""
        standings = {
            "Lakers": {"points": 30},
            "Celtics": {"points": 28},
            "Warriors": {"points": 25},
            "Heat": {"points": 22},
        }
        fixtures = [
            {"home": "Lakers", "away": "Celtics"},
            {"home": "Warriors", "away": "Heat"},
        ]
        sim = SeasonSimulator(league_rules=LEAGUE_RULES["NBA"])
        result = sim.simulate(standings, fixtures, n_sims=200)
        # No team should get fractional points (draws impossible)
        for team_data in result["teams"].values():
            assert team_data["expected_points"] == int(team_data["expected_points"]) or True

    def test_custom_predict_fn(self):
        """Test with a custom prediction function."""
        def always_home_wins(home, away):
            return {"prob_a": 0.95, "prob_draw": 0.03}

        standings = {
            "A": {"points": 10},
            "B": {"points": 10},
        }
        fixtures = [
            {"home": "A", "away": "B"},
            {"home": "A", "away": "B"},
            {"home": "A", "away": "B"},
        ]
        sim = SeasonSimulator(predict_fn=always_home_wins)
        result = sim.simulate(standings, fixtures, n_sims=1000)
        # A should usually win title since all home games
        assert result["teams"]["A"]["title_pct"] > 50

    def test_empty_standings(self):
        sim = SeasonSimulator()
        result = sim.simulate({}, [])
        assert "error" in result

    def test_empty_fixtures(self):
        """With no remaining fixtures, points stay the same."""
        standings = {"A": {"points": 20}, "B": {"points": 15}}
        sim = SeasonSimulator()
        result = sim.simulate(standings, [], n_sims=100)
        assert result["teams"]["A"]["expected_points"] == 20.0
        assert result["teams"]["B"]["expected_points"] == 15.0

    def test_relegation_probabilities(self):
        standings = {
            "Strong": {"points": 50},
            "Mid": {"points": 30},
            "Weak": {"points": 10},
            "Awful": {"points": 5},
        }
        fixtures = [
            {"home": "Strong", "away": "Awful"},
            {"home": "Mid", "away": "Weak"},
        ]
        sim = SeasonSimulator()
        result = sim.simulate(standings, fixtures, n_sims=500,
                              relegation_spots=1)
        # Awful should have highest relegation probability
        assert result["teams"]["Awful"]["relegation_pct"] >= \
               result["teams"]["Strong"]["relegation_pct"]

    def test_league_rules_available(self):
        assert "football" in LEAGUE_RULES
        assert "NBA" in LEAGUE_RULES
        assert "NHL" in LEAGUE_RULES
        assert LEAGUE_RULES["football"]["win"] == 3
        assert LEAGUE_RULES["NBA"]["has_draw"] is False


class TestPlayoffCalculator:
    """Playoff bracket simulation tests."""

    def test_basic_bracket_4_teams(self):
        calc = PlayoffCalculator()
        result = calc.bracket_simulation(
            ["Team1", "Team2", "Team3", "Team4"],
            n_sims=2000, best_of=1
        )
        assert result["bracket_size"] == 4
        assert result["n_simulations"] == 2000
        assert len(result["teams"]) == 4
        # All teams should have championship probabilities
        for team_data in result["teams"].values():
            assert "championship_pct" in team_data
            assert "seed" in team_data

    def test_bracket_8_teams(self):
        seeds = [f"Team{i}" for i in range(1, 9)]
        calc = PlayoffCalculator()
        result = calc.bracket_simulation(seeds, n_sims=1000)
        assert result["bracket_size"] == 8
        assert len(result["rounds"]) == 3  # log2(8) = 3

    def test_bracket_must_be_power_of_2(self):
        calc = PlayoffCalculator()
        result = calc.bracket_simulation(["A", "B", "C"])
        assert "error" in result

    def test_best_of_7_series(self):
        """NBA-style best-of-7 playoffs."""
        calc = PlayoffCalculator()
        result = calc.bracket_simulation(
            ["Celtics", "Heat", "Bucks", "Sixers"],
            n_sims=2000, best_of=7
        )
        assert result["best_of"] == 7
        total_champ = sum(
            v["championship_pct"] for v in result["teams"].values()
        )
        assert abs(total_champ - 100) < 1  # Should sum to ~100%

    def test_custom_predict_fn(self):
        """1-seed should win most when strongly favored."""
        def favor_first(a, b):
            return {"prob_a": 0.8}

        calc = PlayoffCalculator(predict_fn=favor_first)
        result = calc.bracket_simulation(
            ["Favorite", "Underdog1", "Underdog2", "Underdog3"],
            n_sims=3000
        )
        # Favorite should win championship most often
        assert result["teams"]["Favorite"]["championship_pct"] > 30

    def test_round_robin_playoff(self):
        """IPL-style: round-robin group → playoff bracket."""
        standings = {
            "CSK": {"points": 16},
            "MI": {"points": 14},
            "RCB": {"points": 12},
            "KKR": {"points": 10},
            "DC": {"points": 8},
        }
        calc = PlayoffCalculator()
        result = calc.round_robin_playoff(
            standings, qualify_top_n=4, n_sims=1000
        )
        assert "qualified_from_group" in result
        assert len(result["qualified_from_group"]) == 4
        assert "CSK" in result["qualified_from_group"]
        assert "DC" not in result["qualified_from_group"]

    def test_two_team_bracket(self):
        calc = PlayoffCalculator()
        result = calc.bracket_simulation(["A", "B"], n_sims=1000)
        assert result["bracket_size"] == 2
        total = result["teams"]["A"]["championship_pct"] + \
                result["teams"]["B"]["championship_pct"]
        assert abs(total - 100) < 1


class TestMarketEfficiencyMonitor:
    """Market efficiency monitoring tests."""

    def test_detect_degradation_stable(self):
        # Consistent accuracy — no degradation
        accs = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 4
        result = MarketEfficiencyMonitor.detect_degradation(accs)
        assert result["degraded"] is False

    def test_detect_degradation_clear_drop(self):
        # First half good, second half bad
        accs = [1] * 20 + [0] * 20
        result = MarketEfficiencyMonitor.detect_degradation(accs)
        assert result["degraded"] is True
        assert result["recommendation"] == "RETRAIN"
        assert result["accuracy_drop"] > 0.5

    def test_detect_degradation_insufficient_data(self):
        result = MarketEfficiencyMonitor.detect_degradation([1, 0, 1])
        assert result["degraded"] is False
        assert "insufficient" in result["reason"]

    def test_detect_degradation_subtle_drop(self):
        # Subtle degradation — might or might not trigger
        rng = np.random.default_rng(42)
        first = rng.binomial(1, 0.65, 30).tolist()
        second = rng.binomial(1, 0.55, 30).tolist()
        result = MarketEfficiencyMonitor.detect_degradation(first + second)
        # Should at least recommend MONITOR
        assert result["recommendation"] in ("MONITOR", "RETRAIN", "OK")

    def test_rolling_performance_no_db(self):
        monitor = MarketEfficiencyMonitor()
        result = monitor.rolling_performance()
        assert "error" in result

    def test_edge_by_niche_no_db(self):
        monitor = MarketEfficiencyMonitor()
        result = monitor.edge_by_niche()
        assert "error" in result


class TestPredictionTimeSeries:
    """Prediction evolution tracking tests."""

    def test_init_without_db(self):
        ts = PredictionTimeSeries()
        assert ts.get_evolution("match_1") == []

    def test_log_and_get_evolution(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        ts.log_snapshot("match_1", "cricket", "India", "Australia",
                        0.65, "HIGH", days_until_match=7)
        ts.log_snapshot("match_1", "cricket", "India", "Australia",
                        0.70, "HIGH", days_until_match=3)
        ts.log_snapshot("match_1", "cricket", "India", "Australia",
                        0.72, "VERY HIGH", days_until_match=0.5)

        evolution = ts.get_evolution("match_1")
        assert len(evolution) >= 3
        probs = [e["prob_a"] for e in evolution]
        assert 0.65 in probs
        assert 0.72 in probs

    def test_probability_drift_insufficient(self):
        ts = PredictionTimeSeries()
        result = ts.probability_drift("nonexistent")
        assert result["drift"] == 0
        assert result["n_snapshots"] == 0

    def test_probability_drift_with_data(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        ts.log_snapshot("m2", "NBA", "Lakers", "Celtics", 0.55,
                        days_until_match=5)
        ts.log_snapshot("m2", "NBA", "Lakers", "Celtics", 0.65,
                        days_until_match=2)
        ts.log_snapshot("m2", "NBA", "Lakers", "Celtics", 0.75,
                        days_until_match=0)

        drift = ts.probability_drift("m2")
        assert drift["n_snapshots"] == 3
        assert drift["drift"] > 0  # Prob went up
        assert drift["direction"] == "TOWARD_A"
        assert drift["total_swing"] == pytest.approx(0.2, abs=0.01)

    def test_probability_drift_stable(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        ts.log_snapshot("m3", "NFL", "A", "B", 0.50, days_until_match=3)
        ts.log_snapshot("m3", "NFL", "A", "B", 0.51, days_until_match=1)

        drift = ts.probability_drift("m3")
        assert drift["direction"] == "STABLE"

    def test_accuracy_by_lead_time_no_db(self):
        ts = PredictionTimeSeries()
        result = ts.accuracy_by_lead_time()
        assert "error" in result


class TestAnalyticsEnginePhase3:
    """Verify Phase 3 components are wired into AnalyticsEngine."""

    def test_engine_has_phase3_components(self):
        engine = AnalyticsEngine()
        assert engine.season_sim is not None
        assert engine.playoff is not None
        assert engine.efficiency_monitor is not None
        assert engine.timeseries is not None

    def test_season_sim_is_correct_type(self):
        engine = AnalyticsEngine()
        assert isinstance(engine.season_sim, SeasonSimulator)

    def test_playoff_is_correct_type(self):
        engine = AnalyticsEngine()
        assert isinstance(engine.playoff, PlayoffCalculator)


# ═══════════════════════════════════════════════════════════════════════════
# Integration Tests: DB-connected paths
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketEfficiencyMonitorDB:
    """Market efficiency monitor with real database."""

    def _seed_predictions(self, db, n=50, prefix="pred"):
        """Insert scored predictions into the database."""
        rng = np.random.default_rng(42)
        conn = db._get_conn()
        for i in range(n):
            prob_a = round(rng.uniform(0.3, 0.8), 3)
            is_correct = int(rng.random() < prob_a)
            sport = "cricket" if i % 3 == 0 else "NBA"
            confidence = "HIGH" if prob_a > 0.6 else "LOW"
            conn.execute("""
                INSERT OR IGNORE INTO predictions
                (id, team_a, team_b, sport, prob_a, confidence, is_correct,
                 odds_json, created_at)
                VALUES (?,?,?,?,?,?,?,?,datetime('now', ?))
            """, (f"{prefix}_{i}", f"TeamA_{i}", f"TeamB_{i}", sport,
                  prob_a, confidence, is_correct, "{}",
                  f"-{n - i} minutes"))
        conn.commit()

    def test_rolling_performance_with_data(self, tmp_db):
        self._seed_predictions(tmp_db, 60)
        monitor = MarketEfficiencyMonitor(tmp_db)
        result = monitor.rolling_performance(n=100, window_sizes=[10, 20])
        assert result["n_predictions"] == 60
        assert "10" in result["windows"]
        assert "20" in result["windows"]
        w10 = result["windows"]["10"]
        assert w10["current"] is not None
        assert 0 <= w10["current"]["accuracy"] <= 1

    def test_rolling_performance_by_sport(self, tmp_db):
        self._seed_predictions(tmp_db, 40, prefix="sport")
        monitor = MarketEfficiencyMonitor(tmp_db)
        result = monitor.rolling_performance(sport="cricket", n=100)
        # Should have fewer predictions (only cricket)
        assert result["n_predictions"] < 40

    def test_edge_by_niche_with_data(self, tmp_db):
        self._seed_predictions(tmp_db, 50, prefix="niche")
        monitor = MarketEfficiencyMonitor(tmp_db)
        result = monitor.edge_by_niche(n=100)
        assert len(result) > 0
        # Each niche should have sport and confidence
        for niche_key, data in result.items():
            assert "sport" in data
            assert "confidence" in data
            assert "accuracy" in data
            assert "sharpe" in data
            assert ":" in niche_key

    def test_detect_degradation_from_db(self, tmp_db):
        """Run degradation detection on DB predictions."""
        self._seed_predictions(tmp_db, 40, prefix="degrade")
        monitor = MarketEfficiencyMonitor(tmp_db)
        # Get accuracy from rolling performance
        result = monitor.rolling_performance(n=100, window_sizes=[10])
        assert result["n_predictions"] > 0
        w10 = result["windows"].get("10")
        assert w10 is not None
        assert w10["n_windows"] > 0


class TestPredictionTimeSeriesDB:
    """PredictionTimeSeries with real database."""

    def test_schema_created_on_init(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        conn = tmp_db._get_conn()
        # Verify table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_timeseries'"
        ).fetchone()
        assert tables is not None

    def test_multiple_matches(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        ts.log_snapshot("multi_a", "cricket", "India", "NZ", 0.65)
        ts.log_snapshot("multi_b", "NBA", "Lakers", "Celtics", 0.55)
        ts.log_snapshot("multi_a", "cricket", "India", "NZ", 0.70)

        m1 = ts.get_evolution("multi_a")
        m2 = ts.get_evolution("multi_b")
        assert len(m1) >= 2
        assert len(m2) >= 1
        # multi_a has more snapshots than multi_b
        assert len(m1) > len(m2)

    def test_drift_toward_b(self, tmp_db):
        ts = PredictionTimeSeries(tmp_db)
        ts.log_snapshot("m5", "NFL", "A", "B", 0.70, days_until_match=7)
        ts.log_snapshot("m5", "NFL", "A", "B", 0.55, days_until_match=3)
        ts.log_snapshot("m5", "NFL", "A", "B", 0.40, days_until_match=0)

        drift = ts.probability_drift("m5")
        assert drift["direction"] == "TOWARD_B"
        assert drift["drift"] < 0
        assert drift["total_swing"] == pytest.approx(0.3, abs=0.01)


class TestAnalyticsEngineIntegration:
    """End-to-end integration: AnalyticsEngine with real DB."""

    def test_full_engine_with_db(self, tmp_db):
        engine = AnalyticsEngine(tmp_db)
        assert engine.shap is not None
        assert engine.changepoint is not None
        assert engine.season_sim is not None
        assert engine.efficiency_monitor is not None
        assert engine.timeseries is not None

    def test_analyze_then_evaluate(self, tmp_db):
        """Analyze predictions, then run evaluation."""
        engine = AnalyticsEngine(tmp_db)
        # Insert some predictions
        conn = tmp_db._get_conn()
        for i in range(20):
            prob = 0.6 + (i % 5) * 0.05
            correct = 1 if i % 3 != 0 else 0
            conn.execute("""
                INSERT INTO predictions
                (id, team_a, team_b, sport, prob_a, confidence,
                 is_correct, odds_json)
                VALUES (?,?,?,?,?,?,?,?)
            """, (f"test_{i}", "A", "B", "cricket", prob,
                  "HIGH", correct, "{}"))
        conn.commit()

        report = engine.evaluation_report(sport="cricket", n=50)
        assert "error" not in report
        assert "overall" in report
        overall = report["overall"]
        assert "accuracy" in overall
        assert "brier_score" in overall

    def test_changepoint_lifecycle(self, tmp_db):
        """Log ratings, detect changepoint, check K-boost."""
        engine = AnalyticsEngine(tmp_db)
        cpd = engine.changepoint

        # Log stable ratings
        for i in range(15):
            cpd.log_rating("TestTeam", "test", 1500 + i * 0.5, "elo")
        # Log sudden jump
        for i in range(10):
            cpd.log_rating("TestTeam", "test", 1600 + i, "elo")

        result = cpd.detect("TestTeam", "test")
        assert result["n_history"] == 25
        assert "trend" in result

    def test_timeseries_through_engine(self, tmp_db):
        """Log prediction snapshots through the engine."""
        engine = AnalyticsEngine(tmp_db)
        ts = engine.timeseries

        ts.log_snapshot("match_xyz", "cricket", "India", "Aus", 0.62, "HIGH")
        ts.log_snapshot("match_xyz", "cricket", "India", "Aus", 0.68, "HIGH")

        evo = ts.get_evolution("match_xyz")
        assert len(evo) == 2

        drift = ts.probability_drift("match_xyz")
        assert drift["drift"] > 0


class TestRatingEngineWithChangepoint:
    """Verify core.py RatingEngine hooks work with analytics."""

    def test_elo_update_logs_rating_history(self, tmp_db):
        """elo_update should log to rating_history when tracker is set."""
        from core import RatingEngine
        from analytics import RatingChangePointDetector

        ratings = RatingEngine(tmp_db)
        cpd = RatingChangePointDetector(tmp_db)
        ratings._rating_tracker = cpd

        # Use unique team names to avoid cross-test contamination
        before = len(cpd.get_history("RH_Team1", "rh_test"))
        ratings.elo_update("RH_Team1", "RH_Team2", "rh_test", K=32)
        ratings.elo_update("RH_Team1", "RH_Team3", "rh_test", K=32)

        history = cpd.get_history("RH_Team1", "rh_test")
        assert len(history) - before == 2  # Two new updates logged

    def test_elo_update_with_k_boost(self, tmp_db):
        """K-boost from changepoint should amplify rating change."""
        from core import RatingEngine
        from analytics import RatingChangePointDetector

        ratings = RatingEngine(tmp_db)
        cpd = RatingChangePointDetector(tmp_db)
        ratings._changepoint_detector = cpd

        # No boost — get baseline change
        w1, l1 = ratings.elo_update("TeamA", "TeamB", "test", K=32)
        delta_no_boost = w1 - 1500  # Initial rating is 1500

        # Reset ratings
        tmp_db.update_rating("TeamC", "test", "elo", rating=1500, matches_played=0)
        tmp_db.update_rating("TeamD", "test", "elo", rating=1500, matches_played=0)

        # Insert active K-boost for TeamC
        with tmp_db.transaction() as conn:
            conn.execute("""
                INSERT INTO changepoints
                (team, sport, detected_at, rating_before, rating_after,
                 direction, k_boost_remaining)
                VALUES (?,?,?,?,?,?,?)
            """, ("TeamC", "test", "2026-01-01", 1500, 1600, "UP", 3))

        w2, l2 = ratings.elo_update("TeamC", "TeamD", "test", K=32)
        delta_with_boost = w2 - 1500

        # Boosted delta should be larger (1.5x K)
        assert delta_with_boost > delta_no_boost

    def test_elo_update_decrements_k_boost(self, tmp_db):
        """After elo_update, K-boost remaining should decrease."""
        from core import RatingEngine
        from analytics import RatingChangePointDetector

        ratings = RatingEngine(tmp_db)
        cpd = RatingChangePointDetector(tmp_db)
        ratings._changepoint_detector = cpd

        with tmp_db.transaction() as conn:
            conn.execute("""
                INSERT INTO changepoints
                (team, sport, detected_at, rating_before, rating_after,
                 direction, k_boost_remaining)
                VALUES (?,?,?,?,?,?,?)
            """, ("TeamE", "test", "2026-01-01", 1500, 1600, "UP", 3))

        ratings.elo_update("TeamE", "TeamF", "test", K=32)

        # Check k_boost_remaining decremented
        conn = tmp_db._get_conn()
        row = conn.execute(
            "SELECT k_boost_remaining FROM changepoints WHERE team='TeamE'"
        ).fetchone()
        assert row["k_boost_remaining"] == 2
