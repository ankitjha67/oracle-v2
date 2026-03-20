"""Tests for core infrastructure — Database, Ratings, Calibration, Backtesting."""

import numpy as np

from core import (
    BacktestResult,
    BiasAuditor,
    MonteCarloSimulator,
    ProbabilityCalibrator,
    RateLimiter,
    RatingEngine,
)


class TestOracleDB:
    """Database layer tests."""

    def test_insert_and_get_match(self, tmp_db):
        mid = tmp_db.insert_match(
            {
                "sport": "cricket",
                "team_a": "India",
                "team_b": "England",
                "venue": "Mumbai",
                "date": "2026-03-01",
                "winner": "India",
            }
        )
        assert mid
        matches = tmp_db.get_matches(sport="cricket")
        assert len(matches) >= 1
        assert matches[0]["team_a"] == "India"

    def test_head_to_head(self, tmp_db):
        for winner in ["India", "India", "England"]:
            loser = "England" if winner == "India" else "India"
            tmp_db.insert_match(
                {
                    "sport": "cricket",
                    "team_a": winner,
                    "team_b": loser,
                    "winner": winner,
                    "date": "2026-01-01",
                }
            )
        h2h = tmp_db.get_h2h("India", "England", "cricket")
        assert h2h["total"] == 3
        assert h2h["a_wins"] >= 1

    def test_insert_prediction(self, tmp_db):
        pid = tmp_db.insert_prediction(
            {
                "sport": "cricket",
                "team_a": "India",
                "team_b": "NZ",
                "prob_a": 0.65,
                "prob_b": 0.35,
                "predicted_winner": "India",
                "confidence": "HIGH",
            }
        )
        assert pid

    def test_cache_get_set(self, tmp_db):
        tmp_db.cache_set("test_key", {"value": 42}, ttl_hours=1)
        result = tmp_db.cache_get("test_key")
        assert result == {"value": 42}

    def test_cache_miss(self, tmp_db):
        result = tmp_db.cache_get("nonexistent_key")
        assert result is None

    def test_get_stats(self, tmp_db):
        stats = tmp_db.get_stats()
        assert "matches" in stats
        assert "predictions" in stats
        assert "players" in stats

    def test_audit_log(self, tmp_db):
        tmp_db.log_event("test_event", {"detail": "testing"})
        # Should not raise


class TestRateLimiter:
    """Rate limiter tests."""

    def test_acquire_token(self):
        limiter = RateLimiter("default")
        assert limiter.acquire(timeout=1.0)

    def test_singleton_pattern(self):
        rl1 = RateLimiter.get("test_api")
        rl2 = RateLimiter.get("test_api")
        assert rl1 is rl2

    def test_wait_class_method(self):
        assert RateLimiter.wait("default")


class TestRatingEngine:
    """Rating system tests."""

    def test_elo_update(self, tmp_db):
        engine = RatingEngine(tmp_db)
        new_w, new_l = engine.elo_update("India", "England", "cricket")
        assert new_w > 1500
        assert new_l < 1500

    def test_elo_with_margin(self, tmp_db):
        engine = RatingEngine(tmp_db)
        w1, _ = engine.elo_update("TeamA", "TeamB", "cricket", margin=0)
        engine2 = RatingEngine(tmp_db)
        w2, _ = engine2.elo_update("TeamC", "TeamD", "cricket", margin=100)
        # Larger margin should produce bigger rating change
        assert abs(w2 - 1500) > abs(w1 - 1500)

    def test_update_all(self, tmp_db):
        engine = RatingEngine(tmp_db)
        engine.update_all("India", "NZ", "cricket", margin=50)
        ratings = engine.get_all_ratings("India", "cricket")
        assert "elo" in ratings
        assert "glicko2" in ratings
        assert "trueskill" in ratings

    def test_surface_elo(self, tmp_db):
        engine = RatingEngine(tmp_db)
        w, l = engine.surface_elo_update("Djokovic", "Nadal", "tennis", "clay")
        assert w > 1500
        assert l < 1500


class TestProbabilityCalibrator:
    """Calibration tests."""

    def test_brier_score(self):
        cal = ProbabilityCalibrator()
        predicted = np.array([0.8, 0.6, 0.4, 0.2])
        actual = np.array([1, 1, 0, 0])
        brier = cal.brier_score(predicted, actual)
        assert 0 <= brier <= 1

    def test_log_loss(self):
        cal = ProbabilityCalibrator()
        predicted = np.array([0.9, 0.7, 0.3, 0.1])
        actual = np.array([1, 1, 0, 0])
        ll = cal.log_loss_score(predicted, actual)
        assert ll > 0

    def test_platt_scaling(self):
        cal = ProbabilityCalibrator()
        probs = np.array([0.7, 0.6, 0.8, 0.55, 0.9, 0.65, 0.75, 0.5, 0.85, 0.6])
        actuals = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        cal.fit_platt(probs, actuals)
        calibrated = cal.calibrate(0.7)
        assert 0 < calibrated < 1

    def test_uncalibrated_passthrough(self):
        cal = ProbabilityCalibrator()
        # Without fitting, calibrate should return input unchanged
        assert cal.calibrate(0.7) == 0.7

    def test_isotonic_regression(self):
        cal = ProbabilityCalibrator()
        probs = np.array([0.7, 0.6, 0.8, 0.55, 0.9, 0.65, 0.75, 0.5, 0.85, 0.6])
        actuals = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        cal.fit_isotonic(probs, actuals)
        diagram = cal.reliability_diagram()
        assert "bins" in diagram
        assert len(diagram["bins"]) > 0


class TestMonteCarloSimulator:
    """Monte Carlo simulation tests."""

    def test_simulate_match(self):
        def predict(a, b):
            return 0.7, 0.3

        mc = MonteCarloSimulator(predict, n_simulations=100, seed=42)
        result = mc.simulate_match("A", "B")
        assert result in ("A", "B")

    def test_simulate_series(self):
        def predict(a, b):
            return 0.6, 0.4

        mc = MonteCarloSimulator(predict, n_simulations=1000, seed=42)
        result = mc.simulate_series("India", "NZ", best_of=5)
        assert "India" in result
        assert "NZ" in result
        assert abs(result["India"] + result["NZ"] - 100) < 0.01

    def test_simulate_tournament(self):
        def predict(a, b):
            elos = {"A": 1600, "B": 1550, "C": 1500, "D": 1450}
            ea, eb = elos.get(a, 1500), elos.get(b, 1500)
            pa = 1 / (1 + 10 ** ((eb - ea) / 400))
            return pa, 1 - pa

        mc = MonteCarloSimulator(predict, n_simulations=500, seed=42)
        results = mc.simulate_tournament({"G1": ["A", "B"], "G2": ["C", "D"]})
        assert len(results) > 0


class TestBiasAuditor:
    """Bias auditor tests."""

    def test_audit_empty(self):
        result = BiasAuditor.audit([])
        assert "error" in result

    def test_audit_with_predictions(self):
        preds = [
            {
                "team_a": "A",
                "team_b": "B",
                "prob_a": 0.7,
                "predicted_winner": "A",
                "actual_winner": "A",
                "is_correct": True,
                "stage": "group",
            },
            {
                "team_a": "C",
                "team_b": "D",
                "prob_a": 0.6,
                "predicted_winner": "C",
                "actual_winner": "D",
                "is_correct": False,
                "stage": "semi",
            },
        ]
        result = BiasAuditor.audit(preds)
        assert result["overall_accuracy"] == 0.5
        assert result["total_predictions"] == 2


class TestBacktestResult:
    """BacktestResult dataclass tests."""

    def test_default_values(self):
        bt = BacktestResult()
        assert bt.total_matches == 0
        assert bt.accuracy == 0.0
        assert bt.predictions == []
