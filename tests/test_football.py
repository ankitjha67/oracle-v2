"""Tests for the football pipeline — Elo, features, Poisson scoring."""
import math

import pytest

from football_pipeline import (
    FootballElo,
    build_football_models,
    poisson_score_predict,
)


class TestFootballElo:
    """Football Elo rating tests."""

    def test_initial_rating(self):
        elo = FootballElo()
        assert elo.get("Arsenal") == 1500

    def test_home_win_updates(self):
        elo = FootballElo()
        elo.update("Arsenal", "Chelsea", "H", 2, 0)
        assert elo.get("Arsenal") > 1500
        assert elo.get("Chelsea") < 1500

    def test_draw_updates(self):
        elo = FootballElo()
        # Give Arsenal a big advantage first
        elo.ratings["Arsenal"] = 1600
        elo.ratings["Chelsea"] = 1400
        elo.update("Arsenal", "Chelsea", "D", 1, 1)
        # Draw should lower the favorite slightly
        assert elo.get("Arsenal") < 1600

    def test_away_win(self):
        elo = FootballElo()
        elo.update("Arsenal", "Chelsea", "A", 0, 1)
        assert elo.get("Chelsea") > 1500
        assert elo.get("Arsenal") < 1500

    def test_margin_of_victory(self):
        elo1 = FootballElo()
        elo1.update("A", "B", "H", 1, 0)
        elo2 = FootballElo()
        elo2.update("C", "D", "H", 5, 0)
        # Bigger margin should create bigger rating change
        assert abs(elo2.get("C") - 1500) > abs(elo1.get("A") - 1500)

    def test_get_all_sorted(self):
        elo = FootballElo()
        elo.update("Arsenal", "Chelsea", "H", 3, 0)
        elo.update("Liverpool", "Chelsea", "H", 2, 0)
        all_ratings = elo.get_all()
        ratings_list = list(all_ratings.values())
        assert ratings_list == sorted(ratings_list, reverse=True)


class TestPoissonScorePredict:
    """Poisson score prediction tests."""

    def test_probabilities_sum_to_100(self):
        result = poisson_score_predict(1.5, 1.2, 1.3, 1.1)
        total = result["home_win_pct"] + result["draw_pct"] + result["away_win_pct"]
        assert abs(total - 100) < 1.0  # Allow small rounding error

    def test_expected_goals_positive(self):
        result = poisson_score_predict(1.5, 1.2, 1.3, 1.1)
        assert result["home_expected_goals"] > 0
        assert result["away_expected_goals"] > 0

    def test_predicted_score_format(self):
        result = poisson_score_predict(1.5, 1.2, 1.3, 1.1)
        assert "-" in result["predicted_score"]
        home, away = result["predicted_score"].split("-")
        assert home.isdigit()
        assert away.isdigit()

    def test_elo_diff_affects_prediction(self):
        result_even = poisson_score_predict(1.5, 1.2, 1.3, 1.1, elo_diff=0)
        result_strong_home = poisson_score_predict(1.5, 1.2, 1.3, 1.1, elo_diff=200)
        assert result_strong_home["home_win_pct"] > result_even["home_win_pct"]


class TestBuildFootballModels:
    """Football model building tests."""

    def test_returns_models(self):
        models = build_football_models()
        assert isinstance(models, dict)
        assert len(models) >= 4  # RF, LR, Ada, NB at minimum

    def test_models_are_classifiers(self):
        models = build_football_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"
