"""Tests for the prediction engine — features, models, player database."""

import numpy as np

from engine import (
    FEATURE_NAMES,
    PlayerDatabase,
    build_models,
    extract_features,
    get_venue_data,
)


class TestPlayerDatabase:
    """Player database tests."""

    def test_get_known_player(self):
        player = PlayerDatabase.get_player("Jasprit Bumrah")
        assert player is not None
        assert player["team"] == "India"
        assert player["role"] == "bowl_pace"

    def test_get_unknown_player(self):
        player = PlayerDatabase.get_player("Unknown Player XYZ")
        assert player is None

    def test_get_team_players(self):
        india_players = PlayerDatabase.get_team_players("India")
        assert len(india_players) > 0
        assert all(p["team"] == "India" for p in india_players)

    def test_team_aggregate_stats(self):
        stats = PlayerDatabase.team_aggregate_stats("India", "New Zealand", "Mumbai")
        assert "avg_impact" in stats
        assert "batting_form_avg" in stats
        assert "fitness_avg" in stats
        assert stats["player_count"] > 0

    def test_aggregate_unknown_team(self):
        stats = PlayerDatabase.team_aggregate_stats("Unknown Team")
        assert stats["player_count"] == 0
        assert stats["avg_impact"] == 50  # default


class TestVenueData:
    """Venue data tests."""

    def test_known_venue(self):
        vd = get_venue_data("Mumbai")
        assert vd["country"] == "India"
        assert vd["avg_score"] > 0

    def test_unknown_venue(self):
        vd = get_venue_data("Unknown Venue XYZ")
        assert "bat_first_win" in vd
        assert vd["avg_score"] == 170  # default


class TestFeatureExtraction:
    """Feature extraction tests."""

    def test_feature_count(self):
        match = {
            "team_a": "India",
            "team_b": "New Zealand",
            "sport": "cricket",
            "venue": "Mumbai",
            "stage": "final",
        }
        features = extract_features(match)
        assert len(features) == len(FEATURE_NAMES)
        assert len(features) == 56

    def test_features_are_numeric(self):
        match = {
            "team_a": "India",
            "team_b": "England",
            "sport": "cricket",
            "venue": "Ahmedabad",
            "stage": "group",
        }
        features = extract_features(match)
        assert all(np.isfinite(features))

    def test_feature_names_match(self):
        assert len(FEATURE_NAMES) == 56
        assert "elo_diff" in FEATURE_NAMES
        assert "home_advantage" in FEATURE_NAMES
        assert "form_velocity_a" in FEATURE_NAMES
        assert "momentum_x_pressure" in FEATURE_NAMES


class TestBuildModels:
    """ML model building tests."""

    def test_build_models_returns_dict(self):
        models = build_models()
        assert isinstance(models, dict)
        assert len(models) >= 8  # at minimum: RF, GB, LR, MLP, Ada, SVM, NB, Bagging

    def test_models_have_fit_method(self):
        models = build_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"

    def test_models_can_train_on_small_data(self):
        """Smoke test: models can train on minimal data."""
        models = build_models()
        X = np.random.rand(20, 50)
        y = np.array([0, 1] * 10)
        for _name, model in list(models.items())[:3]:  # test first 3 for speed
            model.fit(X, y)
            pred = model.predict(X[:1])
            assert pred[0] in (0, 1)
