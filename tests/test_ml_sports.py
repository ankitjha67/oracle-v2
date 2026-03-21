"""Tests for universal ML sports prediction pipeline."""

import numpy as np

from ml_sports import FEATURE_NAMES, RosterTracker, SportMLEngine, TeamStats, extract_features


class TestTeamStats:
    """Test per-team statistics tracking."""

    def test_win_rate(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-01", True, "Opp", 1500)
        ts.add(False, 3, "2026-01-02", False, "Opp2", 1500)
        ts.add(True, 10, "2026-01-03", True, "Opp3", 1500)
        assert ts.win_rate(3) == 2 / 3
        assert ts.win_rate(1) == 1.0

    def test_avg_margin(self):
        ts = TeamStats()
        ts.add(True, 10, "2026-01-01", True, "A", 1500)
        ts.add(False, 5, "2026-01-02", False, "B", 1500)
        # Margins: +10, -5
        assert ts.avg_margin(2) == 2.5

    def test_streak_positive(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-01", True, "A", 1500)
        ts.add(True, 3, "2026-01-02", True, "B", 1500)
        ts.add(True, 7, "2026-01-03", True, "C", 1500)
        assert ts.streak() == 3

    def test_streak_negative(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-01", True, "A", 1500)
        ts.add(False, 3, "2026-01-02", False, "B", 1500)
        ts.add(False, 7, "2026-01-03", False, "C", 1500)
        assert ts.streak() == -2

    def test_rest_days(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-10", True, "A", 1500)
        assert ts.rest_days("2026-01-13") == 3

    def test_rest_days_no_history(self):
        ts = TeamStats()
        assert ts.rest_days("2026-01-01") == 7.0

    def test_home_away_rates(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-01", True, "A", 1500)
        ts.add(True, 3, "2026-01-02", True, "B", 1500)
        ts.add(False, 2, "2026-01-03", False, "C", 1500)
        assert ts.home_win_rate() == 1.0
        assert ts.away_win_rate() == 0.0

    def test_strength_of_schedule(self):
        ts = TeamStats()
        ts.add(True, 5, "2026-01-01", True, "A", 1600)
        ts.add(True, 3, "2026-01-02", True, "B", 1400)
        assert ts.strength_of_schedule() == 1500.0

    def test_consistency(self):
        ts = TeamStats()
        ts.add(True, 10, "2026-01-01", True, "A", 1500)
        ts.add(True, 10, "2026-01-02", True, "B", 1500)
        assert ts.consistency() == 0.0  # identical margins

    def test_form_velocity(self):
        ts = TeamStats()
        # Improving form: loss, loss, win, win, win
        ts.add(False, 5, "2026-01-01", True, "A", 1500)
        ts.add(False, 3, "2026-01-02", True, "B", 1500)
        ts.add(True, 2, "2026-01-03", True, "C", 1500)
        ts.add(True, 4, "2026-01-04", True, "D", 1500)
        ts.add(True, 6, "2026-01-05", True, "E", 1500)
        assert ts.form_velocity() > 0  # positive slope = improving


class TestExtractFeatures:
    """Test feature extraction."""

    def test_feature_vector_length(self):
        from collections import defaultdict

        stats = defaultdict(TeamStats)
        h2h = defaultdict(list)
        features = extract_features("A", "B", 1500, 1500, "A", "2026-01-01", stats, h2h, 50)
        assert len(features) == len(FEATURE_NAMES)

    def test_elo_diff_direction(self):
        from collections import defaultdict

        stats = defaultdict(TeamStats)
        h2h = defaultdict(list)
        f1 = extract_features("A", "B", 1600, 1400, None, "2026-01-01", stats, h2h, 0)
        f2 = extract_features("A", "B", 1400, 1600, None, "2026-01-01", stats, h2h, 0)
        # elo_diff is at index 2
        assert f1[2] > 0
        assert f2[2] < 0

    def test_roster_features_included(self):
        from collections import defaultdict

        stats = defaultdict(TeamStats)
        h2h = defaultdict(list)
        roster_a = {"new_players": 2, "avg_experience": 5.0, "injured_count": 1}
        roster_b = {"new_players": 0, "avg_experience": 8.0, "injured_count": 3}
        features = extract_features("A", "B", 1500, 1500, None, "2026-01-01", stats, h2h, 0, roster_a, roster_b)
        assert len(features) == len(FEATURE_NAMES)
        # Check roster features are at the end (indices 29-34)
        assert features[29] == 2 / 5  # new_players_a normalized
        assert features[30] == 0 / 5  # new_players_b normalized
        assert features[31] == 5 / 15  # avg_exp_a normalized
        assert features[33] == 1 / 10  # injured_a normalized

    def test_roster_features_default_zero(self):
        from collections import defaultdict

        stats = defaultdict(TeamStats)
        h2h = defaultdict(list)
        features = extract_features("A", "B", 1500, 1500, None, "2026-01-01", stats, h2h, 0)
        # Roster features should be 0 when no roster provided
        assert features[29] == 0
        assert features[30] == 0


class TestRosterTracker:
    """Test roster tracking logic."""

    def test_check_roster_unknown_team(self):
        tracker = RosterTracker("basketball/nba")
        tracker._loaded = True  # Skip API call
        result = tracker.check_roster("Nonexistent Team")
        assert result["new_players"] == 0
        assert result["avg_experience"] == 0.0
        assert result["injured_count"] == 0

    def test_get_cached_info_empty(self):
        tracker = RosterTracker("basketball/nba")
        info = tracker.get_cached_info("Unknown")
        assert info["new_players"] == 0
        assert info["avg_experience"] == 0.0

    def test_get_cached_info_with_data(self):
        tracker = RosterTracker("basketball/nba")
        tracker._rosters["Team A"] = {
            "players": {"Player1", "Player2"},
            "avg_experience": 6.5,
            "injured_count": 1,
            "total": 2,
        }
        info = tracker.get_cached_info("Team A")
        assert info["avg_experience"] == 6.5
        assert info["injured_count"] == 1

    def test_turnover_detection(self):
        tracker = RosterTracker("basketball/nba")
        tracker._loaded = True
        # Simulate previous roster
        tracker._previous_rosters["Team A"] = {"Player1", "Player2", "Player3"}
        # Simulate current roster (Player3 left, Player4 joined)
        tracker._rosters["Team A"] = {
            "players": {"Player1", "Player2", "Player4"},
            "avg_experience": 5.0,
            "injured_count": 0,
            "total": 3,
        }
        tracker._team_ids["Team A"] = "1"

        # Override fetch_roster to return our mock data
        original_fetch = tracker.fetch_roster
        tracker.fetch_roster = lambda name: tracker._rosters.get(
            name, {"players": set(), "avg_experience": 0, "injured_count": 0, "total": 0}
        )
        result = tracker.check_roster("Team A")
        tracker.fetch_roster = original_fetch

        assert result["new_players"] == 1  # Player4
        assert result["departed_players"] == 1  # Player3
        assert result["turnover_rate"] > 0


class TestSportMLEngine:
    """Test the full ML engine pipeline."""

    def _make_engine_with_data(self, n_matches=80):
        """Create an engine with synthetic match data."""
        engine = SportMLEngine("TEST", K=25, home_adv=50)
        rng = np.random.default_rng(42)
        teams = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]

        for i in range(n_matches):
            a, b = rng.choice(teams, 2, replace=False)
            home = a if rng.random() > 0.3 else (b if rng.random() > 0.5 else None)
            # Stronger teams win more often
            strength = {"Alpha": 0.7, "Bravo": 0.6, "Charlie": 0.55, "Delta": 0.45, "Echo": 0.4, "Foxtrot": 0.35}
            p_a_wins = strength[a] / (strength[a] + strength[b])
            if home == a:
                p_a_wins += 0.05
            elif home == b:
                p_a_wins -= 0.05
            a_won = rng.random() < p_a_wins
            sa = int(rng.integers(80, 130)) if a_won else int(rng.integers(70, 110))
            sb = int(rng.integers(70, 110)) if a_won else int(rng.integers(80, 130))
            date = f"2026-01-{(i % 28) + 1:02d}"
            engine.add_result(a, b, sa, sb, home, date)

        return engine

    def test_add_result_updates_elo(self):
        engine = SportMLEngine("TEST", K=25, home_adv=0)
        engine.add_result("A", "B", 100, 90, None, "2026-01-01")
        assert engine.elo["A"] > 1500
        assert engine.elo["B"] < 1500

    def test_add_result_tracks_stats(self):
        engine = SportMLEngine("TEST", K=25, home_adv=0)
        engine.add_result("A", "B", 100, 90, None, "2026-01-01")
        assert len(engine.stats["A"].results) == 1
        assert engine.stats["A"].results[0] == 1  # won

    def test_add_result_tracks_h2h(self):
        engine = SportMLEngine("TEST", K=25, home_adv=0)
        engine.add_result("A", "B", 100, 90, None, "2026-01-01")
        assert engine.h2h[("A", "B")] == [1]
        assert engine.h2h[("B", "A")] == [0]

    def test_train_insufficient_data(self):
        engine = SportMLEngine("TEST", K=25, home_adv=0)
        engine.add_result("A", "B", 100, 90, None, "2026-01-01")
        result = engine.train(min_matches=30)
        assert "error" in result

    def test_train_success(self):
        engine = self._make_engine_with_data(80)
        result = engine.train()
        assert engine.is_trained
        assert result["models_trained"] >= 7  # RF, GBM, LR, AdaBoost, SVM, Bagging, NB

    def test_train_includes_all_model_types(self):
        engine = self._make_engine_with_data(80)
        result = engine.train()
        names = result["model_names"]
        # At minimum these should all be trained
        for expected in [
            "RandomForest",
            "GradientBoosting",
            "LogisticRegression",
            "AdaBoost",
            "SVM",
            "Bagging",
            "NaiveBayes",
        ]:
            assert expected in names, f"{expected} not in trained models"

    def test_train_has_super_ensemble(self):
        engine = self._make_engine_with_data(80)
        result = engine.train()
        assert "SuperEnsemble" in result["model_names"]

    def test_train_has_stacking_meta(self):
        engine = self._make_engine_with_data(80)
        result = engine.train()
        assert "StackingMeta" in result["model_names"]
        assert engine.meta_learner is not None

    def test_predict_with_trained_model(self):
        engine = self._make_engine_with_data(80)
        engine.train()
        pred = engine.predict("Alpha", "Foxtrot", "Alpha", "2026-02-01")
        assert "winner" in pred
        assert "prob_a" in pred
        assert "prob_b" in pred
        assert "confidence" in pred
        assert pred["method"] == "ml_ensemble"
        assert 0 < pred["prob_a"] < 100
        assert 0 < pred["prob_b"] < 100

    def test_predict_includes_stacking_meta_vote(self):
        engine = self._make_engine_with_data(80)
        engine.train()
        pred = engine.predict("Alpha", "Bravo", None, "2026-02-01")
        assert "StackingMeta" in pred.get("model_votes", {})

    def test_predict_without_training(self):
        engine = SportMLEngine("TEST", K=25, home_adv=50)
        engine.add_result("A", "B", 100, 90, "A", "2026-01-01")
        pred = engine.predict("A", "B", "A", "2026-01-05")
        assert pred["method"] == "elo"
        assert pred["prob_a"] > 50  # A already beat B

    def test_predict_with_roster_data(self):
        engine = self._make_engine_with_data(80)
        engine.train()
        roster_a = {"new_players": 3, "avg_experience": 2.0, "injured_count": 2}
        roster_b = {"new_players": 0, "avg_experience": 8.0, "injured_count": 0}
        pred = engine.predict("Alpha", "Bravo", None, "2026-02-01", roster_a, roster_b)
        assert pred["method"] == "ml_ensemble"
        # Roster alert for team with new players
        assert "roster_alert_a" in pred
        assert "3 new player(s)" in pred["roster_alert_a"]

    def test_stronger_team_has_higher_elo(self):
        engine = self._make_engine_with_data(100)
        engine.train()
        # Alpha (0.7 strength) should have higher Elo than Foxtrot (0.35)
        assert engine.elo["Alpha"] > engine.elo["Foxtrot"]

    def test_rankings(self):
        engine = self._make_engine_with_data(80)
        rankings = engine.rankings(3)
        assert len(rankings) <= 3
        values = list(rankings.values())
        assert values == sorted(values, reverse=True)

    def test_draw_ignored(self):
        engine = SportMLEngine("TEST", K=25, home_adv=0)
        engine.add_result("A", "B", 100, 100, None, "2026-01-01")
        assert len(engine.match_history) == 0
        assert engine.elo["A"] == 1500

    def test_model_votes_in_prediction(self):
        engine = self._make_engine_with_data(80)
        engine.train()
        pred = engine.predict("Alpha", "Bravo", None, "2026-02-01")
        assert "model_votes" in pred
        assert len(pred["model_votes"]) >= 7  # All base models + super + stacking

    def test_ml_blends_with_elo(self):
        engine = self._make_engine_with_data(80)
        engine.train()
        pred = engine.predict("Alpha", "Bravo", None, "2026-02-01")
        assert "ml_prob" in pred
        assert "elo_prob" in pred

    def test_espn_path_creates_roster_tracker(self):
        engine = SportMLEngine("NBA", K=25, home_adv=55, espn_path="basketball/nba")
        assert engine.roster_tracker is not None
        assert engine.roster_tracker.espn_path == "basketball/nba"

    def test_no_espn_path_no_roster_tracker(self):
        engine = SportMLEngine("TEST", K=25, home_adv=50)
        assert engine.roster_tracker is None
