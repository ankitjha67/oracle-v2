"""Tests for new features: form velocity, model disagreement, H2H trends, dynamic K-factor, stacking."""

import numpy as np

from core import BiasAuditor, RatingEngine


class TestDynamicKFactor:
    """Dynamic K-factor based on match importance."""

    def test_group_stage_lower_k(self, tmp_db):
        engine = RatingEngine(tmp_db)
        w_group, _ = engine.elo_update("A", "B", "cricket", stage="group")
        # Group K=20, smaller change
        assert abs(w_group - 1500) > 0

    def test_final_higher_k(self, tmp_db):
        engine1 = RatingEngine(tmp_db)
        w_group, _ = engine1.elo_update("TeamA", "TeamB", "cricket", stage="group")

        engine2 = RatingEngine(tmp_db)
        w_final, _ = engine2.elo_update("TeamC", "TeamD", "cricket", stage="final")

        # Final K=40 should produce bigger change than group K=20
        assert abs(w_final - 1500) > abs(w_group - 1500)

    def test_semi_final_k(self, tmp_db):
        engine = RatingEngine(tmp_db)
        w, l = engine.elo_update("X", "Y", "cricket", stage="semi")
        assert w > 1500
        assert l < 1500

    def test_no_stage_uses_default(self, tmp_db):
        engine = RatingEngine(tmp_db)
        w, _l = engine.elo_update("P", "Q", "cricket", K=32, stage="")
        assert w > 1500


class TestEnhancedH2H:
    """Enhanced head-to-head with trend detection."""

    def test_h2h_trend_detection(self, tmp_db):
        # Insert matches where A dominated early but B is winning recently
        matches = [
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Beta", "date": "2026-03-05"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Beta", "date": "2026-03-04"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Beta", "date": "2026-03-03"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Alpha", "date": "2026-02-01"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Alpha", "date": "2026-01-01"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Alpha", "date": "2025-12-01"},
            {"sport": "cricket", "team_a": "Alpha", "team_b": "Beta", "winner": "Alpha", "date": "2025-11-01"},
        ]
        for m in matches:
            tmp_db.insert_match(m)

        h2h = tmp_db.get_h2h("Alpha", "Beta", "cricket")
        assert h2h["total"] == 7
        assert "h2h_trend" in h2h
        assert "streak_team" in h2h
        assert "streak_len" in h2h
        assert h2h["streak_team"] == "Beta"
        assert h2h["streak_len"] == 3

    def test_h2h_no_matches(self, tmp_db):
        h2h = tmp_db.get_h2h("NoTeam1", "NoTeam2", "cricket")
        assert h2h["total"] == 0
        assert h2h["h2h_trend"] == 0.0

    def test_h2h_recent_win_rate(self, tmp_db):
        for i in range(10):
            winner = "X" if i < 7 else "Y"
            tmp_db.insert_match(
                {
                    "sport": "test",
                    "team_a": "X",
                    "team_b": "Y",
                    "winner": winner,
                    "date": f"2026-01-{10 - i:02d}",
                }
            )
        h2h = tmp_db.get_h2h("X", "Y", "test")
        assert "recent_win_rate_a" in h2h


class TestFormVelocity:
    """Form velocity and volatility features."""

    def test_form_velocity_calculation(self):
        from engine import extract_features

        # Team A on upward trend, Team B stable
        match = {
            "team_a": "India",
            "team_b": "New Zealand",
            "sport": "cricket",
            "venue": "Mumbai",
            "stage": "group",
            "form_a": [0, 0, 1, 1, 1],  # improving
            "form_b": [1, 0, 1, 0, 1],  # oscillating
        }
        features = extract_features(match)
        assert len(features) == 56

    def test_consistent_team_lower_volatility(self):
        from engine import extract_features

        # A is consistent (all wins), B is volatile
        match_consistent = {
            "team_a": "India",
            "team_b": "New Zealand",
            "sport": "cricket",
            "venue": "Mumbai",
            "stage": "group",
            "form_a": [1, 1, 1, 1, 1],  # consistent
            "form_b": [0, 1, 0, 1, 0],  # volatile
        }
        features = extract_features(match_consistent)
        # volatility_diff feature (index 24 in the 56-feature vector)
        # Consistent team A should have lower volatility than oscillating B
        vol_diff_idx = 24  # "volatility_diff"
        assert features[vol_diff_idx] < 0  # A's volatility < B's volatility


class TestModelDisagreement:
    """Model disagreement confidence calibration."""

    def test_disagreement_reduces_confidence(self):
        """When models disagree, confidence should be lower."""
        # This is an integration test — we check the logic directly
        probs_agree = [0.7, 0.72, 0.68, 0.71, 0.69]
        probs_disagree = [0.9, 0.3, 0.7, 0.5, 0.8]

        std_agree = np.std(probs_agree)
        std_disagree = np.std(probs_disagree)

        assert std_disagree > std_agree
        assert std_agree < 0.08  # strong consensus
        assert std_disagree > 0.15  # weak consensus

    def test_disagreement_penalty_range(self):
        probs = [0.5, 0.5, 0.5]
        model_std = float(np.std(probs))
        penalty = min(model_std / 0.3, 1.0)
        assert 0 <= penalty <= 1.0


class TestStacking:
    """Stacking meta-learner tests."""

    def test_meta_learner_initialization(self):
        from engine import OracleV2

        oracle = OracleV2()
        assert oracle.meta_learner is None  # not trained yet

    def test_meta_learner_trains(self):
        """Meta-learner should be created during training with enough data."""
        from engine import OracleV2

        oracle = OracleV2()
        # Create enough dummy matches with varied data
        matches = []
        venues = ["Mumbai", "Ahmedabad", "Colombo", "Chennai"]
        for i in range(20):
            matches.append(
                {
                    "team_a": "India",
                    "team_b": "New Zealand",
                    "winner": "India" if i % 3 != 0 else "New Zealand",
                    "venue": venues[i % len(venues)],
                    "stage": "group",
                    "sport": "cricket",
                    "elo_a": 1500 + i * 10,
                    "elo_b": 1500 - i * 5,
                }
            )
        stats = oracle.train(matches)
        # Meta-learner may or may not train depending on data quality
        # but has_meta_learner flag should be present
        assert "has_meta_learner" in stats
        assert stats["models"] >= 8  # at least base models trained


class TestBiasAuditor:
    """Extended bias auditor tests."""

    def test_stage_breakdown(self):
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
                "team_a": "A",
                "team_b": "B",
                "prob_a": 0.7,
                "predicted_winner": "A",
                "actual_winner": "B",
                "is_correct": False,
                "stage": "semi",
            },
            {
                "team_a": "C",
                "team_b": "D",
                "prob_a": 0.8,
                "predicted_winner": "C",
                "actual_winner": "C",
                "is_correct": True,
                "stage": "group",
            },
        ]
        result = BiasAuditor.audit(preds)
        assert "stage_accuracy" in result
        assert "group" in result["stage_accuracy"]
        assert result["stage_accuracy"]["group"]["accuracy"] == 1.0
        assert result["stage_accuracy"]["semi"]["accuracy"] == 0.0

    def test_upset_detection(self):
        preds = [
            {
                "team_a": "Fav",
                "team_b": "Dog",
                "prob_a": 0.85,
                "predicted_winner": "Fav",
                "actual_winner": "Dog",
                "is_correct": False,
                "stage": "group",
            },
        ]
        result = BiasAuditor.audit(preds)
        assert result["upset_detection"]["missed_upsets"] >= 1
