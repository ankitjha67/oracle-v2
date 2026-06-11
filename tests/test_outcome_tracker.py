"""Tests for outcome tracker — scoring, dedupe, date matching, learning loop."""

from outcome_tracker import (
    _date_diff_days,
    _names_match,
    score_predictions,
    store_prediction,
)


class TestDateMatching:
    def test_exact_date(self):
        assert _date_diff_days("2026-06-11", "2026-06-11") == 0

    def test_adjacent_date(self):
        assert _date_diff_days("2026-06-12", "2026-06-11") == 1

    def test_far_date(self):
        assert _date_diff_days("2026-06-15", "2026-06-11") == 4

    def test_missing_dates_return_none(self):
        assert _date_diff_days("", "2026-06-11") is None
        assert _date_diff_days("2026-06-11", "") is None

    def test_invalid_dates_return_none(self):
        assert _date_diff_days("not-a-date", "2026-06-11") is None


class TestNamesMatch:
    def test_exact(self):
        assert _names_match("Arsenal", "Arsenal")

    def test_substring(self):
        assert _names_match("Arsenal", "Arsenal FC")

    def test_no_match(self):
        assert not _names_match("Arsenal", "Chelsea")


class TestStorePrediction:
    def test_dedupe_across_runs(self, tmp_db):
        """Re-storing the same fixture must update, not duplicate."""
        pred = {
            "match": "Detroit Tigers vs Minnesota Twins",
            "date": "2026-06-11",
            "prediction": {"winner": "Detroit Tigers", "prob_a": 57.0, "prob_b": 43.0},
        }
        pid1 = store_prediction(tmp_db, pred, sport="MLB")
        pid2 = store_prediction(tmp_db, pred, sport="MLB")
        assert pid1 == pid2
        conn = tmp_db._get_conn()
        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        assert n == 1

    def test_different_dates_are_different_predictions(self, tmp_db):
        """Same teams on different days (MLB series) = separate predictions."""
        base = {
            "match": "Chicago Cubs vs Colorado Rockies",
            "prediction": {"winner": "Chicago Cubs", "prob_a": 60.0, "prob_b": 40.0},
        }
        pid1 = store_prediction(tmp_db, {**base, "date": "2026-06-11"}, sport="MLB")
        pid2 = store_prediction(tmp_db, {**base, "date": "2026-06-12"}, sport="MLB")
        assert pid1 != pid2

    def test_scored_prediction_is_frozen(self, tmp_db):
        """A scored prediction must not be overwritten by a re-run."""
        pred = {
            "match": "A vs B",
            "date": "2026-06-11",
            "prediction": {"winner": "A", "prob_a": 70.0, "prob_b": 30.0},
        }
        pid = store_prediction(tmp_db, pred, sport="TEST")
        conn = tmp_db._get_conn()
        conn.execute("UPDATE predictions SET is_correct=1, actual_winner='A' WHERE id=?", (pid,))
        conn.commit()

        # Re-store with different probabilities — must be ignored
        pred["prediction"]["prob_a"] = 10.0
        store_prediction(tmp_db, pred, sport="TEST")
        row = conn.execute("SELECT prob_a, is_correct FROM predictions WHERE id=?", (pid,)).fetchone()
        assert row["is_correct"] == 1
        assert row["prob_a"] == 0.7


class TestScorePredictions:
    def _result(self, team_a, team_b, winner, date):
        return {
            "sport": "MLB",
            "league": "MLB",
            "team_a": team_a,
            "team_b": team_b,
            "date": date,
            "winner": winner,
            "score_a": "5",
            "score_b": "3",
            "status": "finished",
        }

    def test_scores_correct_prediction(self, tmp_db):
        store_prediction(
            tmp_db,
            {
                "match": "Tigers vs Twins",
                "date": "2026-06-11",
                "prediction": {"winner": "Tigers", "prob_a": 60.0, "prob_b": 40.0},
            },
            sport="MLB",
        )
        results = [self._result("Tigers", "Twins", "Tigers", "2026-06-11")]
        summary = score_predictions(tmp_db, results)
        assert summary["scored"] == 1
        assert summary["correct"] == 1

    def test_date_mismatch_not_scored(self, tmp_db):
        """A prediction for a later series game must not consume today's result."""
        store_prediction(
            tmp_db,
            {
                "match": "Cubs vs Rockies",
                "date": "2026-06-14",
                "prediction": {"winner": "Cubs", "prob_a": 60.0, "prob_b": 40.0},
            },
            sport="MLB",
        )
        results = [self._result("Cubs", "Rockies", "Rockies", "2026-06-11")]
        summary = score_predictions(tmp_db, results)
        assert summary["scored"] == 0

    def test_closest_date_result_wins(self, tmp_db):
        """With two finished games vs the same opponent, score against the same-day one."""
        store_prediction(
            tmp_db,
            {
                "match": "Cubs vs Rockies",
                "date": "2026-06-12",
                "prediction": {"winner": "Cubs", "prob_a": 60.0, "prob_b": 40.0},
            },
            sport="MLB",
        )
        results = [
            self._result("Cubs", "Rockies", "Rockies", "2026-06-11"),
            self._result("Cubs", "Rockies", "Cubs", "2026-06-12"),
        ]
        summary = score_predictions(tmp_db, results)
        assert summary["scored"] == 1
        assert summary["correct"] == 1
