"""Tests for multi-sport Elo engine."""

from multi_sport import SportElo


class TestSportElo:
    """Universal sport Elo tests."""

    def test_initial_rating(self):
        elo = SportElo()
        assert elo.ratings["New Team"] == 1500

    def test_update_winner_gains(self):
        elo = SportElo()
        elo.update("Winner", "Loser")
        assert elo.ratings["Winner"] > 1500
        assert elo.ratings["Loser"] < 1500

    def test_home_advantage(self):
        elo1 = SportElo(home_adv=50)
        elo1.update("A", "B", home_team="A")
        elo2 = SportElo(home_adv=50)
        elo2.update("C", "D", home_team=None)
        # Home winner should gain less (was expected to win)
        assert elo1.ratings["A"] < elo2.ratings["C"]

    def test_predict(self):
        elo = SportElo()
        elo.ratings["Strong"] = 1600
        elo.ratings["Weak"] = 1400
        pa, pb = elo.predict("Strong", "Weak")
        assert pa > 50
        assert pb < 50
        assert abs(pa + pb - 100) < 0.1

    def test_rankings(self):
        elo = SportElo()
        elo.ratings["A"] = 1600
        elo.ratings["B"] = 1550
        elo.ratings["C"] = 1500
        rankings = elo.rankings(n=2)
        assert len(rankings) == 2
        assert next(iter(rankings.keys())) == "A"

    def test_set_prior_from_record(self):
        elo = SportElo()
        elo.set_prior_from_record("Fighter", "15-2-0")
        assert elo.ratings["Fighter"] > 1500

    def test_set_prior_invalid_record(self):
        elo = SportElo()
        elo.set_prior_from_record("Fighter", "invalid")
        # Should not crash, rating stays default
        assert elo.ratings["Fighter"] == 1500

    def test_margin_affects_update(self):
        elo1 = SportElo()
        elo1.update("A", "B", margin=0)
        elo2 = SportElo()
        elo2.update("C", "D", margin=50)
        assert abs(elo2.ratings["C"] - 1500) > abs(elo1.ratings["A"] - 1500)
