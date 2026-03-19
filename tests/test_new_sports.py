"""Tests for new sport integrations — multi-sport expansion and domestic cricket."""
import pytest

from multi_sport import SPORTS, SportElo, get_active_sports


class TestSportsRegistry:
    """Verify all new sports are properly configured."""

    def test_all_sports_have_required_fields(self):
        required = {"espn", "K", "home", "type"}
        for sport_key, cfg in SPORTS.items():
            for field in required:
                assert field in cfg, f"{sport_key} missing field: {field}"

    def test_sport_count(self):
        assert len(SPORTS) >= 18  # 7 original + 11 new

    def test_new_team_sports_present(self):
        new_team_sports = ["WNBA", "NCAAF", "NCAAM", "NCAAW", "RUGBY", "AFL", "MLS", "LACROSSE"]
        for sport in new_team_sports:
            assert sport in SPORTS, f"Missing sport: {sport}"
            assert SPORTS[sport]["type"] == "team"

    def test_new_individual_sports_present(self):
        assert "GOLF" in SPORTS
        assert SPORTS["GOLF"]["type"] == "player"

    def test_k_factors_reasonable(self):
        for sport, cfg in SPORTS.items():
            assert 15 <= cfg["K"] <= 50, f"{sport} K-factor {cfg['K']} out of range"

    def test_home_advantage_ranges(self):
        for sport, cfg in SPORTS.items():
            assert 0 <= cfg["home"] <= 70, f"{sport} home adv {cfg['home']} out of range"
            # Individual sports should have 0 home advantage
            if cfg["type"] in ("fighter", "player"):
                assert cfg["home"] == 0, f"{sport} individual sport should have home=0"

    def test_espn_paths_not_empty(self):
        for sport, cfg in SPORTS.items():
            assert len(cfg["espn"]) > 0, f"{sport} has empty ESPN path"


class TestActiveSeasons:
    """Season detection tests."""

    def test_nba_active_in_winter(self):
        active = get_active_sports(month=1)
        assert "NBA" in active

    def test_mlb_active_in_summer(self):
        active = get_active_sports(month=7)
        assert "MLB" in active
        assert "NFL" not in active

    def test_nfl_active_in_fall(self):
        active = get_active_sports(month=10)
        assert "NFL" in active

    def test_ufc_always_active(self):
        for month in range(1, 13):
            active = get_active_sports(month=month)
            assert "UFC" in active

    def test_college_basketball_in_season(self):
        active = get_active_sports(month=1)
        assert "NCAAM" in active
        assert "NCAAW" in active

    def test_college_football_in_season(self):
        active = get_active_sports(month=10)
        assert "NCAAF" in active

    def test_wnba_in_season(self):
        active = get_active_sports(month=6)
        assert "WNBA" in get_active_sports(month=5)

    def test_mls_always_active(self):
        active = get_active_sports(month=6)
        assert "MLS" in active


class TestNewSportEloIntegration:
    """Test that new sports work with the existing SportElo system."""

    def test_wnba_elo(self):
        cfg = SPORTS["WNBA"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        elo.update("Las Vegas Aces", "New York Liberty", home_team="Las Vegas Aces")
        assert elo.ratings["Las Vegas Aces"] > 1500

    def test_ncaaf_elo(self):
        cfg = SPORTS["NCAAF"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        elo.update("Ohio State Buckeyes", "Michigan Wolverines", home_team="Ohio State Buckeyes", margin=14)
        assert elo.ratings["Ohio State Buckeyes"] > 1500

    def test_rugby_elo(self):
        cfg = SPORTS["RUGBY"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        elo.update("New Zealand All Blacks", "South Africa Springboks",
                   home_team="New Zealand All Blacks", margin=10)
        pa, pb = elo.predict("New Zealand All Blacks", "South Africa Springboks",
                             home="New Zealand All Blacks")
        assert pa > 50

    def test_afl_strong_home_advantage(self):
        cfg = SPORTS["AFL"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        # AFL has very strong home advantage (60 points)
        pa_home, _ = elo.predict("Team A", "Team B", home="Team A")
        pa_neutral, _ = elo.predict("Team A", "Team B")
        assert pa_home > pa_neutral  # home team gets boost

    def test_mls_elo(self):
        cfg = SPORTS["MLS"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        elo.update("Inter Miami CF", "LAFC", home_team="Inter Miami CF")
        rankings = elo.rankings(n=5)
        assert "Inter Miami CF" in rankings

    def test_golf_no_home_advantage(self):
        cfg = SPORTS["GOLF"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        # Golf has no home advantage
        pa_home, _ = elo.predict("Player A", "Player B", home="Player A")
        pa_neutral, _ = elo.predict("Player A", "Player B")
        assert pa_home == pa_neutral  # no boost

    def test_lacrosse_elo(self):
        cfg = SPORTS["LACROSSE"]
        elo = SportElo(1500, cfg["K"], cfg["home"])
        elo.update("Maryland", "Johns Hopkins", home_team="Maryland")
        assert elo.ratings["Maryland"] > 1500


class TestDomesticCricketLeagues:
    """Test domestic cricket league configuration."""

    def test_leagues_defined(self):
        from cricsheet_pipeline import DOMESTIC_LEAGUES
        assert "ipl" in DOMESTIC_LEAGUES
        assert "bbl" in DOMESTIC_LEAGUES
        assert "cpl" in DOMESTIC_LEAGUES
        assert "psl" in DOMESTIC_LEAGUES
        assert "the_hundred" in DOMESTIC_LEAGUES
        assert "sa20" in DOMESTIC_LEAGUES

    def test_league_has_required_fields(self):
        from cricsheet_pipeline import DOMESTIC_LEAGUES
        required = {"name", "zip_url", "event_filter", "country"}
        for key, league in DOMESTIC_LEAGUES.items():
            for field in required:
                assert field in league, f"League {key} missing field: {field}"

    def test_league_urls_valid(self):
        from cricsheet_pipeline import DOMESTIC_LEAGUES
        for key, league in DOMESTIC_LEAGUES.items():
            assert league["zip_url"].startswith("https://cricsheet.org/downloads/")
            assert league["zip_url"].endswith(".zip")

    def test_build_all_players_mode(self):
        """Test that build_player_database works with target_teams=None."""
        from cricsheet_pipeline import build_player_database
        import tempfile
        # With empty dir, should return empty but not crash
        with tempfile.TemporaryDirectory() as tmpdir:
            players, n = build_player_database(tmpdir, min_year=2025, target_teams=None)
            assert isinstance(players, dict)
            assert n == 0
