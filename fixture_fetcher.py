"""
Oracle V2 — Live Fixture Fetcher (REWRITTEN)
Sources: ESPN (free) + football-data.org (free key) + OpenF1
UCL two-leg aggregate tracking.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

FOOTBALL_LEAGUES = {
    "EPL": "soccer/eng.1",
    "La Liga": "soccer/esp.1",
    "Serie A": "soccer/ita.1",
    "Bundesliga": "soccer/ger.1",
    "Ligue 1": "soccer/fra.1",
    "UCL": "soccer/uefa.champions",
}

# 70+ ESPN→football-data.co.uk name mappings
NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
    "AFC Bournemouth": "Bournemouth",
    "West Ham United": "West Ham",
    "Crystal Palace": "Crystal Palace",
    "Aston Villa": "Aston Villa",
    "Leeds United": "Leeds",
    "Burnley FC": "Burnley",
    "Sunderland AFC": "Sunderland",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Brentford FC": "Brentford",
    "Leicester City": "Leicester",
    "Atlético Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Sociedad": "Sociedad",
    "Real Betis Balompié": "Betis",
    "Deportivo Alavés": "Alaves",
    "Paris Saint-Germain": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "RC Lens": "Lens",
    "Stade Rennais FC": "Rennes",
    "Stade de Reims": "Reims",
    "OGC Nice": "Nice",
    "RC Strasbourg Alsace": "Strasbourg",
    "AJ Auxerre": "Auxerre",
    "Borussia Dortmund": "Dortmund",
    "Bayern Munich": "Bayern Munich",
    "Bayer Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "Borussia Mönchengladbach": "M'gladbach",
    "VfB Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "SC Freiburg": "Freiburg",
    "1. FSV Mainz 05": "Mainz",
    "FC Augsburg": "Augsburg",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Wolfsburg": "Wolfsburg",
    "1. FC Union Berlin": "Union Berlin",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "FC St. Pauli": "St. Pauli",
    "1. FC Heidenheim 1846": "Heidenheim",
    "VfL Bochum 1848": "Bochum",
    "Holstein Kiel": "Holstein Kiel",
    "AC Milan": "Milan",
    "AS Roma": "Roma",
    "Inter Milan": "Inter",
    "SSC Napoli": "Napoli",
    "Juventus": "Juventus",
    "ACF Fiorentina": "Fiorentina",
    "US Lecce": "Lecce",
    "Udinese Calcio": "Udinese",
    "Torino FC": "Torino",
    "Parma Calcio 1913": "Parma",
    "Cagliari Calcio": "Cagliari",
    "Genoa CFC": "Genoa",
    "Hellas Verona FC": "Verona",
    "Como 1907": "Como",
    "Empoli FC": "Empoli",
    "Bologna FC 1909": "Bologna",
    "Sporting CP": "Sporting CP",
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Galatasaray": "Galatasaray",
    "Bodø/Glimt": "Bodoe/Glimt",
    "Real Valladolid": "Valladolid",
    "RCD Mallorca": "Mallorca",
    "Girona FC": "Girona",
    "Celta de Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    "CA Osasuna": "Osasuna",
    "UD Las Palmas": "Las Palmas",
    "Getafe CF": "Getafe",
    "Wolverhampton Wanderers FC": "Wolves",
    "Nottingham Forest FC": "Nott'm Forest",
}


def _fetch(url, key, ttl=1):
    cf = CACHE_DIR / f"{key}.json"
    if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < ttl:
        with open(cf) as f:
            return json.load(f)
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "OracleV2/2.3"})
        if r.status_code == 200:
            d = r.json()
            with open(cf, "w") as f:
                json.dump(d, f)
            return d
    except Exception:
        pass
    return None


def _parse_espn(data, league):
    matches = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        teams = comp.get("competitors", [])
        if len(teams) != 2:
            continue
        h = [t for t in teams if t.get("homeAway") == "home"]
        a = [t for t in teams if t.get("homeAway") == "away"]
        if not h or not a:
            continue
        h, a = h[0], a[0]
        is_sched = "SCHEDULED" in status.upper()
        is_live = "PROGRESS" in status.upper()
        m = {
            "home": h.get("team", {}).get("displayName", ""),
            "away": a.get("team", {}).get("displayName", ""),
            "league": league,
            "date": ev.get("date", "")[:10],
            "time": ev.get("date", ""),
            "venue": comp.get("venue", {}).get("fullName", ""),
            "status": "scheduled" if is_sched else "live" if is_live else "finished",
            "score_home": h.get("score") if not is_sched else None,
            "score_away": a.get("score") if not is_sched else None,
        }
        # Odds
        odds = comp.get("odds", [])
        if odds and isinstance(odds, list) and odds[0] is not None:
            m["odds_detail"] = odds[0].get("details", "")
        matches.append(m)
    return matches


def fetch_football_data_org(api_key, competition="PL"):
    """Fetch upcoming matches from football-data.org (needs key)."""
    COMPS = {"EPL": "PL", "La Liga": "PD", "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1", "UCL": "CL"}
    code = COMPS.get(competition, competition)
    try:
        url = f"https://api.football-data.org/v4/competitions/{code}/matches?status=SCHEDULED"
        r = requests.get(url, headers={"X-Auth-Token": api_key}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            matches = []
            for m in data.get("matches", []):
                matches.append(
                    {
                        "home": m["homeTeam"]["name"],
                        "away": m["awayTeam"]["name"],
                        "league": competition,
                        "date": m.get("utcDate", "")[:10],
                        "matchday": m.get("matchday"),
                        "stage": m.get("stage", ""),
                    }
                )
            return matches
    except Exception:
        pass
    return []


def fetch_upcoming_football(leagues=None, days_ahead=14):
    """Fetch all upcoming football matches — checks every day."""
    if leagues is None:
        leagues = FOOTBALL_LEAGUES
    all_matches = []
    seen = set()
    for lg, path in leagues.items():
        for days in range(days_ahead + 1):
            dt = (datetime.now() + timedelta(days=days)).strftime("%Y%m%d")
            data = _fetch(f"{ESPN}/{path}/scoreboard?dates={dt}", f"espn_{lg}_{dt}", 1 if days == 0 else 6)
            if not data:
                continue
            for m in _parse_espn(data, lg):
                key = f"{m['home']}_{m['away']}_{m['date']}"
                if key not in seen and m["status"] == "scheduled":
                    seen.add(key)
                    all_matches.append(m)
    all_matches.sort(key=lambda x: x.get("date", ""))
    return all_matches


# ESPN scoreboard header API returns ALL live/upcoming football in one call
ESPN_FOOTBALL_HEADER = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=soccer"


def fetch_all_live_football():
    """Fetch all live and scheduled football matches across ALL competitions.

    Uses the ESPN scoreboard header API which returns every active football
    league (top-5, internationals, women's, lower divisions, etc.) in one request.
    Returns matches sorted with live games first.
    """
    matches = []
    seen = set()

    data = _fetch(ESPN_FOOTBALL_HEADER, "espn_football_header", ttl=0.05)
    if not data:
        return matches

    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            league_name = league.get("name", "Unknown")
            league_id = league.get("id", "")
            for ev in league.get("events", []):
                competitors = ev.get("competitors", [])
                if len(competitors) != 2:
                    continue
                status_info = ev.get("fullStatus", {}).get("type", {})
                status_state = status_info.get("state", "").lower()
                status_desc = status_info.get("description", "").lower()
                is_scheduled = status_state == "pre"
                is_live = status_state == "in"
                is_finished = status_state == "post"
                home = competitors[0].get("displayName", "?")
                away = competitors[1].get("displayName", "?")
                key = f"{home}_{away}_{ev.get('date', '')[:10]}"
                if key in seen:
                    continue
                seen.add(key)
                score_h = competitors[0].get("score", "")
                score_a = competitors[1].get("score", "")
                m = {
                    "home": home,
                    "away": away,
                    "league": league_name,
                    "league_id": league_id,
                    "date": ev.get("date", "")[:10],
                    "time": ev.get("date", ""),
                    "status": "live" if is_live else "scheduled" if is_scheduled else "finished",
                    "status_detail": ev.get("summary", ""),
                    "score_home": score_h if not is_scheduled else None,
                    "score_away": score_a if not is_scheduled else None,
                }
                matches.append(m)

    matches.sort(key=lambda x: (x["status"] != "live", x["status"] != "scheduled", x.get("date", "")))
    return matches


def format_for_oracle(matches):
    """Map ESPN names to football-data.co.uk names for Elo lookup."""
    out = []
    for m in matches:
        out.append(
            {
                "home": NAME_MAP.get(m["home"], m["home"]),
                "away": NAME_MAP.get(m["away"], m["away"]),
                "league": m["league"],
                "date": m["date"],
                "venue": m.get("venue", ""),
                "sr_prob": {"H": 33.3, "D": 33.3, "A": 33.3},
            }
        )
    return out


# ═══════════════════════════════════════════════════════════════════════
# CRICKET FIXTURE FETCHER — All formats via ESPN
# ═══════════════════════════════════════════════════════════════════════

# ESPN league IDs for cricket — scoreboard endpoint
CRICKET_LEAGUES = {
    # Domestic T20 franchise leagues
    8048: "IPL",
    8044: "BBL",
    8679: "PSL",
    # International (ICC)
    # These use the header API fallback since individual IDs are unreliable
}

# ESPN scoreboard header API returns ALL live/upcoming cricket in one call
ESPN_CRICKET_HEADER = "https://site.web.api.espn.com/apis/v2/scoreboard/header?sport=cricket"


def fetch_upcoming_cricket():
    """Fetch all upcoming/live cricket matches across all formats.

    Uses the ESPN scoreboard header API which returns every active cricket
    league (international + domestic) in a single request.
    """
    matches = []
    seen = set()

    # Primary source: ESPN header API (returns all leagues at once)
    data = _fetch(ESPN_CRICKET_HEADER, "espn_cricket_header", ttl=0.25)
    if data:
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                league_name = league.get("name", "Unknown")
                league_id = league.get("id", "")
                for ev in league.get("events", []):
                    competitors = ev.get("competitors", [])
                    if len(competitors) != 2:
                        continue
                    status_info = ev.get("fullStatus", {}).get("type", {})
                    status_state = status_info.get("state", "").lower()
                    status_desc = status_info.get("description", "").lower()
                    is_scheduled = status_state == "pre" or "scheduled" in status_desc
                    is_live = status_state == "in" or "live" in status_desc
                    team_a = competitors[0].get("displayName", "?")
                    team_b = competitors[1].get("displayName", "?")
                    key = f"{team_a}_{team_b}_{ev.get('date', '')[:10]}"
                    if key in seen:
                        continue
                    seen.add(key)
                    score_a = competitors[0].get("score", "")
                    score_b = competitors[1].get("score", "")
                    m = {
                        "match": f"{team_a} vs {team_b}",
                        "team_a": team_a,
                        "team_b": team_b,
                        "date": ev.get("date", "")[:10],
                        "time": ev.get("date", ""),
                        "league": league_name,
                        "league_id": league_id,
                        "status": "scheduled" if is_scheduled else "live" if is_live else "finished",
                        "status_detail": ev.get("summary", ""),
                        "score_a": score_a if not is_scheduled else None,
                        "score_b": score_b if not is_scheduled else None,
                    }
                    matches.append(m)

    # Fallback: hit individual league scoreboards for key domestic T20 leagues
    # (in case header API misses recently added leagues)
    for league_id, league_name in CRICKET_LEAGUES.items():
        url = f"https://site.api.espn.com/apis/site/v2/sports/cricket/{league_id}/scoreboard"
        data = _fetch(url, f"espn_cricket_{league_id}", ttl=0.5)
        if not data:
            continue
        for ev in data.get("events", []):
            comp = ev.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {}).get("name", "")
            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue
            team_a = teams[0].get("team", {}).get("displayName", "?")
            team_b = teams[1].get("team", {}).get("displayName", "?")
            key = f"{team_a}_{team_b}_{ev.get('date', '')[:10]}"
            if key in seen:
                continue
            seen.add(key)
            is_scheduled = "SCHEDULED" in status.upper()
            is_live = "PROGRESS" in status.upper()
            m = {
                "match": f"{team_a} vs {team_b}",
                "team_a": team_a,
                "team_b": team_b,
                "date": ev.get("date", "")[:10],
                "time": ev.get("date", ""),
                "league": league_name,
                "league_id": str(league_id),
                "venue": comp.get("venue", {}).get("fullName", ""),
                "status": "scheduled" if is_scheduled else "live" if is_live else "finished",
                "status_detail": comp.get("status", {}).get("type", {}).get("detail", ""),
                "score_a": teams[0].get("score") if not is_scheduled else None,
                "score_b": teams[1].get("score") if not is_scheduled else None,
            }
            matches.append(m)

    matches.sort(key=lambda x: (x.get("status") != "live", x.get("date", "")))
    return matches


def identify_ucl_two_legs(matches):
    """Identify UCL two-leg ties from fixture list."""
    ties = {}
    for m in matches:
        if m["league"] != "UCL":
            continue
        pair = tuple(sorted([m["home"], m["away"]]))
        if pair not in ties:
            ties[pair] = []
        ties[pair].append(m)
    two_legs = [{"teams": list(k), "legs": v, "is_two_leg": len(v) == 2} for k, v in ties.items()]
    return two_legs


def fetch_and_format():
    """Main entry: fetch all upcoming → format for Oracle."""
    print("  📡 Fetching live fixtures from ESPN API (no key needed)...")
    raw = fetch_upcoming_football()
    print(f"    ✅ {len(raw)} upcoming matches across {len({m['league'] for m in raw})} leagues")
    by_lg = {}
    for m in raw:
        if m["league"] not in by_lg:
            by_lg[m["league"]] = []
        by_lg[m["league"]].append(m)
    for lg, ms in sorted(by_lg.items()):
        print(f"    {lg}: {len(ms)} matches")
        for m in ms[:3]:
            print(f"      {m['date']} {m['home']:<25} vs {m['away']}")
        if len(ms) > 3:
            print(f"      ... +{len(ms) - 3} more")
    # Check for football-data.org key
    try:
        from env_loader import FOOTBALL_DATA_KEY

        if FOOTBALL_DATA_KEY:
            print("    📡 Also fetching from football-data.org (key found)...")
            for comp in ["EPL", "La Liga", "UCL"]:
                extra = fetch_football_data_org(FOOTBALL_DATA_KEY, comp)
                if extra:
                    print(f"      +{len(extra)} from {comp}")
    except Exception:
        pass
    return format_for_oracle(raw), raw


if __name__ == "__main__":
    m, r = fetch_and_format()
    print(f"\n  {len(m)} matches ready for Oracle")
    ucl = identify_ucl_two_legs(r)
    if ucl:
        print(f"  UCL two-leg ties: {len(ucl)}")
