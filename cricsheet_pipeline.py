"""
Oracle V2 — CricSheet Auto-Pipeline
Parses 3,196 T20I matches ball-by-ball to auto-generate player stats.
Computes: batting avg/SR, bowling avg/econ/SR, phase splits, venue splits,
vs-team records, recent form, impact ratings — for EVERY player.
"""

import csv
import glob
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "cricsheet_data"
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# DOMESTIC T20 LEAGUE SUPPORT
# CricSheet provides ball-by-ball data for IPL, BBL, CPL, PSL, and more.
# ═══════════════════════════════════════════════════════════════════════

DOMESTIC_LEAGUES = {
    "ipl": {
        "name": "Indian Premier League",
        "zip_url": "https://cricsheet.org/downloads/ipl_male_csv2.zip",
        "event_filter": "Indian Premier League",
        "country": "India",
    },
    "bbl": {
        "name": "Big Bash League",
        "zip_url": "https://cricsheet.org/downloads/bbl_male_csv2.zip",
        "event_filter": "Big Bash League",
        "country": "Australia",
    },
    "cpl": {
        "name": "Caribbean Premier League",
        "zip_url": "https://cricsheet.org/downloads/cpl_male_csv2.zip",
        "event_filter": "Caribbean Premier League",
        "country": "West Indies",
    },
    "psl": {
        "name": "Pakistan Super League",
        "zip_url": "https://cricsheet.org/downloads/psl_male_csv2.zip",
        "event_filter": "Pakistan Super League",
        "country": "Pakistan",
    },
    "the_hundred": {
        "name": "The Hundred",
        "zip_url": "https://cricsheet.org/downloads/the_hundred_male_csv2.zip",
        "event_filter": "The Hundred",
        "country": "England",
    },
    "sa20": {
        "name": "SA20",
        "zip_url": "https://cricsheet.org/downloads/sa20_male_csv2.zip",
        "event_filter": "SA20",
        "country": "South Africa",
    },
}


def download_league_data(league_key: str, data_dir: str | None = None) -> str:
    """Download CricSheet data for a specific domestic T20 league.

    Returns the directory path where data was extracted.
    """
    import zipfile

    import requests

    if league_key not in DOMESTIC_LEAGUES:
        print(f"  Unknown league: {league_key}. Available: {list(DOMESTIC_LEAGUES.keys())}")
        return ""

    league = DOMESTIC_LEAGUES[league_key]
    if data_dir is None:
        data_dir = str(Path(os.path.dirname(os.path.abspath(__file__))) / f"cricsheet_{league_key}")

    league_dir = Path(data_dir)
    if league_dir.exists() and len(list(league_dir.glob("*_info.csv"))) > 10:
        print(f"  ✅ {league['name']}: {len(list(league_dir.glob('*_info.csv')))} matches cached")
        return str(league_dir)

    print(f"  📥 Downloading {league['name']} data from CricSheet...")
    league_dir.mkdir(parents=True, exist_ok=True)

    try:
        r = requests.get(league["zip_url"], timeout=120)
        zip_path = league_dir.parent / f"cricsheet_{league_key}_temp.zip"
        zip_path.write_bytes(r.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(league_dir)
        zip_path.unlink()
        n_matches = len(list(league_dir.glob("*_info.csv")))
        print(f"    ✅ {n_matches} {league['name']} matches extracted")
        return str(league_dir)
    except Exception as e:
        print(f"    ⚠️ Failed to download {league['name']}: {e}")
        return ""


def build_league_database(league_key: str, min_year: int = 2022) -> tuple[dict, int]:
    """Build player database for a specific domestic T20 league.

    Returns (players_dict, matches_parsed) just like build_player_database.
    """
    data_dir = download_league_data(league_key)
    if not data_dir:
        return {}, 0

    # For domestic leagues, we don't filter by squads — include all players
    return build_player_database(data_dir, min_year=min_year, target_teams=None)


# T20 WC 2026 squads (all 8 Super 8 teams + 12 group stage teams = 20 teams)
WC_SQUADS = {
    "India": [
        "Abhishek Sharma",
        "Sanju Samson",
        "Ishan Kishan",
        "Suryakumar Yadav",
        "Tilak Varma",
        "Hardik Pandya",
        "Shivam Dube",
        "Axar Patel",
        "Varun Chakaravarthy",
        "Arshdeep Singh",
        "Jasprit Bumrah",
        "Kuldeep Yadav",
        "Washington Sundar",
        "Rinku Singh",
        "Mohammed Siraj",
    ],
    "New Zealand": [
        "Tim Seifert",
        "Finn Allen",
        "Rachin Ravindra",
        "Glenn Phillips",
        "Mark Chapman",
        "Daryl Mitchell",
        "James Neesham",
        "Mitchell Santner",
        "Cole McConchie",
        "Matt Henry",
        "Lockie Ferguson",
        "Jacob Duffy",
        "Ish Sodhi",
        "Adam Milne",
        "Devon Conway",
    ],
    "England": [
        "Phil Salt",
        "Harry Brook",
        "Jacob Bethell",
        "Jos Buttler",
        "Ben Duckett",
        "Will Jacks",
        "Sam Curran",
        "Rehan Ahmed",
        "Adil Rashid",
        "Jofra Archer",
        "Mark Wood",
        "Josh Tongue",
        "Liam Dawson",
        "Tom Banton",
        "Brydon Carse",
    ],
    "South Africa": [
        "Quinton de Kock",
        "Aiden Markram",
        "Tristan Stubbs",
        "Heinrich Klaasen",
        "David Miller",
        "Dewald Brevis",
        "Marco Jansen",
        "Kagiso Rabada",
        "Anrich Nortje",
        "Lungi Ngidi",
        "Tabraiz Shamsi",
        "Keshav Maharaj",
        "Corbin Bosch",
        "Ryan Rickelton",
        "George Linde",
    ],
    "West Indies": [
        "Brandon King",
        "Johnson Charles",
        "Shai Hope",
        "Shimron Hetmyer",
        "Rovman Powell",
        "Sherfane Rutherford",
        "Roston Chase",
        "Romario Shepherd",
        "Akeal Hosein",
        "Gudakesh Motie",
        "Shamar Joseph",
        "Jayden Seales",
        "Jason Holder",
        "Matthew Forde",
        "Quentin Sampson",
    ],
    "Pakistan": [
        "Babar Azam",
        "Sahibzada Farhan",
        "Fakhar Zaman",
        "Mohammad Rizwan",
        "Salman Ali Agha",
        "Shadab Khan",
        "Faheem Ashraf",
        "Shaheen Afridi",
        "Naseem Shah",
        "Haris Rauf",
        "Usman Tariq",
        "Imad Wasim",
        "Mohammad Nawaz",
        "Azam Khan",
        "Iftikhar Ahmed",
    ],
    "Sri Lanka": [
        "Pathum Nissanka",
        "Kusal Mendis",
        "Charith Asalanka",
        "Kamindu Mendis",
        "Dasun Shanaka",
        "Wanindu Hasaranga",
        "Maheesh Theekshana",
        "Dushmantha Chameera",
        "Dushan Hemantha",
        "Pavan Rathnayake",
        "Dunith Wellalage",
        "Matheesha Pathirana",
        "Bhanuka Rajapaksa",
        "Nuwan Thushara",
        "Asitha Fernando",
    ],
    "Zimbabwe": [
        "Brian Bennett",
        "Sikandar Raza",
        "Craig Ervine",
        "Sean Williams",
        "Tony Munyonga",
        "Ryan Burl",
        "Luke Jongwe",
        "Blessing Muzarabani",
        "Tendai Chatara",
        "Wellington Masakadza",
        "Richard Ngarava",
        "Clive Madande",
        "Milton Shumba",
        "Dion Myers",
        "Brad Evans",
    ],
    "Australia": [
        "Mitchell Marsh",
        "Travis Head",
        "Glenn Maxwell",
        "David Warner",
        "Marcus Stoinis",
        "Tim David",
        "Matthew Wade",
        "Pat Cummins",
        "Mitchell Starc",
        "Adam Zampa",
        "Josh Hazlewood",
        "Josh Inglis",
        "Cameron Green",
        "Ben Dwarshuis",
        "Spencer Johnson",
    ],
    "Afghanistan": [
        "Rashid Khan",
        "Ibrahim Zadran",
        "Rahmanullah Gurbaz",
        "Najibullah Zadran",
        "Mohammad Nabi",
        "Azmatullah Omarzai",
        "Naveen-ul-Haq",
        "Fazalhaq Farooqi",
        "Gulbadin Naib",
        "Mujeeb Ur Rahman",
        "Noor Ahmad",
        "Karim Janat",
        "Hazratullah Zazai",
        "Sediqullah Atal",
        "Fareed Ahmad",
    ],
}

# Name matching aliases (CricSheet uses different name formats)
NAME_ALIASES = {
    "SKY": "Suryakumar Yadav",
    "SV Samson": "Sanju Samson",
    "JJ Bumrah": "Jasprit Bumrah",
    "HH Pandya": "Hardik Pandya",
    "Arshdeep Singh": "Arshdeep Singh",
    "RA Jadeja": "Ravindra Jadeja",
    "KL Rahul": "KL Rahul",
    "V Kohli": "Virat Kohli",
    "RG Sharma": "Rohit Sharma",
    "TA Boult": "Trent Boult",
    "MJ Santner": "Mitchell Santner",
    "FH Allen": "Finn Allen",
    "TL Seifert": "Tim Seifert",
    "GD Phillips": "Glenn Phillips",
    "DJ Mitchell": "Daryl Mitchell",
    "HC Brook": "Harry Brook",
    "PD Salt": "Phil Salt",
    "JC Buttler": "Jos Buttler",
    "JC Archer": "Jofra Archer",
    "MA Wood": "Mark Wood",
    "AU Rashid": "Adil Rashid",
    "Q de Kock": "Quinton de Kock",
    "AK Markram": "Aiden Markram",
    "H Klaasen": "Heinrich Klaasen",
    "KG Rabada": "Kagiso Rabada",
    "A Nortje": "Anrich Nortje",
    "M Jansen": "Marco Jansen",
    "Babar Azam": "Babar Azam",
    "Shaheen Shah Afridi": "Shaheen Afridi",
    "Mohammad Rizwan": "Mohammad Rizwan",
    "Shadab Khan": "Shadab Khan",
    "Rashid Khan": "Rashid Khan",
    "Ibrahim Zadran": "Ibrahim Zadran",
    "Rahmanullah Gurbaz": "Rahmanullah Gurbaz",
    "P Nissanka": "Pathum Nissanka",
    "BKG Mendis": "Kusal Mendis",
    "W Hasaranga": "Wanindu Hasaranga",
}


def get_phase(ball_num: float) -> str:
    """Classify ball into T20 phase."""
    over = int(ball_num)
    if over < 6:
        return "powerplay"
    elif over < 15:
        return "middle"
    else:
        return "death"


def parse_info_file(filepath: str) -> dict:
    """Parse match info CSV."""
    info = {
        "teams": [],
        "date": "",
        "venue": "",
        "winner": "",
        "toss_winner": "",
        "toss_decision": "",
        "season": "",
        "event": "",
    }
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                if row[0] == "info":
                    key, val = row[1], row[2]
                    if key == "team":
                        info["teams"].append(val)
                    elif key == "date":
                        info["date"] = val
                    elif key == "venue":
                        info["venue"] = val
                    elif key == "winner":
                        info["winner"] = val
                    elif key == "toss_winner":
                        info["toss_winner"] = val
                    elif key == "toss_decision":
                        info["toss_decision"] = val
                    elif key == "season":
                        info["season"] = val
                    elif key == "event":
                        info["event"] = val
    except Exception:
        pass
    return info


# CricSheet csv2 column layouts. Some zips ship ball-by-ball files WITHOUT a
# header row — the standard documented layout has 22 columns, and an extended
# variant observed in 2026 downloads has 27 (extra over_ball, non_boundary,
# and fielder columns). Mapped empirically against known matches.
_CSV2_COLS_22 = [
    "match_id",
    "season",
    "start_date",
    "venue",
    "innings",
    "ball",
    "batting_team",
    "bowling_team",
    "striker",
    "non_striker",
    "bowler",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "wicket_type",
    "player_dismissed",
    "other_wicket_type",
    "other_player_dismissed",
]
_CSV2_COLS_27 = [
    "match_id",
    "season",
    "start_date",
    "venue",
    "innings",
    "ball",
    "over_ball",
    "batting_team",
    "bowling_team",
    "striker",
    "non_striker",
    "bowler",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "non_boundary",
    "wicket_type",
    "player_dismissed",
    "other_wicket_type",
    "other_player_dismissed",
    "fielder1",
    "fielder2",
    "unused",
]


def parse_ball_data(filepath: str) -> list[dict]:
    """Parse ball-by-ball CSV, handling both headered and headerless files."""
    balls = []
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
            f.seek(0)
            if "striker" in first_line:
                reader = csv.DictReader(f)
            else:
                # Headerless file — pick column layout by field count
                n_cols = first_line.count(",") + 1
                fieldnames = _CSV2_COLS_27 if n_cols >= 25 else _CSV2_COLS_22
                reader = csv.DictReader(f, fieldnames=fieldnames)
            for row in reader:
                try:
                    balls.append(
                        {
                            "innings": int(row.get("innings", 0)),
                            "ball": float(row.get("ball", 0)),
                            "batting_team": row.get("batting_team", ""),
                            "bowling_team": row.get("bowling_team", ""),
                            "striker": row.get("striker", ""),
                            "bowler": row.get("bowler", ""),
                            "runs": int(row.get("runs_off_bat", 0)),
                            "extras": int(row.get("extras", 0)),
                            "wides": int(row.get("wides", 0) or 0),
                            "noballs": int(row.get("noballs", 0) or 0),
                            "wicket_type": row.get("wicket_type", ""),
                            "dismissed": row.get("player_dismissed", ""),
                        }
                    )
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    return balls


def build_player_database(data_dir: str, min_year: int = 2022, target_teams: dict | None = None) -> dict:
    """
    Parse ALL CricSheet ball-by-ball data and build comprehensive player stats.
    Returns dict[player_name] -> full stats.
    """
    # Build set of target player names for fast lookup
    target_players = set()
    player_team_map = {}
    if target_teams:
        for team, players in target_teams.items():
            for p in players:
                target_players.add(p)
                player_team_map[p] = team

    # Also add aliases
    for alias, real in NAME_ALIASES.items():
        if real in target_players:
            target_players.add(alias)

    # Stats accumulators
    batting = defaultdict(
        lambda: {
            "innings": 0,
            "runs": 0,
            "balls": 0,
            "dismissals": 0,
            "fours": 0,
            "sixes": 0,
            "not_outs": 0,
            "by_phase": defaultdict(lambda: {"runs": 0, "balls": 0}),
            "by_venue": defaultdict(lambda: {"runs": 0, "balls": 0, "innings": 0}),
            "by_opponent": defaultdict(lambda: {"runs": 0, "balls": 0, "innings": 0}),
            "recent_scores": [],
            "matches": set(),
        }
    )

    bowling = defaultdict(
        lambda: {
            "balls": 0,
            "runs_conceded": 0,
            "wickets": 0,
            "wides": 0,
            "noballs": 0,
            "by_phase": defaultdict(lambda: {"balls": 0, "runs": 0, "wickets": 0}),
            "by_venue": defaultdict(lambda: {"balls": 0, "runs": 0, "wickets": 0}),
            "by_opponent": defaultdict(lambda: {"balls": 0, "runs": 0, "wickets": 0}),
            "recent_wickets": [],
            "matches": set(),
        }
    )

    # Find all match files
    info_files = sorted(glob.glob(os.path.join(data_dir, "*_info.csv")))
    print(f"  Found {len(info_files)} match info files")

    matches_parsed = 0
    recent_matches = 0

    for info_path in info_files:
        info = parse_info_file(info_path)
        # Filter by year
        try:
            year = int(info["date"][:4])
        except (ValueError, IndexError):
            continue

        if year < min_year:
            continue

        match_id = os.path.basename(info_path).replace("_info.csv", "")
        ball_path = info_path.replace("_info.csv", ".csv")
        if not os.path.exists(ball_path):
            continue

        balls = parse_ball_data(ball_path)
        if not balls:
            continue

        matches_parsed += 1
        recent_matches += 1
        venue = info.get("venue", "Unknown")

        # Track innings per batter per match
        match_bat_runs = defaultdict(int)
        match_bat_balls = defaultdict(int)

        for b in balls:
            striker = b["striker"]
            bowler = b["bowler"]
            phase = get_phase(b["ball"])
            opp_team_bat = b["bowling_team"]
            opp_team_bowl = b["batting_team"]

            # ── Batting stats ──
            bat = batting[striker]
            bat["matches"].add(match_id)
            bat["runs"] += b["runs"]
            if b["wides"] == 0:  # wides don't count as balls faced
                bat["balls"] += 1
                bat["by_phase"][phase]["balls"] += 1
            bat["by_phase"][phase]["runs"] += b["runs"]
            bat["by_venue"][venue]["runs"] += b["runs"]
            bat["by_venue"][venue]["balls"] += 1
            bat["by_opponent"][opp_team_bat]["runs"] += b["runs"]
            bat["by_opponent"][opp_team_bat]["balls"] += 1
            match_bat_runs[striker] += b["runs"]
            match_bat_balls[striker] += 1 if b["wides"] == 0 else 0
            if b["runs"] == 4:
                bat["fours"] += 1
            if b["runs"] == 6:
                bat["sixes"] += 1

            # Dismissal
            if b["dismissed"] and b["dismissed"] == striker:
                bat["dismissals"] += 1

            # ── Bowling stats ──
            bwl = bowling[bowler]
            bwl["matches"].add(match_id)
            legal = 1 if b["wides"] == 0 and b["noballs"] == 0 else 0
            bwl["balls"] += legal
            bwl["runs_conceded"] += b["runs"] + b["extras"]
            bwl["wides"] += b["wides"]
            bwl["noballs"] += b["noballs"]
            bwl["by_phase"][phase]["balls"] += legal
            bwl["by_phase"][phase]["runs"] += b["runs"] + b["extras"]
            bwl["by_venue"][venue]["balls"] += legal
            bwl["by_venue"][venue]["runs"] += b["runs"] + b["extras"]
            bwl["by_opponent"][opp_team_bowl]["balls"] += legal
            bwl["by_opponent"][opp_team_bowl]["runs"] += b["runs"] + b["extras"]

            if b["wicket_type"] and b["wicket_type"] not in (
                "run out",
                "retired hurt",
                "retired out",
                "obstructing the field",
            ):
                bwl["wickets"] += 1
                bwl["by_phase"][phase]["wickets"] += 1
                bwl["by_venue"][venue]["wickets"] = bwl["by_venue"][venue].get("wickets", 0) + 1
                bwl["by_opponent"][opp_team_bowl]["wickets"] = bwl["by_opponent"][opp_team_bowl].get("wickets", 0) + 1

        # Track innings scores for batters
        for striker, runs in match_bat_runs.items():
            batting[striker]["innings"] += 1
            batting[striker]["recent_scores"].append(runs)
            batting[striker]["by_venue"][venue]["innings"] += 1
            # Keep only last 10
            batting[striker]["recent_scores"] = batting[striker]["recent_scores"][-10:]

        # Track wickets per match for bowlers
        # (already tracked above)

    print(f"  Parsed {matches_parsed} matches from {min_year}+")
    print(f"  Unique batters: {len(batting)}, Unique bowlers: {len(bowling)}")

    # ── Build final player database ──
    players = {}

    all_cricsheet_names = set(batting.keys()) | set(bowling.keys())

    # If no target teams specified, build database for ALL players with sufficient data
    if target_teams is None:
        for player_name in all_cricsheet_names:
            bat = batting.get(player_name)
            bwl = bowling.get(player_name)
            has_bat = bat and bat["balls"] > 30
            has_bowl = bwl and bwl["balls"] > 30
            if not has_bat and not has_bowl:
                continue

            p = {"name": player_name, "team": "", "cricsheet_name": player_name, "role": "unknown"}

            if has_bat:
                avg = bat["runs"] / max(bat["dismissals"], 1)
                sr = (bat["runs"] / bat["balls"]) * 100
                p["batting"] = {
                    "innings": bat["innings"],
                    "runs": bat["runs"],
                    "balls_faced": bat["balls"],
                    "average": round(avg, 2),
                    "strike_rate": round(sr, 2),
                    "matches": len(bat["matches"]),
                    "recent_scores": bat["recent_scores"][-5:],
                }
            if has_bowl:
                overs = bwl["balls"] / 6
                econ = bwl["runs_conceded"] / overs if overs > 0 else 0
                p["bowling"] = {
                    "balls": bwl["balls"],
                    "wickets": bwl["wickets"],
                    "economy": round(econ, 2),
                    "matches": len(bwl["matches"]),
                }

            if has_bat and has_bowl:
                p["role"] = "allrounder"
            elif has_bat:
                p["role"] = "batter"
            elif has_bowl:
                p["role"] = "bowler"

            impact = 50
            if has_bat:
                impact = max(impact, min(p["batting"]["strike_rate"] / 2, 45) + min(p["batting"]["average"] / 2, 30))
            if has_bowl:
                impact = max(impact, max(0, 40 - p["bowling"]["economy"] * 3) + min(p["bowling"]["wickets"] / 3, 25))
            p["impact_rating"] = round(min(impact, 99), 1)

            players[player_name] = p
        return players, matches_parsed

    # Process players that appear in target squads
    for team_name, squad in target_teams.items():
        for player_name in squad:
            # Find matching CricSheet name
            cs_name = None
            for alias, real in NAME_ALIASES.items():
                if real == player_name and alias in all_cricsheet_names:
                    cs_name = alias
                    break
            if not cs_name:
                # Try direct match or partial match
                for n in all_cricsheet_names:
                    surname = player_name.split()[-1] if player_name.split() else ""
                    if n == player_name or (surname and surname in n and len(surname) > 3):
                        cs_name = n
                        break

            bat = batting.get(cs_name, batting.get(player_name, None))
            bwl = bowling.get(cs_name, bowling.get(player_name, None))

            p = {
                "name": player_name,
                "team": team_name,
                "cricsheet_name": cs_name or "NOT_FOUND",
                "role": "unknown",
            }

            # Batting
            if bat and bat["balls"] > 0:
                avg = bat["runs"] / max(bat["dismissals"], 1)
                sr = (bat["runs"] / bat["balls"]) * 100 if bat["balls"] else 0
                p["batting"] = {
                    "innings": bat["innings"],
                    "runs": bat["runs"],
                    "balls_faced": bat["balls"],
                    "dismissals": bat["dismissals"],
                    "average": round(avg, 2),
                    "strike_rate": round(sr, 2),
                    "fours": bat["fours"],
                    "sixes": bat["sixes"],
                    "matches": len(bat["matches"]),
                    "recent_scores": bat["recent_scores"][-5:],
                    "form_avg_last5": round(sum(bat["recent_scores"][-5:]) / max(len(bat["recent_scores"][-5:]), 1), 1),
                }

                # Phase splits
                p["batting"]["phases"] = {}
                for phase in ["powerplay", "middle", "death"]:
                    pd = bat["by_phase"].get(phase, {"runs": 0, "balls": 0})
                    if pd["balls"] > 0:
                        p["batting"]["phases"][phase] = {
                            "runs": pd["runs"],
                            "balls": pd["balls"],
                            "sr": round((pd["runs"] / pd["balls"]) * 100, 1),
                        }

                # Top venues
                p["batting"]["venues"] = {}
                top_venues = sorted(bat["by_venue"].items(), key=lambda x: x[1].get("innings", 0), reverse=True)[:5]
                for vname, vd in top_venues:
                    if vd.get("innings", 0) >= 1:
                        p["batting"]["venues"][vname] = {
                            "innings": vd.get("innings", 0),
                            "runs": vd["runs"],
                            "avg": round(vd["runs"] / max(vd.get("innings", 1), 1), 1),
                        }

                # Vs opponents
                p["batting"]["vs_team"] = {}
                for opp, od in sorted(bat["by_opponent"].items(), key=lambda x: x[1].get("innings", 0), reverse=True)[
                    :8
                ]:
                    if od["balls"] > 5:
                        p["batting"]["vs_team"][opp] = {
                            "runs": od["runs"],
                            "balls": od["balls"],
                            "sr": round((od["runs"] / od["balls"]) * 100, 1),
                        }

            # Bowling
            if bwl and bwl["balls"] > 0:
                overs = bwl["balls"] / 6
                econ = bwl["runs_conceded"] / overs if overs > 0 else 0
                avg = bwl["runs_conceded"] / max(bwl["wickets"], 1)
                bowl_sr = bwl["balls"] / max(bwl["wickets"], 1)
                p["bowling"] = {
                    "balls": bwl["balls"],
                    "overs": round(overs, 1),
                    "runs_conceded": bwl["runs_conceded"],
                    "wickets": bwl["wickets"],
                    "economy": round(econ, 2),
                    "average": round(avg, 2),
                    "strike_rate": round(bowl_sr, 2),
                    "matches": len(bwl["matches"]),
                }

                # Phase splits
                p["bowling"]["phases"] = {}
                for phase in ["powerplay", "middle", "death"]:
                    pd = bwl["by_phase"].get(phase, {"balls": 0, "runs": 0, "wickets": 0})
                    if pd["balls"] > 0:
                        overs_p = pd["balls"] / 6
                        p["bowling"]["phases"][phase] = {
                            "balls": pd["balls"],
                            "runs": pd["runs"],
                            "wickets": pd["wickets"],
                            "econ": round(pd["runs"] / overs_p, 2) if overs_p > 0 else 0,
                        }

                # Vs opponents
                p["bowling"]["vs_team"] = {}
                for opp, od in sorted(bwl["by_opponent"].items(), key=lambda x: x[1]["balls"], reverse=True)[:8]:
                    if od["balls"] > 6:
                        overs_o = od["balls"] / 6
                        p["bowling"]["vs_team"][opp] = {
                            "balls": od["balls"],
                            "runs": od["runs"],
                            "wickets": od.get("wickets", 0),
                            "econ": round(od["runs"] / overs_o, 2) if overs_o > 0 else 0,
                        }

            # Determine role
            has_bat = "batting" in p and p["batting"]["balls_faced"] > 30
            has_bowl = "bowling" in p and p["bowling"]["balls"] > 30
            if has_bat and has_bowl:
                p["role"] = "allrounder"
            elif has_bat:
                p["role"] = "batter"
            elif has_bowl:
                if bwl and any("spin" in str(bwl).lower() for _ in [1]):
                    p["role"] = "bowler"
                else:
                    p["role"] = "bowler"
            else:
                p["role"] = "unknown"

            # Impact rating (composite)
            impact = 50
            if has_bat:
                bat_impact = min(p["batting"]["strike_rate"] / 2, 45) + min(p["batting"]["average"] / 2, 30)
                impact = max(impact, bat_impact)
            if has_bowl:
                bowl_impact = max(0, 40 - p["bowling"]["economy"] * 3) + min(p["bowling"]["wickets"] / 3, 25)
                impact = max(impact, bowl_impact)
            p["impact_rating"] = round(min(impact, 99), 1)

            players[player_name] = p

    return players, matches_parsed


def main():
    print("=" * 80)
    print("  ORACLE V2 — CRICSHEET AUTO-PIPELINE")
    print("  Parsing ALL T20I ball-by-ball data (2022-2026)")
    print("=" * 80)

    t0 = time.time()
    players, n_matches = build_player_database(str(DATA_DIR), min_year=2022, target_teams=WC_SQUADS)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # ── Stats summary ──
    found = sum(1 for p in players.values() if p.get("cricsheet_name") != "NOT_FOUND")
    not_found = sum(1 for p in players.values() if p.get("cricsheet_name") == "NOT_FOUND")
    total_players = len(players)

    print(f"\n  Players in database: {total_players}")
    print(f"  Matched in CricSheet: {found}")
    print(f"  Not found (new/alias): {not_found}")

    # Team summaries
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  TEAM PLAYER COVERAGE                               │")
    print("  └─────────────────────────────────────────────────────┘")
    for team in sorted(WC_SQUADS.keys()):
        team_players = [p for p in players.values() if p["team"] == team]
        matched = sum(1 for p in team_players if p.get("cricsheet_name") != "NOT_FOUND")
        total_runs = sum(p.get("batting", {}).get("runs", 0) for p in team_players)
        total_wkts = sum(p.get("bowling", {}).get("wickets", 0) for p in team_players)
        top_batter = max(team_players, key=lambda x: x.get("batting", {}).get("runs", 0))
        top_bowler = max(team_players, key=lambda x: x.get("bowling", {}).get("wickets", 0))
        print(
            f"  {team:<16} {matched}/{len(team_players)} matched | "
            f"Runs: {total_runs:>5} | Wkts: {total_wkts:>3} | "
            f"Top bat: {top_batter['name'][:15]} | Top bowl: {top_bowler['name'][:15]}"
        )

    # ── Save full JSON ──
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "CricSheet ball-by-ball T20I data",
            "matches_parsed": n_matches,
            "year_range": "2022-2026",
            "total_players": total_players,
            "matched_in_cricsheet": found,
            "teams": len(WC_SQUADS),
        },
        "teams": {},
        "players": {},
    }

    for team in sorted(WC_SQUADS.keys()):
        team_players = {name: players[name] for name in WC_SQUADS[team] if name in players}
        output["teams"][team] = {
            "squad_size": len(WC_SQUADS[team]),
            "players_with_data": sum(1 for p in team_players.values() if p.get("cricsheet_name") != "NOT_FOUND"),
            "total_runs": sum(p.get("batting", {}).get("runs", 0) for p in team_players.values()),
            "total_wickets": sum(p.get("bowling", {}).get("wickets", 0) for p in team_players.values()),
            "avg_impact": round(
                sum(p.get("impact_rating", 50) for p in team_players.values()) / max(len(team_players), 1), 1
            ),
            "squad": list(team_players.keys()),
        }
        for name, pdata in team_players.items():
            output["players"][name] = pdata

    # Write comprehensive JSON
    json_path = OUTPUT_DIR / "complete_player_database.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📁 Saved: {json_path}")
    print(f"     Size: {json_path.stat().st_size / 1024:.1f} KB")

    # ── Print sample players ──
    print("\n  ┌─────────────────────────────────────────────────────┐")
    print("  │  SAMPLE PLAYER CARDS (auto-generated from CricSheet)│")
    print("  └─────────────────────────────────────────────────────┘")

    sample_players = [
        "Jasprit Bumrah",
        "Sanju Samson",
        "Finn Allen",
        "Mitchell Santner",
        "Harry Brook",
        "Aiden Markram",
        "Pathum Nissanka",
        "Rashid Khan",
        "Babar Azam",
        "Sikandar Raza",
    ]

    for name in sample_players:
        p = players.get(name)
        if not p:
            continue
        print(f"\n  ━━ {name} ({p['team']}) ━━ Impact: {p.get('impact_rating', '?')}")
        print(f"     CricSheet: {p.get('cricsheet_name', 'N/A')}")
        if "batting" in p:
            b = p["batting"]
            print(
                f"     BAT: {b['runs']} runs @ {b['average']} avg, SR {b['strike_rate']} "
                f"({b['innings']} inn, {b['matches']} matches)"
            )
            print(f"          4s: {b['fours']} | 6s: {b['sixes']} | Form (last 5): {b.get('recent_scores', [])}")
            if b.get("phases"):
                for ph, pd in b["phases"].items():
                    print(f"          {ph:>10}: SR {pd['sr']}")
            if b.get("vs_team"):
                for opp, od in list(b["vs_team"].items())[:3]:
                    print(f"          vs {opp}: SR {od['sr']} ({od['runs']}r/{od['balls']}b)")
        if "bowling" in p:
            bw = p["bowling"]
            print(f"     BOWL: {bw['wickets']} wkts @ {bw['average']} avg, Econ {bw['economy']} ({bw['overs']} ov)")
            if bw.get("phases"):
                for ph, pd in bw["phases"].items():
                    print(f"          {ph:>10}: Econ {pd['econ']}, Wkts {pd['wickets']}")
            if bw.get("vs_team"):
                for opp, od in list(bw["vs_team"].items())[:3]:
                    print(f"          vs {opp}: Econ {od['econ']}, Wkts {od.get('wickets', 0)}")

    print(f"\n{'=' * 80}")
    return output


if __name__ == "__main__":
    main()
