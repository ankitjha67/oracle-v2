"""
Oracle V2 — Fan Sentiment Engine
Extracts crowd/betting sentiment from ESPN embedded odds (DraftKings),
The Odds API (40+ bookmakers, if key available), and converts into
prediction-ready features.

Sentiment Signal = "What does the crowd/market believe?"
Model Signal = "What does our Elo/ML engine say?"
Final Prediction = weighted blend of both.
"""

import json, os, time, math, requests
from collections import defaultdict
from pathlib import Path

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# ═══════════════════════════════════════════════════════════════════════
# ESPN ODDS EXTRACTION (Free — DraftKings data embedded in scoreboard)
# ═══════════════════════════════════════════════════════════════════════
def extract_espn_odds(sport_espn_path, dates=None):
    """Extract DraftKings odds embedded in ESPN scoreboard data.

    Returns list of dicts with team names, spread, over/under, and
    implied win probabilities from the betting line.
    """
    if dates is None:
        from datetime import datetime
        dates = [datetime.now().strftime("%Y%m%d")]

    all_odds = []
    for dt in dates:
        url = f"{ESPN}/{sport_espn_path}/scoreboard?dates={dt}"
        cf = CACHE_DIR / f"odds_{sport_espn_path.replace('/','_')}_{dt}.json"
        data = None
        if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < 0.5:
            with open(cf) as f:
                data = json.load(f)
        else:
            try:
                r = requests.get(url, timeout=15, headers={"User-Agent": "OracleV2/3.0"})
                if r.status_code == 200:
                    data = r.json()
                    with open(cf, "w") as f:
                        json.dump(data, f)
            except Exception:
                continue
        if not data:
            continue

        for event in data.get("events", []):
            comp = event.get("competitions", [{}])[0]
            odds_list = comp.get("odds", [])
            teams = comp.get("competitors", [])
            if not odds_list or not odds_list[0] or len(teams) != 2:
                continue

            o = odds_list[0]
            t0, t1 = teams[0], teams[1]
            n0 = t0.get("team", {}).get("displayName", "?")
            n1 = t1.get("team", {}).get("displayName", "?")
            home = n0 if t0.get("homeAway") == "home" else n1

            spread = o.get("spread", 0) or 0
            over_under = o.get("overUnder", 0) or 0

            # Parse home/away odds
            hto = o.get("homeTeamOdds", {})
            ato = o.get("awayTeamOdds", {})
            home_fav = hto.get("favorite", False)
            away_fav = ato.get("favorite", False)
            fav_at_open_h = hto.get("favoriteAtOpen", False)

            # Convert spread to implied win probability
            # NBA/NHL: each point of spread ≈ 3% win probability shift
            # Football: each goal of spread ≈ 15% win probability shift
            if "soccer" in sport_espn_path:
                spread_factor = 15.0  # Goals are rare
            elif "baseball" in sport_espn_path:
                spread_factor = 10.0  # Run lines
            else:
                spread_factor = 3.0  # Points/goals

            # Negative spread = favorite; home team spread perspective
            home_implied = 50.0 + (-spread * spread_factor)
            home_implied = max(5.0, min(95.0, home_implied))  # Clamp

            # Line movement signal
            line_moved = "STABLE"
            if fav_at_open_h and not home_fav:
                line_moved = "AWAY_SHIFT"  # Market moved away from home
            elif not fav_at_open_h and home_fav:
                line_moved = "HOME_SHIFT"  # Market moved toward home

            all_odds.append({
                "home": n0 if t0.get("homeAway") == "home" else n1,
                "away": n1 if t0.get("homeAway") == "home" else n0,
                "spread": float(spread),
                "over_under": float(over_under),
                "home_implied_pct": round(home_implied, 1),
                "away_implied_pct": round(100 - home_implied, 1),
                "home_favorite": home_fav,
                "line_movement": line_moved,
                "provider": o.get("provider", {}).get("name", "Unknown"),
                "date": event.get("date", "")[:10],
            })
    return all_odds


# ═══════════════════════════════════════════════════════════════════════
# THE ODDS API (40+ bookmakers — needs API key)
# ═══════════════════════════════════════════════════════════════════════
def fetch_odds_api(api_key, sport="upcoming", regions="us,uk,eu", markets="h2h,spreads"):
    """Fetch odds from The Odds API (40+ bookmakers). Needs free API key."""
    if not api_key:
        return []

    SPORT_MAP = {
        "EPL": "soccer_epl", "La Liga": "soccer_spain_la_liga",
        "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
        "Ligue 1": "soccer_france_ligue_one", "UCL": "soccer_uefa_champions_league",
        "NBA": "basketball_nba", "NHL": "icehockey_nhl",
        "MLB": "baseball_mlb", "NFL": "americanfootball_nfl",
        "UFC": "mma_mixed_martial_arts",
    }

    all_odds = []
    for league, api_sport in SPORT_MAP.items():
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{api_sport}/odds/"
            r = requests.get(url, params={
                "apiKey": api_key, "regions": regions, "markets": markets,
                "oddsFormat": "decimal"
            }, timeout=15)
            if r.status_code != 200:
                continue
            for game in r.json():
                home = game.get("home_team", "")
                away = game.get("away_team", "")
                for bm in game.get("bookmakers", []):
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                            all_odds.append({
                                "home": home, "away": away, "league": league,
                                "bookmaker": bm.get("title", ""),
                                "home_odds": outcomes.get(home, 0),
                                "away_odds": outcomes.get(away, 0),
                                "draw_odds": outcomes.get("Draw", 0),
                                "date": game.get("commence_time", "")[:10],
                            })
        except Exception:
            continue
    return all_odds


# ═══════════════════════════════════════════════════════════════════════
# SENTIMENT-WEIGHTED PREDICTION BLENDER
# ═══════════════════════════════════════════════════════════════════════
def blend_prediction(model_prob_a, sentiment_prob_a, model_weight=0.65, sentiment_weight=0.35):
    """Blend model prediction with fan/market sentiment.

    Dynamic weights: when model is uncertain (45-55%), trust market more.
    When model is confident (>65%), trust model more.
    """
    # Dynamic weight adjustment based on model confidence
    model_confidence = abs(model_prob_a - 50)  # 0 = coin flip, 50 = certain
    if model_confidence < 7:
        # Model is very uncertain — trust market more
        mw, sw = 0.45, 0.55
    elif model_confidence < 15:
        # Model has slight lean — balanced blend
        mw, sw = 0.55, 0.45
    else:
        # Model is confident — trust model more
        mw, sw = 0.70, 0.30

    blended = model_prob_a * mw + sentiment_prob_a * sw
    blended = max(5.0, min(95.0, blended))

    agreement = abs(model_prob_a - sentiment_prob_a) < 10
    direction_agree = (model_prob_a > 50 and sentiment_prob_a > 50) or \
                      (model_prob_a < 50 and sentiment_prob_a < 50)

    if direction_agree and agreement:
        confidence = "HIGH"
        signal = "MODEL+MARKET AGREE"
    elif direction_agree:
        confidence = "MODERATE"
        signal = "SAME PICK, DIFFERENT MARGIN"
    else:
        confidence = "CONTRARIAN"
        signal = f"MODEL says {'HOME' if model_prob_a>50 else 'AWAY'}, MARKET says {'HOME' if sentiment_prob_a>50 else 'AWAY'}"

    return {
        "blended_prob_a": round(blended, 1),
        "blended_prob_b": round(100 - blended, 1),
        "model_prob_a": round(model_prob_a, 1),
        "sentiment_prob_a": round(sentiment_prob_a, 1),
        "confidence": confidence,
        "signal": signal,
        "model_weight": mw,
        "sentiment_weight": sw,
    }


def enrich_predictions_with_sentiment(predictions, sport_espn_path, odds_api_key=None):
    """Add sentiment data to existing predictions.

    For each prediction, finds matching ESPN odds and blends them.
    """
    from datetime import datetime, timedelta
    dates = [(datetime.now() + timedelta(days=d)).strftime("%Y%m%d") for d in range(8)]
    espn_odds = extract_espn_odds(sport_espn_path, dates)

    # Build lookup by team names
    odds_lookup = {}
    for o in espn_odds:
        key1 = f"{o['home']}_{o['away']}"
        key2 = f"{o['away']}_{o['home']}"
        odds_lookup[key1] = o
        odds_lookup[key2] = o

    # Also fetch from Odds API if key available
    bookmaker_odds = {}
    if odds_api_key:
        bm_odds = fetch_odds_api(odds_api_key)
        for o in bm_odds:
            key = f"{o['home']}_{o['away']}"
            if key not in bookmaker_odds:
                bookmaker_odds[key] = []
            bookmaker_odds[key].append(o)

    enriched = 0
    for pred in predictions:
        match = pred.get("match", "")
        parts = match.split(" vs ")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()

        # Try to find ESPN odds
        key = f"{a}_{b}"
        espn = odds_lookup.get(key) or odds_lookup.get(f"{b}_{a}")

        if espn:
            # Get model probability
            model_prob_a = pred.get("prediction", {}).get("prob_a",
                            pred.get("oracle_prediction", {}).get("home_pct", 50))

            # Get sentiment probability from ESPN spread
            if espn["home"] == a:
                sentiment_prob_a = espn["home_implied_pct"]
            else:
                sentiment_prob_a = espn["away_implied_pct"]

            blend = blend_prediction(model_prob_a, sentiment_prob_a)
            pred["sentiment"] = {
                "espn_spread": espn["spread"],
                "espn_over_under": espn["over_under"],
                "espn_provider": espn["provider"],
                "line_movement": espn["line_movement"],
                "implied_prob_a": sentiment_prob_a,
                **blend,
            }
            enriched += 1

        # Add bookmaker consensus if available
        bm = bookmaker_odds.get(key, [])
        if bm:
            avg_home = sum(o["home_odds"] for o in bm) / len(bm)
            avg_away = sum(o["away_odds"] for o in bm) / len(bm)
            pred["bookmaker_consensus"] = {
                "n_bookmakers": len(bm),
                "avg_home_odds": round(avg_home, 2),
                "avg_away_odds": round(avg_away, 2),
                "implied_home_pct": round(1 / avg_home * 100, 1) if avg_home > 0 else 50,
            }
            enriched += 1

    return enriched


if __name__ == "__main__":
    print("=== ESPN Embedded Odds (DraftKings) ===")
    for sport, path in [("NBA", "basketball/nba"), ("NHL", "hockey/nhl"),
                         ("EPL", "soccer/eng.1")]:
        odds = extract_espn_odds(path)
        print(f"\n{sport}: {len(odds)} matches with odds")
        for o in odds[:3]:
            print(f"  {o['home']:<25} vs {o['away']:<25} "
                  f"Spread: {o['spread']:>5} | O/U: {o['over_under']:>5} | "
                  f"Home: {o['home_implied_pct']}% | {o['line_movement']}")
