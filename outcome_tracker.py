"""
Oracle V2 — Outcome Tracker & Self-Improvement Loop

Fetches results for previously predicted matches, scores predictions as
correct/incorrect, and generates learning insights to improve future accuracy.

Flow:
  1. fetch_all_results()  — Pull finished matches from ESPN (all sports)
  2. score_predictions()  — Match results against stored predictions
  3. learn_from_mistakes() — Analyze misses, generate recalibration weights
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger("oracle.outcome_tracker")

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
ESPN_HEADER = "https://site.web.api.espn.com/apis/v2/scoreboard/header"
ESPN = "https://site.api.espn.com/apis/site/v2/sports"


def _fetch_json(url, cache_key, ttl_hours=0.25):
    """Cached JSON fetch with short TTL for live data."""
    cf = CACHE_DIR / f"{cache_key}.json"
    if cf.exists() and (time.time() - cf.stat().st_mtime) / 3600 < ttl_hours:
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


# ═══════════════════════════════════════════════════════════════════════
# 1. UNIVERSAL LIVE/RESULT FETCHER — All sports via ESPN header API
# ═══════════════════════════════════════════════════════════════════════

# Sport keys mapped to ESPN header API sport parameter
HEADER_SPORTS = {
    "soccer": "football",
    "cricket": "cricket",
    "basketball": "basketball",
    "hockey": "hockey",
    "baseball": "baseball",
    "football": "american-football",
    "mma": "mma",
    "tennis": "tennis",
}


def fetch_all_live_and_results():
    """Fetch live, scheduled, and finished matches across ALL sports.

    Returns a list of dicts with normalized fields:
        sport, league, home/team_a, away/team_b, score, status, date, etc.
    """
    all_matches = []
    seen = set()

    for espn_sport, oracle_sport in HEADER_SPORTS.items():
        url = f"{ESPN_HEADER}?sport={espn_sport}"
        data = _fetch_json(url, f"header_{espn_sport}", ttl_hours=0.05)
        if not data:
            continue

        for sport_block in data.get("sports", []):
            for league in sport_block.get("leagues", []):
                league_name = league.get("name", "Unknown")
                league_id = league.get("id", "")
                for ev in league.get("events", []):
                    competitors = ev.get("competitors", [])
                    if len(competitors) != 2:
                        continue
                    status_info = ev.get("fullStatus", {}).get("type", {})
                    state = status_info.get("state", "").lower()
                    team_a = competitors[0].get("displayName", "?")
                    team_b = competitors[1].get("displayName", "?")
                    date = ev.get("date", "")[:10]
                    key = f"{oracle_sport}_{team_a}_{team_b}_{date}"
                    if key in seen:
                        continue
                    seen.add(key)

                    score_a = competitors[0].get("score", "")
                    score_b = competitors[1].get("score", "")
                    # Determine winner for finished matches
                    winner = ""
                    if state == "post" and score_a and score_b:
                        try:
                            sa = int(str(score_a).split("/")[0].split("(")[0].strip())
                            sb = int(str(score_b).split("/")[0].split("(")[0].strip())
                            if sa > sb:
                                winner = team_a
                            elif sb > sa:
                                winner = team_b
                        except (ValueError, IndexError):
                            pass

                    m = {
                        "sport": oracle_sport,
                        "league": league_name,
                        "league_id": league_id,
                        "team_a": team_a,
                        "team_b": team_b,
                        "date": date,
                        "time": ev.get("date", ""),
                        "score_a": str(score_a) if score_a else None,
                        "score_b": str(score_b) if score_b else None,
                        "winner": winner,
                        "status": "live" if state == "in" else "scheduled" if state == "pre" else "finished",
                        "status_detail": ev.get("summary", ""),
                    }
                    all_matches.append(m)

    all_matches.sort(key=lambda x: (x["status"] != "live", x["status"] != "scheduled", x.get("date", "")))
    return all_matches


# ═══════════════════════════════════════════════════════════════════════
# 2. PREDICTION SCORER — Match outcomes against stored predictions
# ═══════════════════════════════════════════════════════════════════════


def _normalize_name(name):
    """Normalize team names for fuzzy matching."""
    import re

    name = name.strip().lower()
    # Remove common suffixes/prefixes
    for suffix in [" fc", " cf", " sc", " afc", " united", " city"]:
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            pass  # Keep — these can be important for disambiguation
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def _names_match(a, b):
    """Fuzzy match two team names."""
    if not a or not b:
        return False
    na, nb = _normalize_name(a), _normalize_name(b)
    if na == nb:
        return True
    # Check if one contains the other (e.g. "Arsenal" matches "Arsenal FC")
    return bool(na in nb or nb in na)


def _date_diff_days(pred_date, result_date):
    """Days between prediction and result dates, or None if either is missing.

    Critical for sports like MLB where the same teams play a multi-game
    series on consecutive days — without date checks, a prediction for
    tomorrow's game gets scored against today's result.
    """
    pd, rd = str(pred_date or "")[:10], str(result_date or "")[:10]
    if not pd or not rd:
        return None
    try:
        d1 = datetime.strptime(pd, "%Y-%m-%d")
        d2 = datetime.strptime(rd, "%Y-%m-%d")
        return abs((d1 - d2).days)
    except ValueError:
        return None


def score_predictions(db, results):
    """Score unscored predictions against actual results.

    Args:
        db: OracleDB instance
        results: List of finished match dicts from fetch_all_live_and_results()

    Returns:
        dict with scoring summary
    """
    conn = db._get_conn()

    # Get all unscored predictions
    unscored = conn.execute("SELECT * FROM predictions WHERE is_correct = -1").fetchall()
    if not unscored:
        return {"scored": 0, "already_scored": 0, "message": "No unscored predictions"}

    unscored = [dict(r) for r in unscored]
    finished = [r for r in results if r["status"] == "finished" and r["winner"]]

    scored_count = 0
    correct_count = 0
    wrong_count = 0
    details = []

    for pred in unscored:
        pred_a = pred["team_a"]
        pred_b = pred["team_b"]
        pred_winner = pred["predicted_winner"]
        pred_date = pred.get("match_date", "")

        # Find the matching result — among name matches, take the one with
        # the closest date (max ±1 day) so consecutive-day series games
        # aren't scored against the wrong fixture.
        best_result = None
        best_diff = None
        for result in finished:
            if (_names_match(pred_a, result["team_a"]) and _names_match(pred_b, result["team_b"])) or (
                _names_match(pred_a, result["team_b"]) and _names_match(pred_b, result["team_a"])
            ):
                diff = _date_diff_days(pred_date, result.get("date", ""))
                if diff is not None and diff > 1:
                    continue
                rank = 2 if diff is None else diff  # Prefer exact > ±1 > undated
                if best_diff is None or rank < best_diff:
                    best_result = result
                    best_diff = rank
                    if rank == 0:
                        break

        if best_result is not None:
            result = best_result
            actual_winner = result["winner"]
            is_correct = 1 if _names_match(pred_winner, actual_winner) else 0
            # Handle draw predictions
            if pred_winner.upper() == "DRAW" and not result["winner"]:
                is_correct = 1

            conn.execute(
                "UPDATE predictions SET actual_winner=?, is_correct=? WHERE id=?",
                (actual_winner, is_correct, pred["id"]),
            )

            # Also update the match record if it exists
            match_id = pred.get("match_id", "")
            if match_id:
                conn.execute(
                    "UPDATE matches SET winner=?, score_a=?, score_b=? WHERE id=?",
                    (actual_winner, result.get("score_a", ""), result.get("score_b", ""), match_id),
                )

            scored_count += 1
            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            details.append(
                {
                    "match": f"{pred_a} vs {pred_b}",
                    "predicted": pred_winner,
                    "actual": actual_winner,
                    "correct": bool(is_correct),
                    "prob_a": pred.get("prob_a", 0),
                    "prob_b": pred.get("prob_b", 0),
                    "confidence": pred.get("confidence", ""),
                    "sport": pred.get("sport", ""),
                }
            )

    conn.commit()

    accuracy = correct_count / scored_count if scored_count > 0 else 0
    result = {
        "scored": scored_count,
        "correct": correct_count,
        "wrong": wrong_count,
        "accuracy": round(accuracy, 3),
        "unscored_remaining": len(unscored) - scored_count,
        "details": details,
    }
    if scored_count == 0:
        result["message"] = f"{len(unscored)} predictions awaiting results (matches not yet finished)"
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. STORE PREDICTIONS — Save new predictions to DB for future scoring
# ═══════════════════════════════════════════════════════════════════════


def store_prediction(db, pred, sport=""):
    """Store a prediction in the DB for future outcome tracking.

    Args:
        db: OracleDB instance
        pred: Dict with prediction fields (from any pipeline)
        sport: Sport identifier
    """
    # Normalize from different prediction formats
    team_a = pred.get("team_a", "")
    team_b = pred.get("team_b", "")

    # Football pipeline format
    if not team_a and "match" in pred:
        parts = pred["match"].split(" vs ")
        if len(parts) == 2:
            team_a, team_b = parts[0].strip(), parts[1].strip()

    if not team_a or not team_b:
        return None

    oracle = pred.get("oracle_prediction", pred.get("prediction", {}))
    prob_a = (
        oracle.get("home_pct", oracle.get("prob_a", 50)) / 100
        if oracle.get("home_pct", oracle.get("prob_a", 0)) > 1
        else oracle.get("prob_a", 0.5)
    )
    prob_b = (
        oracle.get("away_pct", oracle.get("prob_b", 50)) / 100
        if oracle.get("away_pct", oracle.get("prob_b", 0)) > 1
        else oracle.get("prob_b", 0.5)
    )
    prob_draw = (
        oracle.get("draw_pct", oracle.get("prob_draw", 0)) / 100
        if oracle.get("draw_pct", oracle.get("prob_draw", 0)) > 1
        else oracle.get("prob_draw", 0)
    )
    predicted_winner = oracle.get("predicted_result", oracle.get("winner", ""))
    confidence = oracle.get("confidence", "")
    model_votes = oracle.get("model_votes", {})

    match_date = str(pred.get("date", pred.get("match_date", "")))[:10]

    pred_record = {
        "sport": sport or pred.get("sport", pred.get("league", "")),
        "team_a": team_a,
        "team_b": team_b,
        "prob_a": prob_a,
        "prob_b": prob_b,
        "prob_draw": prob_draw,
        "predicted_winner": predicted_winner,
        "confidence": confidence,
        "model_votes": model_votes,
        "match_date": match_date,
    }

    return db.insert_prediction(pred_record)


# ═══════════════════════════════════════════════════════════════════════
# 4. SELF-IMPROVEMENT — Analyze misses and generate recalibration
# ═══════════════════════════════════════════════════════════════════════

LEARNING_FILE = Path(os.path.dirname(os.path.abspath(__file__))) / "oracle_learning.json"


def learn_from_mistakes(db):
    """Analyze scored predictions and generate learning insights.

    Writes recalibration weights to oracle_learning.json that the
    prediction pipeline can load on next run.

    Returns:
        dict with learning insights
    """
    conn = db._get_conn()

    # Get all scored predictions
    rows = conn.execute("SELECT * FROM predictions WHERE is_correct >= 0 ORDER BY created_at DESC").fetchall()
    if not rows:
        return {"message": "No scored predictions to learn from", "n": 0}

    preds = [dict(r) for r in rows]

    # Analyze by sport
    by_sport = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0, "high_conf_wrong": 0, "low_conf_right": 0})
    by_confidence = defaultdict(lambda: {"correct": 0, "total": 0})

    # Track overconfidence and underconfidence
    overconfident_misses = []  # High confidence but wrong
    upset_misses = []  # Strong favorite lost

    for p in preds:
        sport = p.get("sport", "unknown")
        s = by_sport[sport]
        s["total"] += 1
        conf = p.get("confidence", "MODERATE")
        bc = by_confidence[conf]
        bc["total"] += 1

        if p["is_correct"] == 1:
            s["correct"] += 1
            bc["correct"] += 1
            if conf == "LOW":
                s["low_conf_right"] += 1
        else:
            s["wrong"] += 1
            if conf == "HIGH":
                s["high_conf_wrong"] += 1
                overconfident_misses.append(
                    {
                        "match": f"{p['team_a']} vs {p['team_b']}",
                        "predicted": p["predicted_winner"],
                        "actual": p.get("actual_winner", "?"),
                        "prob_a": p.get("prob_a", 0),
                        "sport": sport,
                    }
                )

            # Check if we picked a strong favorite that lost
            max_prob = max(p.get("prob_a", 0), p.get("prob_b", 0))
            if max_prob > 0.65:
                upset_misses.append(
                    {
                        "match": f"{p['team_a']} vs {p['team_b']}",
                        "predicted": p["predicted_winner"],
                        "actual": p.get("actual_winner", "?"),
                        "max_prob": round(max_prob, 3),
                        "sport": sport,
                    }
                )

    # Generate recalibration weights
    sport_weights = {}
    for sport, s in by_sport.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.5
        # If accuracy is below 50%, the model needs significant recalibration
        # If accuracy is above 60%, it's well-calibrated
        # Scale confidence dampening based on accuracy
        if s["total"] >= 5:  # Need minimum sample
            conf_dampen = max(0.5, min(1.0, acc / 0.55))  # Dampen if below 55%
            draw_boost = 1.0
            if s["high_conf_wrong"] > s["total"] * 0.3:
                conf_dampen *= 0.85  # Extra dampening for overconfident models
                draw_boost = 1.15  # Boost draw probability
            sport_weights[sport] = {
                "accuracy": round(acc, 3),
                "n": s["total"],
                "confidence_dampen": round(conf_dampen, 3),
                "draw_boost": round(draw_boost, 3),
                "high_conf_miss_rate": round(s["high_conf_wrong"] / max(s["total"], 1), 3),
            }

    # Confidence-level calibration
    conf_calibration = {}
    for conf, bc in by_confidence.items():
        if bc["total"] >= 3:
            acc = bc["correct"] / bc["total"]
            conf_calibration[conf] = {
                "accuracy": round(acc, 3),
                "n": bc["total"],
                "well_calibrated": 0.45 <= acc <= 0.75,
            }

    # Save learning to file for next run
    learning = {
        "updated_at": datetime.now().isoformat(),
        "total_scored": len(preds),
        "overall_accuracy": round(sum(1 for p in preds if p["is_correct"] == 1) / len(preds), 3),
        "sport_weights": sport_weights,
        "confidence_calibration": conf_calibration,
        "recent_overconfident_misses": overconfident_misses[:10],
        "recent_upset_misses": upset_misses[:10],
    }

    with open(LEARNING_FILE, "w") as f:
        json.dump(learning, f, indent=2, default=str)

    return learning


def load_learning():
    """Load previously saved learning insights for prediction adjustment.

    Returns:
        dict with sport_weights and confidence_calibration, or empty defaults.
    """
    if LEARNING_FILE.exists():
        try:
            with open(LEARNING_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"sport_weights": {}, "confidence_calibration": {}}


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN ORCHESTRATOR — Run the full tracking loop
# ═══════════════════════════════════════════════════════════════════════


def run_outcome_tracking(db):
    """Run the full outcome tracking and learning loop.

    1. Fetch all live/finished results from ESPN
    2. Score any unscored predictions
    3. Analyze mistakes and generate recalibration
    4. Return summary

    Args:
        db: OracleDB instance

    Returns:
        dict with tracking results
    """
    print("\n  [OT] Fetching results across all sports...")
    all_matches = fetch_all_live_and_results()

    live = [m for m in all_matches if m["status"] == "live"]
    scheduled = [m for m in all_matches if m["status"] == "scheduled"]
    finished = [m for m in all_matches if m["status"] == "finished"]
    n_sports = len({m["sport"] for m in all_matches})
    n_leagues = len({m["league"] for m in all_matches})

    print(f"    ✅ {len(all_matches)} matches ({n_sports} sports, {n_leagues} competitions)")
    print(f"    🔴 {len(live)} live | 📅 {len(scheduled)} scheduled | ✅ {len(finished)} finished")

    # Display live matches by sport
    if live:
        by_sport = defaultdict(list)
        for m in live:
            by_sport[m["sport"]].append(m)
        print(f"\n    🔴 LIVE NOW ({len(live)}):")
        for _sport, ms in sorted(by_sport.items()):
            for m in ms[:5]:
                score = f"{m.get('score_a', '?')}-{m.get('score_b', '?')}"
                detail = f" ({m['status_detail']})" if m.get("status_detail") else ""
                print(f"      [{m['league']:30s}] {m['team_a']:22s} {score:>8}  {m['team_b']}{detail}")
            if len(ms) > 5:
                print(f"      ... +{len(ms) - 5} more")

    # Display scheduled
    if scheduled:
        by_sport = defaultdict(list)
        for m in scheduled:
            by_sport[m["sport"]].append(m)
        print(f"\n    📅 UPCOMING ({len(scheduled)}):")
        for _sport, ms in sorted(by_sport.items()):
            for m in ms[:3]:
                print(f"      [{m['league']:30s}] {m['team_a']:22s}  vs  {m['team_b']}")
            if len(ms) > 3:
                print(f"      ... +{len(ms) - 3} more")

    # Score predictions
    print("\n  [OT] Scoring predictions against outcomes...")
    score_result = score_predictions(db, all_matches)
    if score_result["scored"] > 0:
        print(
            f"    ✅ Scored {score_result['scored']} predictions: "
            f"{score_result['correct']} correct, {score_result['wrong']} wrong "
            f"({score_result['accuracy']:.1%})"
        )
        # Show details
        for d in score_result["details"][:10]:
            icon = "✅" if d["correct"] else "❌"
            print(f"      {icon} {d['match']}: predicted {d['predicted']}, actual {d['actual']}")
        if score_result["unscored_remaining"] > 0:
            print(f"    ⏳ {score_result['unscored_remaining']} predictions still awaiting results")
    else:
        msg = score_result.get("message", "No predictions to score yet (predictions stored for future tracking)")
        print(f"    {msg}")

    # Learn from scored predictions
    print("\n  [OT] Analyzing prediction performance...")
    learning = learn_from_mistakes(db)
    if learning.get("total_scored", 0) > 0:
        print(
            f"    📊 Overall accuracy: {learning['overall_accuracy']:.1%} "
            f"({learning['total_scored']} scored predictions)"
        )
        if learning.get("sport_weights"):
            print("    Per-sport performance:")
            for sport, w in sorted(learning["sport_weights"].items()):
                acc = w["accuracy"]
                icon = "🟢" if acc >= 0.55 else "🟡" if acc >= 0.45 else "🔴"
                dampen = w.get("confidence_dampen", 1.0)
                adj = f" (dampening: {dampen:.2f})" if dampen < 1.0 else ""
                print(f"      {icon} {sport:15s} {acc:.1%} ({w['n']} preds){adj}")
        if learning.get("recent_overconfident_misses"):
            n_oc = len(learning["recent_overconfident_misses"])
            print(f"    ⚠️ {n_oc} high-confidence misses detected — model will auto-dampen")
        print(f"    💾 Learning saved to {LEARNING_FILE.name}")
    else:
        print(f"    {learning.get('message', 'No data yet')}")

    return {
        "matches_fetched": len(all_matches),
        "live": len(live),
        "scheduled": len(scheduled),
        "finished": len(finished),
        "scoring": score_result,
        "learning": learning,
    }
