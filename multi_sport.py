"""
Oracle V2 — Multi-Sport Engine (REWRITTEN)
NBA, NHL, MLB, NFL, UFC (fighters), F1 (drivers), Tennis (players),
WNBA, College Football, College Basketball, Rugby, AFL, MLS, and more.
All data from ESPN free API + OpenF1. Zero hardcoded matches.
"""

import json, os, time, math, requests, logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("oracle.multi_sport")

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
ESPN = "https://site.api.espn.com/apis/site/v2/sports"

# ═══════════════════════════════════════════════════════════════════════
# SPORT REGISTRY — All ESPN-backed sports with Elo parameters
# ═══════════════════════════════════════════════════════════════════════
SPORTS = {
    # --- Major US leagues ---
    "NBA":      {"espn": "basketball/nba",              "K": 25, "home": 55, "type": "team"},
    "NHL":      {"espn": "hockey/nhl",                  "K": 20, "home": 30, "type": "team"},
    "MLB":      {"espn": "baseball/mlb",                "K": 20, "home": 25, "type": "team"},
    "NFL":      {"espn": "football/nfl",                "K": 30, "home": 50, "type": "team"},
    "MLS":      {"espn": "soccer/usa.1",                "K": 22, "home": 45, "type": "team"},
    "WNBA":     {"espn": "basketball/wnba",             "K": 25, "home": 30, "type": "team"},

    # --- US College ---
    "NCAAF":    {"espn": "football/college-football",   "K": 28, "home": 55, "type": "team"},
    "NCAAM":    {"espn": "basketball/mens-college-basketball",  "K": 28, "home": 45, "type": "team"},
    "NCAAW":    {"espn": "basketball/womens-college-basketball","K": 28, "home": 40, "type": "team"},

    # --- Combat / Individual ---
    "UFC":      {"espn": "mma/ufc",                     "K": 40, "home": 0,  "type": "fighter"},
    "ATP":      {"espn": "tennis/atp",                  "K": 32, "home": 0,  "type": "player"},
    "WTA":      {"espn": "tennis/wta",                  "K": 32, "home": 0,  "type": "player"},

    # --- International ---
    "RUGBY":    {"espn": "rugby",                       "K": 30, "home": 40, "type": "team"},
    "RUGBY_L":  {"espn": "rugby-league",                "K": 30, "home": 38, "type": "team"},
    "AFL":      {"espn": "australian-football",          "K": 32, "home": 60, "type": "team"},
    "FIELD_HOCKEY": {"espn": "field-hockey",             "K": 28, "home": 35, "type": "team"},

    # --- Other ---
    "GOLF":     {"espn": "golf",                         "K": 30, "home": 0,  "type": "player"},
    "LACROSSE": {"espn": "lacrosse",                     "K": 25, "home": 50, "type": "team"},
}

# ═══════════════════════════════════════════════════════════════════════
# UNIVERSAL ELO
# ═══════════════════════════════════════════════════════════════════════
class SportElo:
    def __init__(self, default=1500, K=25, home_adv=50):
        self.ratings = defaultdict(lambda: default)
        self.matches = defaultdict(int)
        self.K = K; self.home_adv = home_adv

    def set_prior_from_record(self, name, record_str):
        """Set initial Elo from fighter/player record. E.g. '13-1-0' → higher prior."""
        try:
            parts = record_str.replace(" ","").split("-")
            wins = int(parts[0]); losses = int(parts[1])
            draws = int(parts[2]) if len(parts) > 2 else 0
            total = wins + losses + draws
            if total == 0: return
            win_rate = wins / total
            # Map win rate to Elo: 50% → 1500, 90% → 1650, 30% → 1350
            prior = 1500 + (win_rate - 0.5) * 300
            # Experience bonus: more fights = more reliable rating
            exp_bonus = min(total * 1.5, 50)  # Cap at +50
            self.ratings[name] = prior + exp_bonus
        except Exception:
            pass

    def update(self, winner, loser, home_team=None, margin=0):
        ha = self.home_adv if home_team==winner else (-self.home_adv if home_team==loser else 0)
        ea = 1/(1+10**((self.ratings[loser]-(self.ratings[winner]+ha))/400))
        mov = 1+math.log(max(margin,1))*0.3 if margin>0 else 1.0
        delta = self.K*mov*(1-ea)
        self.ratings[winner]+=delta; self.ratings[loser]-=delta
        self.matches[winner]+=1; self.matches[loser]+=1

    def predict(self, a, b, home=None):
        ha = self.home_adv if home==a else (-self.home_adv if home==b else 0)
        pa = 1/(1+10**((self.ratings[b]-(self.ratings[a]+ha))/400))
        return round(pa*100,1), round((1-pa)*100,1)

    def rankings(self, n=30):
        return dict(sorted(self.ratings.items(), key=lambda x:-x[1])[:n])

# ═══════════════════════════════════════════════════════════════════════
# FETCH HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _fetch(url, key, ttl=1):
    cf=CACHE_DIR/f"{key}.json"
    if cf.exists() and (time.time()-cf.stat().st_mtime)/3600<ttl:
        with open(cf) as f: return json.load(f)
    try:
        r=requests.get(url,timeout=15,headers={"User-Agent":"OracleV2/2.3"})
        if r.status_code==200:
            d=r.json()
            with open(cf,"w") as f: json.dump(d,f)
            return d
    except Exception:
        pass
    return None

def _parse_team_event(event):
    """Parse a standard ESPN team sport event."""
    comp=event.get("competitions",[{}])[0]
    status=comp.get("status",{}).get("type",{}).get("name","")
    teams=comp.get("competitors",[])
    if len(teams)!=2: return None
    t0,t1=teams[0],teams[1]
    n0=t0.get("team",{}).get("displayName","?")
    n1=t1.get("team",{}).get("displayName","?")
    s0=int(t0.get("score",0) or 0); s1=int(t1.get("score",0) or 0)
    home=n0 if t0.get("homeAway")=="home" else (n1 if t1.get("homeAway")=="home" else None)
    return {"name0":n0,"name1":n1,"score0":s0,"score1":s1,"home":home,"status":status,
            "date":event.get("date","")[:10],"venue":comp.get("venue",{}).get("fullName","")}

def _parse_fight_event(event):
    """Parse ESPN UFC event with fighter names and records."""
    fights=[]
    for comp in event.get("competitions",[]):
        fighters=comp.get("competitors",[])
        if len(fighters)!=2: continue
        f0,f1=fighters[0],fighters[1]
        a0=f0.get("athlete",{}); a1=f1.get("athlete",{})
        n0=a0.get("displayName","?"); n1=a1.get("displayName","?")
        if n0=="?" and n1=="?": continue
        r0=f0.get("records",[{}])[0].get("summary","0-0-0") if f0.get("records") else "0-0-0"
        r1=f1.get("records",[{}])[0].get("summary","0-0-0") if f1.get("records") else "0-0-0"
        w0=f0.get("winner",False); w1=f1.get("winner",False)
        status=comp.get("status",{}).get("type",{}).get("name","")
        fights.append({"fighter_a":n0,"fighter_b":n1,"record_a":r0,"record_b":r1,
                       "winner":n0 if w0 else (n1 if w1 else None),
                       "status":status,"date":event.get("date","")[:10],
                       "event_name":event.get("name","")})
    return fights

# ═══════════════════════════════════════════════════════════════════════
# SPORT-SPECIFIC PIPELINES
# ═══════════════════════════════════════════════════════════════════════
def build_team_sport(sport_key, days_back=30, days_ahead=7):
    """NBA, NHL, MLB, NFL — standard team sport pipeline."""
    cfg=SPORTS[sport_key]; elo=SportElo(1500,cfg["K"],cfg["home"])
    # Build Elo from recent results
    results=[]
    for d in range(0,days_back,3):
        dt=(datetime.now()-timedelta(days=d)).strftime("%Y%m%d")
        data=_fetch(f"{ESPN}/{cfg['espn']}/scoreboard?dates={dt}",f"h_{sport_key}_{dt}",24)
        if not data: continue
        for ev in data.get("events",[]):
            p=_parse_team_event(ev)
            if not p or "FINAL" not in p["status"].upper(): continue
            if p["score0"]==p["score1"]: continue
            w,l=(p["name0"],p["name1"]) if p["score0"]>p["score1"] else (p["name1"],p["name0"])
            elo.update(w,l,p["home"],abs(p["score0"]-p["score1"]))
            results.append({"winner":w,"loser":l,"score":f"{p['score0']}-{p['score1']}","date":p["date"]})
    # Fetch upcoming
    preds=[]
    for d in range(0,days_ahead+1):
        dt=(datetime.now()+timedelta(days=d)).strftime("%Y%m%d")
        data=_fetch(f"{ESPN}/{cfg['espn']}/scoreboard?dates={dt}",f"u_{sport_key}_{dt}",1)
        if not data: continue
        for ev in data.get("events",[]):
            p=_parse_team_event(ev)
            if not p or "SCHEDULED" not in p["status"].upper(): continue
            pa,pb=elo.predict(p["name0"],p["name1"],p["home"])
            winner=p["name0"] if pa>50 else p["name1"]
            diff=abs(pa-50)
            conf="HIGH" if diff>15 else "MODERATE" if diff>7 else "LOW"
            preds.append({"match":f"{p['name0']} vs {p['name1']}","date":p["date"],
                "venue":p["venue"],"sport":sport_key,
                "prediction":{"winner":winner,"prob_a":pa,"prob_b":pb,"confidence":conf},
                "elo":{"a":round(elo.ratings[p["name0"]],1),"b":round(elo.ratings[p["name1"]],1)},
                "home":p["home"]})
    return {"sport":sport_key,"results_used":len(results),"teams":len(elo.ratings),
            "rankings":elo.rankings(20),"predictions":preds}

def build_ufc(days_back=60, days_ahead=14):
    """UFC — fighter-level predictions from ESPN."""
    elo=SportElo(1500,40,0)
    # Build Elo from recent fight results
    results=[]
    for d in range(0,days_back,7):
        dt=(datetime.now()-timedelta(days=d)).strftime("%Y%m%d")
        data=_fetch(f"{ESPN}/mma/ufc/scoreboard?dates={dt}",f"h_ufc_{dt}",24)
        if not data: continue
        for ev in data.get("events",[]):
            for fight in _parse_fight_event(ev):
                if fight["winner"]:
                    loser=fight["fighter_b"] if fight["winner"]==fight["fighter_a"] else fight["fighter_a"]
                    elo.update(fight["winner"],loser)
                    results.append(fight)
    # Set priors from records for upcoming fighters
    for d in range(0,days_ahead+1):
        dt_str=(datetime.now()+timedelta(days=d)).strftime("%Y%m%d")
        data_up=_fetch(f"{ESPN}/mma/ufc/scoreboard?dates={dt_str}",f"rec_ufc_{dt_str}",1)
        if not data_up: continue
        for ev in data_up.get("events",[]):
            for fight in _parse_fight_event(ev):
                if fight["fighter_a"]!="?" and fight["record_a"]:
                    elo.set_prior_from_record(fight["fighter_a"], fight["record_a"])
                if fight["fighter_b"]!="?" and fight["record_b"]:
                    elo.set_prior_from_record(fight["fighter_b"], fight["record_b"])

    # Upcoming
    preds=[]
    for d in range(0,days_ahead+1):
        dt=(datetime.now()+timedelta(days=d)).strftime("%Y%m%d")
        data=_fetch(f"{ESPN}/mma/ufc/scoreboard?dates={dt}",f"u_ufc_{dt}",1)
        if not data: continue
        for ev in data.get("events",[]):
            for fight in _parse_fight_event(ev):
                if fight["status"] and "SCHEDULED" not in fight["status"].upper(): continue
                if fight["fighter_a"]=="?" or fight["fighter_b"]=="?": continue
                pa,pb=elo.predict(fight["fighter_a"],fight["fighter_b"])
                winner=fight["fighter_a"] if pa>50 else fight["fighter_b"]
                diff=abs(pa-50)
                conf="HIGH" if diff>15 else "MODERATE" if diff>7 else "LOW"
                preds.append({"match":f"{fight['fighter_a']} vs {fight['fighter_b']}",
                    "date":fight["date"],"event":fight["event_name"],"sport":"UFC",
                    "prediction":{"winner":winner,"prob_a":pa,"prob_b":pb,"confidence":conf},
                    "records":{"a":fight["record_a"],"b":fight["record_b"]},
                    "elo":{"a":round(elo.ratings[fight["fighter_a"]],1),"b":round(elo.ratings[fight["fighter_b"]],1)}})
    return {"sport":"UFC","results_used":len(results),"fighters":len(elo.ratings),
            "rankings":elo.rankings(20),"predictions":preds}

def build_f1():
    """F1 — driver championship predictions from OpenF1."""
    try:
        # Get current drivers
        r=requests.get("https://api.openf1.org/v1/drivers?session_key=latest",timeout=10)
        drivers=r.json() if r.status_code==200 else []
        # Get recent race results
        r2=requests.get("https://api.openf1.org/v1/position?session_key=latest&position<=10",timeout=10)
        positions=r2.json() if r2.status_code==200 else []
    except Exception:
        drivers = []
        positions = []
    
    driver_list=[{"number":d.get("driver_number"),"name":d.get("full_name","?"),
                  "team":d.get("team_name","?")} for d in drivers if d.get("full_name")]
    return {"sport":"F1","drivers":len(driver_list),"grid":driver_list[:20],
            "note":"F1 predictions require race-by-race modeling (qualifying→race). Grid data from OpenF1.",
            "predictions":[]}

# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ═══════════════════════════════════════════════════════════════════════
def get_active_sports(month=None):
    """Return sport keys that are currently in-season."""
    if month is None:
        month = datetime.now().month
    active = []
    # Year-round
    active += ["UFC", "F1"]
    # Season-based
    if month in (10, 11, 12, 1, 2, 3, 4, 5, 6):
        active += ["NBA", "NHL", "WNBA"]
    if month in (3, 4, 5, 6, 7, 8, 9, 10):
        active.append("MLB")
    if month in (9, 10, 11, 12, 1, 2):
        active += ["NFL", "NCAAF"]
    if month in (11, 12, 1, 2, 3):
        active += ["NCAAM", "NCAAW"]
    # Year-round international
    active += ["RUGBY", "AFL"]
    # Soccer year-round
    active.append("MLS")
    return active


def run_all_sports(sport_keys=None, days_back=30, days_ahead=7):
    if sport_keys is None:
        sport_keys = get_active_sports()

    # Sport-specific lookback (more history = better Elo convergence)
    LOOKBACK = {
        "NBA": 60, "NHL": 60, "MLB": 45, "NFL": 90, "UFC": 90,
        "WNBA": 45, "NCAAF": 90, "NCAAM": 60, "NCAAW": 60,
        "RUGBY": 60, "AFL": 60, "MLS": 45,
        "RUGBY_L": 60, "FIELD_HOCKEY": 60, "LACROSSE": 45,
    }

    ICONS = {
        "NBA": "🏀", "NHL": "🏒", "MLB": "⚾", "NFL": "🏈",
        "UFC": "🥊", "F1": "🏎️", "ATP": "🎾", "WTA": "🎾",
        "WNBA": "🏀", "NCAAF": "🏈", "NCAAM": "🏀", "NCAAW": "🏀",
        "RUGBY": "🏉", "RUGBY_L": "🏉", "AFL": "🏉",
        "MLS": "⚽", "GOLF": "⛳", "LACROSSE": "🥍",
        "FIELD_HOCKEY": "🏑",
    }

    all_results={}; total=0
    for sport in sport_keys:
        if sport not in SPORTS:
            logger.warning(f"Unknown sport: {sport}")
            continue
        print(f"\n  {ICONS.get(sport,'🏅')} {sport}")

        cfg = SPORTS[sport]
        lb = LOOKBACK.get(sport, days_back)

        if sport == "UFC":
            result = build_ufc(days_back=lb, days_ahead=14)
        elif sport == "F1":
            result = build_f1()
        elif cfg["type"] == "team":
            result = build_team_sport(sport, lb, days_ahead)
        else:
            # Individual sports (ATP, WTA, GOLF) use team sport pipeline
            # with home_adv=0 (no home court for individuals)
            result = build_team_sport(sport, lb, days_ahead)

        all_results[sport]=result
        n=len(result.get("predictions",[])); total+=n
        
        print(f"    Built from: {result.get('results_used',result.get('drivers',0))} results")
        n_entities=result.get("teams",result.get("fighters",result.get("drivers",0)))
        print(f"    Rated: {n_entities} | Upcoming: {n}")

        # Show rankings
        for i,(t,r) in enumerate(list(result.get("rankings",{}).items())[:5]):
            print(f"      {i+1}. {t:<28} {r:.0f}")

        # Show predictions
        for p in result.get("predictions",[])[:8]:
            pr=p["prediction"]; ic="🟠" if pr["confidence"]=="HIGH" else "🟡" if pr["confidence"]=="MODERATE" else "🟢"
            rec = f" [{p['records']['a']} vs {p['records']['b']}]" if "records" in p else ""
            print(f"      {p['date']} {p['match']:<45} {ic} {pr['winner']:<20} ({pr['prob_a']:.0f}-{pr['prob_b']:.0f}){rec}")
        if n>8: print(f"      ... +{n-8} more")

    return all_results, total

if __name__=="__main__":
    print("="*80); print("  ORACLE V2 — MULTI-SPORT"); print("="*80)
    r,t=run_all_sports()
    print(f"\n  Total: {t} predictions")
    with open(Path(os.path.dirname(os.path.abspath(__file__)))/"MULTI_SPORT.json","w") as f:
        json.dump(r,f,indent=2,default=str)


def enrich_sport_with_sentiment(result, sport_key, odds_api_key=None):
    """Enrich a sport's predictions with ESPN odds sentiment."""
    from sentiment import enrich_predictions_with_sentiment
    cfg = SPORTS.get(sport_key, {})
    espn_path = cfg.get("espn", "")
    if espn_path and result.get("predictions"):
        n = enrich_predictions_with_sentiment(result["predictions"], espn_path, odds_api_key)
        result["sentiment_enriched"] = n
    return result
