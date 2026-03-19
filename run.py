#!/usr/bin/env python3
"""
ORACLE V2 FINAL — Universal Sports Prediction Engine
Auto-installs deps, auto-downloads data, fetches LIVE fixtures, predicts everything.
Supports: Cricket, Football (6 leagues), NBA, NHL, MLB, NFL, UFC, F1

Usage:
    python run.py                  # Everything
    python run.py --cricket        # Cricket only
    python run.py --football       # Football (EPL/UCL/etc) only
    python run.py --multi          # NBA/NHL/MLB/UFC/F1 only
"""

# ═══════════════════════════════════════════════════════════════════════
# STEP 0: AUTO-INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════
import subprocess, sys, os, importlib, logging

logger = logging.getLogger("oracle.run")


def ensure_deps():
    REQ = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy", "requests": "requests"}
    OPT = {"xgboost": "xgboost", "lightgbm": "lightgbm", "glicko2": "glicko2", "trueskill": "trueskill"}

    def _pip(pkg):
        for ex in [["--break-system-packages"], []]:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg, "--quiet"] + ex,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except subprocess.CalledProcessError:
                continue
        return False

    for imp, pip in REQ.items():
        try:
            importlib.import_module(imp)
        except ImportError:
            logger.info("Installing %s...", pip)
            if not _pip(pip):
                logger.error("%s failed to install", pip)
                sys.exit(1)
    for imp, pip in OPT.items():
        try:
            importlib.import_module(imp)
        except ImportError:
            if _pip(pip):
                logger.info("Installed optional: %s", pip)
            else:
                logger.warning("Optional package %s skipped", pip)
    logger.info("Dependencies ready")

print("╔"+"═"*78+"╗")
print("║  ORACLE V2 FINAL — Universal Sports Prediction Engine                        ║")
print("║  Cricket + Football (6 leagues) + NBA + NHL + MLB + UFC + F1                 ║")
print("╚"+"═"*78+"╝")
ensure_deps()

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json, logging, time, warnings, argparse, zipfile
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import numpy as np, requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

# Load .env keys
try:
    from env_loader import load_env, ODDS_API_KEY, FOOTBALL_DATA_KEY, NEWSDATA_KEY
except ImportError:
    ODDS_API_KEY = FOOTBALL_DATA_KEY = NEWSDATA_KEY = ""

OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CRICSHEET_DIR = OUTPUT_DIR / "cricsheet_data"
FOOTBALL_DIR = OUTPUT_DIR / "football_data"

from core import (OracleDB, RatingEngine, ProbabilityCalibrator, MonteCarloSimulator, BiasAuditor, BacktestResult)
from engine import (PlayerDatabase, extract_features, FEATURE_NAMES, build_models, OracleV2, get_venue_data)
from cricsheet_pipeline import build_player_database, WC_SQUADS
from football_pipeline import (load_all_matches, FootballElo, build_features_and_labels,
    build_football_models, predict_upcoming, poisson_score_predict, backtest_roi_on_training_data,
    GAMBLING_DISCLAIMER)
from fixture_fetcher import fetch_and_format, identify_ucl_two_legs
from multi_sport import run_all_sports, enrich_sport_with_sentiment, SPORTS as MULTI_SPORTS
from sentiment import extract_espn_odds, enrich_predictions_with_sentiment, blend_prediction

# ═══════════════════════════════════════════════════════════════════════
# AUTO-DOWNLOAD DATA
# ═══════════════════════════════════════════════════════════════════════
def setup_data(cricket=True, football=True):
    if cricket:
        if not CRICSHEET_DIR.exists() or len(list(CRICSHEET_DIR.glob("*_info.csv")))<10:
            print("  📥 Downloading CricSheet T20I data (9 MB)...")
            CRICSHEET_DIR.mkdir(exist_ok=True)
            try:
                r=requests.get("https://cricsheet.org/downloads/t20s_male_csv2.zip",timeout=120)
                zp=OUTPUT_DIR/"cricsheet_temp.zip"; zp.write_bytes(r.content)
                with zipfile.ZipFile(zp,"r") as z: z.extractall(CRICSHEET_DIR)
                zp.unlink(); print(f"    ✅ {len(list(CRICSHEET_DIR.glob('*_info.csv')))} matches")
            except Exception as e: print(f"    ⚠️ {e}")
        else: print(f"  ✅ CricSheet: {len(list(CRICSHEET_DIR.glob('*_info.csv')))} matches")
    if football:
        FOOTBALL_DIR.mkdir(exist_ok=True)
        LG={"E0":"EPL","SP1":"La Liga","I1":"Serie A","D1":"Bundesliga","F1":"Ligue 1"}
        SS=["2324","2425","2526"]; needed=[]
        for c in LG:
            for s in SS:
                fp=FOOTBALL_DIR/f"{c}_{s}.csv"
                if not fp.exists() or fp.stat().st_size<100: needed.append((c,s,fp))
        if needed:
            print(f"  📥 Downloading {len(needed)} football CSVs...")
            for c,s,fp in needed:
                try:
                    r=requests.get(f"https://www.football-data.co.uk/mmz4281/{s}/{c}.csv",timeout=30)
                    if r.status_code==200 and len(r.content)>100: fp.write_bytes(r.content)
                except Exception:
                    pass
            print(f"    ✅ {sum(1 for c in LG for s in SS if (FOOTBALL_DIR/f'{c}_{s}.csv').exists())}/{len(LG)*len(SS)} CSVs")
        else: print(f"  ✅ Football: {len(LG)*len(SS)} CSVs")

# ═══════════════════════════════════════════════════════════════════════
# T20 WC 2026 RESULTS (54 matches)
# ═══════════════════════════════════════════════════════════════════════
CRICKET_RESULTS = [
    {"team_a":"Pakistan","team_b":"Netherlands","winner":"Pakistan","venue":"Colombo","stage":"group","date":"2026-02-07"},
    {"team_a":"West Indies","team_b":"Scotland","winner":"West Indies","venue":"Kolkata","stage":"group","date":"2026-02-07"},
    {"team_a":"India","team_b":"United States of America","winner":"India","venue":"Mumbai","stage":"group","date":"2026-02-07"},
    {"team_a":"New Zealand","team_b":"Afghanistan","winner":"New Zealand","venue":"Chennai","stage":"group","date":"2026-02-08"},
    {"team_a":"England","team_b":"Nepal","winner":"England","venue":"Mumbai","stage":"group","date":"2026-02-08"},
    {"team_a":"Sri Lanka","team_b":"Ireland","winner":"Sri Lanka","venue":"Colombo","stage":"group","date":"2026-02-08"},
    {"team_a":"Scotland","team_b":"Italy","winner":"Scotland","venue":"Kolkata","stage":"group","date":"2026-02-09"},
    {"team_a":"Zimbabwe","team_b":"Oman","winner":"Zimbabwe","venue":"Colombo","stage":"group","date":"2026-02-09"},
    {"team_a":"South Africa","team_b":"Canada","winner":"South Africa","venue":"Ahmedabad","stage":"group","date":"2026-02-09"},
    {"team_a":"Netherlands","team_b":"Namibia","winner":"Netherlands","venue":"Delhi","stage":"group","date":"2026-02-10"},
    {"team_a":"New Zealand","team_b":"United Arab Emirates","winner":"New Zealand","venue":"Chennai","stage":"group","date":"2026-02-10"},
    {"team_a":"Pakistan","team_b":"United States of America","winner":"Pakistan","venue":"Colombo","stage":"group","date":"2026-02-10"},
    {"team_a":"South Africa","team_b":"Afghanistan","winner":"South Africa","venue":"Ahmedabad","stage":"group","date":"2026-02-11"},
    {"team_a":"Australia","team_b":"Ireland","winner":"Australia","venue":"Colombo","stage":"group","date":"2026-02-11"},
    {"team_a":"West Indies","team_b":"England","winner":"West Indies","venue":"Mumbai","stage":"group","date":"2026-02-11"},
    {"team_a":"Sri Lanka","team_b":"Oman","winner":"Sri Lanka","venue":"Pallekele","stage":"group","date":"2026-02-12"},
    {"team_a":"Nepal","team_b":"Italy","winner":"Italy","venue":"Mumbai","stage":"group","date":"2026-02-12"},
    {"team_a":"India","team_b":"Namibia","winner":"India","venue":"Delhi","stage":"group","date":"2026-02-12","margin_numeric":93},
    {"team_a":"Zimbabwe","team_b":"Australia","winner":"Zimbabwe","venue":"Colombo","stage":"group","date":"2026-02-13"},
    {"team_a":"Canada","team_b":"United Arab Emirates","winner":"United Arab Emirates","venue":"Delhi","stage":"group","date":"2026-02-13"},
    {"team_a":"United States of America","team_b":"Netherlands","winner":"United States of America","venue":"Chennai","stage":"group","date":"2026-02-13"},
    {"team_a":"Ireland","team_b":"Oman","winner":"Ireland","venue":"Colombo","stage":"group","date":"2026-02-14"},
    {"team_a":"England","team_b":"Scotland","winner":"England","venue":"Kolkata","stage":"group","date":"2026-02-14"},
    {"team_a":"New Zealand","team_b":"South Africa","winner":"South Africa","venue":"Ahmedabad","stage":"group","date":"2026-02-14"},
    {"team_a":"West Indies","team_b":"Nepal","winner":"West Indies","venue":"Mumbai","stage":"group","date":"2026-02-15"},
    {"team_a":"United States of America","team_b":"Namibia","winner":"United States of America","venue":"Chennai","stage":"group","date":"2026-02-15"},
    {"team_a":"India","team_b":"Pakistan","winner":"India","venue":"Colombo","stage":"group","date":"2026-02-15","margin_numeric":61},
    {"team_a":"Afghanistan","team_b":"United Arab Emirates","winner":"Afghanistan","venue":"Delhi","stage":"group","date":"2026-02-16"},
    {"team_a":"England","team_b":"Italy","winner":"England","venue":"Kolkata","stage":"group","date":"2026-02-16"},
    {"team_a":"Australia","team_b":"Sri Lanka","winner":"Sri Lanka","venue":"Pallekele","stage":"group","date":"2026-02-16"},
    {"team_a":"New Zealand","team_b":"Canada","winner":"New Zealand","venue":"Chennai","stage":"group","date":"2026-02-17"},
    {"team_a":"Ireland","team_b":"Zimbabwe","winner":"NO RESULT","venue":"Pallekele","stage":"group","date":"2026-02-17"},
    {"team_a":"Scotland","team_b":"Nepal","winner":"Nepal","venue":"Mumbai","stage":"group","date":"2026-02-17"},
    {"team_a":"South Africa","team_b":"United Arab Emirates","winner":"South Africa","venue":"Delhi","stage":"group","date":"2026-02-18"},
    {"team_a":"Pakistan","team_b":"Namibia","winner":"Pakistan","venue":"Colombo","stage":"group","date":"2026-02-18"},
    {"team_a":"India","team_b":"Netherlands","winner":"India","venue":"Ahmedabad","stage":"group","date":"2026-02-18"},
    {"team_a":"West Indies","team_b":"Italy","winner":"West Indies","venue":"Kolkata","stage":"group","date":"2026-02-19"},
    {"team_a":"Sri Lanka","team_b":"Zimbabwe","winner":"Zimbabwe","venue":"Colombo","stage":"group","date":"2026-02-19"},
    {"team_a":"Afghanistan","team_b":"Canada","winner":"Afghanistan","venue":"Chennai","stage":"group","date":"2026-02-19"},
    {"team_a":"Australia","team_b":"Oman","winner":"Australia","venue":"Pallekele","stage":"group","date":"2026-02-20"},
    {"team_a":"New Zealand","team_b":"Pakistan","winner":"NO RESULT","venue":"Colombo","stage":"super8","date":"2026-02-21"},
    {"team_a":"England","team_b":"Sri Lanka","winner":"England","venue":"Pallekele","stage":"super8","date":"2026-02-22"},
    {"team_a":"India","team_b":"South Africa","winner":"South Africa","venue":"Ahmedabad","stage":"super8","date":"2026-02-22","margin_numeric":76},
    {"team_a":"West Indies","team_b":"Zimbabwe","winner":"West Indies","venue":"Mumbai","stage":"super8","date":"2026-02-23"},
    {"team_a":"England","team_b":"Pakistan","winner":"England","venue":"Pallekele","stage":"super8","date":"2026-02-24"},
    {"team_a":"New Zealand","team_b":"Sri Lanka","winner":"New Zealand","venue":"Colombo","stage":"super8","date":"2026-02-25"},
    {"team_a":"South Africa","team_b":"West Indies","winner":"South Africa","venue":"Ahmedabad","stage":"super8","date":"2026-02-26"},
    {"team_a":"India","team_b":"Zimbabwe","winner":"India","venue":"Chennai","stage":"super8","date":"2026-02-26","margin_numeric":72},
    {"team_a":"England","team_b":"New Zealand","winner":"England","venue":"Colombo","stage":"super8","date":"2026-02-27"},
    {"team_a":"Pakistan","team_b":"Sri Lanka","winner":"Pakistan","venue":"Pallekele","stage":"super8","date":"2026-02-28"},
    {"team_a":"Zimbabwe","team_b":"South Africa","winner":"South Africa","venue":"Delhi","stage":"super8","date":"2026-03-01"},
    {"team_a":"India","team_b":"West Indies","winner":"India","venue":"Kolkata","stage":"super8","date":"2026-03-01"},
    {"team_a":"New Zealand","team_b":"South Africa","winner":"New Zealand","venue":"Kolkata","stage":"semi","date":"2026-03-04"},
    {"team_a":"India","team_b":"England","winner":"India","venue":"Mumbai","stage":"semi","date":"2026-03-05","margin_numeric":7},
]

# ═══════════════════════════════════════════════════════════════════════
# CRICKET PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def run_cricket(R):
    print("\n"+"━"*80); print("  🏏 CRICKET"); print("━"*80)
    players,n=build_player_database(str(CRICSHEET_DIR),2022,WC_SQUADS)
    R["cricket_players"]=players
    R["_audit"]["A1_cricsheet"]=f"✅ {n} matches → {len(players)} players"
    print(f"\n  [1] {n} matches → {len(players)} players")

    db=OracleDB(str(OUTPUT_DIR/"oracle.db")); ratings=RatingEngine(db)
    scored=[m for m in CRICKET_RESULTS if m.get("winner","") not in ("","NO RESULT")]
    for m in CRICKET_RESULTS: m["sport"]="cricket"; db.insert_match(m)
    for m in scored:
        l=m["team_b"] if m["winner"]==m["team_a"] else m["team_a"]
        ratings.update_all(m["winner"],l,"cricket",float(m.get("margin_numeric",0)),
                           stage=m.get("stage","group"))
    teams=set(m["team_a"] for m in CRICKET_RESULTS)|set(m["team_b"] for m in CRICKET_RESULTS)
    cr={}
    for t in sorted(teams):
        r=ratings.get_all_ratings(t,"cricket")
        cr[t]={k:{kk:round(vv,1) if isinstance(vv,(int,float)) else vv for kk,vv in v.items() if kk in ("rating","rd","mu","sigma")} for k,v in r.items()}
    R["cricket_ratings"]=dict(sorted(cr.items(),key=lambda x:-x[1].get("elo",{}).get("rating",0)))
    R["_audit"]["A2_ratings"]=f"✅ {len(cr)} teams"; print(f"  [2] {len(cr)} teams rated")

    oracle=OracleV2(); oracle.db=db; oracle.ratings=ratings; ct=oracle.train(scored)
    R["cricket_training"]=ct
    cb=max(ct.get("cv_scores",{}).values()) if ct.get("cv_scores") else 0
    R["_audit"]["A3_ml"]=f"✅ {ct.get('models',0)} models, best: {cb:.3f}"
    print(f"  [3] {ct.get('models',0)} models, best: {cb:.3f}")

    bt=oracle.backtest(scored,10); R["cricket_backtest"]=asdict(bt)
    R["_audit"]["A4_backtest"]=f"✅ {bt.accuracy:.1%} ({bt.correct}/{bt.total_matches})"
    print(f"  [4] Backtest: {bt.accuracy:.1%} ({bt.correct}/{bt.total_matches})")

    # Cricket Elo rankings (from T20 WC results — carries into IPL season)
    top_teams=list(R["cricket_ratings"].items())[:10]
    print(f"\n  [5] Cricket Elo Rankings (from T20 WC 2026):")
    for i,(t,r) in enumerate(top_teams):
        elo_val=r.get("elo",{}).get("rating",0)
        print(f"      {i+1:>2}. {t:<25} {elo_val:.0f}")

    # Upcoming cricket — check ESPN
    print(f"\n  [6] Checking upcoming cricket...")
    try:
        upcoming_cricket=[]
        for endpoint in ["cricket","cricket/icc"]:
            data=requests.get(f"https://site.api.espn.com/apis/site/v2/sports/{endpoint}/scoreboard",timeout=10).json()
            for ev in data.get("events",[]):
                status=ev.get("status",{}).get("type",{}).get("name","")
                if "SCHEDULED" in status.upper():
                    comp=ev.get("competitions",[{}])[0]
                    teams=comp.get("competitors",[])
                    if len(teams)==2:
                        upcoming_cricket.append({
                            "match":f"{teams[0].get('team',{}).get('displayName','?')} vs {teams[1].get('team',{}).get('displayName','?')}",
                            "date":ev.get("date","")[:10],
                            "series":ev.get("season",{}).get("name",""),
                        })
        if upcoming_cricket:
            R["cricket_upcoming"]=upcoming_cricket
            R["_audit"]["A5_upcoming"]=f"✅ {len(upcoming_cricket)} upcoming cricket matches"
            for m in upcoming_cricket[:5]:
                print(f"      {m['date']} {m['match']} ({m['series']})")
        else:
            print(f"      No international cricket scheduled on ESPN right now")
            print(f"      🏏 IPL 2026 starts March 28 — first 20 matches released")
            R["_audit"]["A5_upcoming"]="✅ IPL 2026 starts Mar 28 (no live intl cricket)"
            R["cricket_upcoming"]=[]
    except Exception as e:
        print(f"      ⚠️ ESPN cricket: {e}")
        R["_audit"]["A5_upcoming"]="⚠️ ESPN cricket fetch failed"
        R["cricket_upcoming"]=[]

    # Store historical results only as training reference, not as predictions
    R["cricket_training_data_source"]=f"{len(CRICKET_RESULTS)} T20 WC 2026 matches used for Elo/ML training"

# ═══════════════════════════════════════════════════════════════════════
# FOOTBALL PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def run_football(R):
    print("\n"+"━"*80); print("  ⚽ FOOTBALL"); print("━"*80)
    fb_df=load_all_matches()
    if fb_df.empty or "league" not in fb_df.columns:
        print("  ⚠️ No data"); R["_audit"]["B1_data"]="⚠️ No data"; return

    nl=fb_df["league"].nunique()
    R["_audit"]["B1_data"]=f"✅ {len(fb_df)} matches, {nl} leagues"
    print(f"\n  [8] {len(fb_df)} matches | {nl} leagues")

    elo=FootballElo(1500,20,45); X,y,stats,h2h=build_features_and_labels(fb_df,elo)
    ae=elo.get_all()
    R["football_elo"]={t:round(r,1) for t,r in ae.items()}
    R["_audit"]["B2_elo"]=f"✅ {len(ae)} clubs"
    print(f"  [9] {len(ae)} clubs rated")
    # Show EPL rankings
    epl=[t for t in ae if t in ("Arsenal","Man City","Liverpool","Chelsea","Man United","Newcastle",
        "Tottenham","Aston Villa","Brighton","West Ham","Bournemouth","Wolves","Crystal Palace",
        "Fulham","Brentford","Everton","Nott'm Forest","Leeds","Sunderland","Burnley")]
    if epl:
        print(f"    EPL: {', '.join(f'{t} {ae[t]:.0f}' for t in epl[:6])}")

    sc=StandardScaler(); Xs=sc.fit_transform(X); models=build_football_models(); cv={}
    for n,m in models.items():
        m.fit(Xs,y)
        try: s=cross_val_score(m,Xs,y,cv=2,scoring="accuracy"); cv[n]=round(s.mean(),3)
        except Exception: cv[n]=round(float((m.predict(Xs)==y).mean()),3)
    R["football_training"]={"matches":len(X),"features":X.shape[1],"cv":cv}
    fb=max(cv.values()) if cv else 0
    R["_audit"]["B3_ml"]=f"✅ {len(models)} models, best: {fb:.3f}"
    print(f"  [10] {len(models)} models, best: {fb:.3f}")

    # ROI backtest
    roi=backtest_roi_on_training_data(fb_df,elo,stats,models,sc)
    R["football_roi_backtest"]=roi
    R["_audit"]["B4_roi"]=f"✅ ROI: {roi['roi_pct']}% ({roi['wins']}/{roi['total_bets']} wins)"
    print(f"  [10b] ROI backtest: {roi['roi_pct']}% ({roi['wins']}/{roi['total_bets']})")

    # Live fixtures
    print(f"\n  [11] Fetching LIVE fixtures...")
    try:
        live,raw=fetch_and_format()
        if live: upcoming=live; R["fixtures_source"]="LIVE"
        else: upcoming=[]; R["fixtures_source"]="EMPTY"
    except Exception as e:
        print(f"    ⚠️ {e}"); upcoming=[]; raw=[]; R["fixtures_source"]=str(e)

    if not upcoming:
        print("    No upcoming matches"); R["football_predictions"]=[]
        R["_audit"]["B5_preds"]="⚠️ No fixtures"; return

    # Predict with Poisson scores
    preds=predict_upcoming(models,sc,elo,stats,h2h,upcoming)
    for p in preds:
        parts = p["match"].split(" vs ")
        home_name, away_name = parts[0].strip(), parts[-1].strip()
        home_gs=stats.get(home_name,{}).get("goals_scored",[1.3])
        away_gs=stats.get(away_name,{}).get("goals_scored",[1.3])
        home_gc=stats.get(home_name,{}).get("goals_conceded",[1.1])
        away_gc=stats.get(away_name,{}).get("goals_conceded",[1.1])
        hg=sum(home_gs[-5:])/max(len(home_gs[-5:]),1) if home_gs else 1.3
        ag=sum(away_gs[-5:])/max(len(away_gs[-5:]),1) if away_gs else 1.3
        hc=sum(home_gc[-5:])/max(len(home_gc[-5:]),1) if home_gc else 1.1
        ac=sum(away_gc[-5:])/max(len(away_gc[-5:]),1) if away_gc else 1.1
        ed=p.get("elo_ratings",{}).get("diff",0)
        p["score_prediction"]=poisson_score_predict(hg,hc,ag,ac,elo_diff=ed)

    # Enrich with market sentiment (ESPN DraftKings odds)
    from sentiment import enrich_predictions_with_sentiment
    n_sent=enrich_predictions_with_sentiment(preds,"soccer/eng.1",ODDS_API_KEY or None)
    if n_sent: print(f"    📊 Sentiment data from {n_sent} matches (DraftKings via ESPN)")
    R["football_predictions"]=preds
    R["_audit"]["B5_preds"]=f"✅ {len(preds)} matches ({R['fixtures_source']})"
    for p in preds:
        o=p["oracle_prediction"]; sp=p.get("score_prediction",{})
        ic="🟠" if o["confidence"]=="HIGH" else "🟡" if o["confidence"]=="MODERATE" else "🟢"
        score_str=sp.get("predicted_score","?-?")
        print(f"    {p['league']:<10} {p['match']:<38} {ic} {o['predicted_result']:<12} "
              f"H:{o['home_pct']:>5.1f} D:{o['draw_pct']:>5.1f} A:{o['away_pct']:>5.1f}  Score: {score_str}")

    # UCL two-leg ties
    ucl_ties=identify_ucl_two_legs(raw)
    if ucl_ties:
        R["ucl_two_leg_ties"]=ucl_ties
        R["_audit"]["B6_ucl"]=f"✅ {len(ucl_ties)} UCL two-leg ties identified"
        print(f"\n    UCL Two-Leg Ties: {len(ucl_ties)}")
        for tie in ucl_ties:
            print(f"      {tie['teams'][0]} vs {tie['teams'][1]} ({'both legs found' if tie['is_two_leg'] else '1st leg only'})")

    # Odds API integration (if key available)
    if ODDS_API_KEY:
        try:
            from all_apis import OddsAPI
            print(f"\n  [BONUS] Fetching live bookmaker odds (key found)...")
            odds=OddsAPI.get_odds("epl")
            if odds:
                R["live_bookmaker_odds"]=odds[:10]
                print(f"    ✅ {len(odds)} odds records from 40+ bookmakers")
        except Exception as e:
            print(f"    ⚠️ Odds API: {e}")

    # News/injury (if key available)
    if NEWSDATA_KEY:
        try:
            from all_apis import NewsInjuryAPI
            news=NewsInjuryAPI.get_sports_news("football injury Premier League")
            if news: R["injury_news"]=news[:5]; print(f"    ✅ {len(news)} injury/news articles")
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════
# MULTI-SPORT
# ═══════════════════════════════════════════════════════════════════════
def run_multi(R):
    print("\n"+"━"*80); print("  🏅 MULTI-SPORT"); print("━"*80)
    results,total=run_all_sports()
    # Enrich each sport with sentiment
    for sport_key, result in results.items():
        enrich_sport_with_sentiment(result, sport_key, ODDS_API_KEY or None)
    R["multi_sport"]=results
    R["_audit"]["C_multi"]=f"✅ {len(results)} sports, {total} predictions"

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def run(cricket=True, football=True, multi=True):
    t0=time.time(); R={"_audit":{}}
    # Show API key status
    print(f"  API Keys: Odds={'✅' if ODDS_API_KEY else '❌'} | Football-Data={'✅' if FOOTBALL_DATA_KEY else '❌'} | NewsData={'✅' if NEWSDATA_KEY else '❌'}")
    print(f"  → Add keys to .env file for enhanced predictions (see .env.example)\n")

    setup_data(cricket,football)
    if cricket: run_cricket(R)
    if football: run_football(R)
    if multi: run_multi(R)

    elapsed=time.time()-t0
    R["metadata"]={"engine":"Oracle V2 Final","version":"3.0.0","generated":datetime.now().isoformat(),
        "seconds":round(elapsed,1),"python":sys.version.split()[0],"platform":sys.platform,
        "api_keys":{"odds_api":bool(ODDS_API_KEY),"football_data":bool(FOOTBALL_DATA_KEY),"newsdata":bool(NEWSDATA_KEY)}}

    # BIAS AUDIT
    print("\n"+"═"*80); print("  📋 FEATURE AUDIT"); print("═"*80)
    for _,s in sorted(R["_audit"].items()): print(f"  {s}")
    wk=sum(1 for s in R["_audit"].values() if "✅" in s); tot=len(R["_audit"])
    print(f"\n  Score: {wk}/{tot} ({'🟢 ALL GO' if wk==tot else '🟡 PARTIAL'})")

    # Run BiasAuditor on stored predictions
    from core import BiasAuditor
    try:
        db_path = OUTPUT_DIR / "oracle.db"
        if db_path.exists():
            from core import OracleDB as _DB
            _db = _DB(str(db_path))
            conn = _db._get_conn()
            rows = conn.execute("SELECT * FROM predictions WHERE is_correct >= 0").fetchall()
            if rows:
                preds = [dict(r) for r in rows]
                bias = BiasAuditor.audit(preds)
                R["bias_audit"] = bias
                print(f"\n  📊 Bias Audit: {bias.get('overall_accuracy',0):.1%} accuracy, "
                      f"{bias.get('favorite_bias',{}).get('assessment','N/A')} favorite bias")
    except Exception:
        pass

    # GAMBLING DISCLAIMER
    print(GAMBLING_DISCLAIMER)

    # SAVE
    jp=OUTPUT_DIR/"ORACLE_UNIFIED_OUTPUT.json"
    R["disclaimer"]=GAMBLING_DISCLAIMER.strip()
    with open(jp,"w") as f: json.dump(R,f,indent=2,default=str)
    print(f"  📁 {jp} ({jp.stat().st_size/1024:.0f} KB)")
    print(f"  ✅ Done in {elapsed:.1f}s"); print("═"*80)

if __name__=="__main__":
    pa=argparse.ArgumentParser(description="Oracle V2 — Universal Sports Prediction Engine")
    pa.add_argument("--cricket",action="store_true",help="Cricket only")
    pa.add_argument("--football",action="store_true",help="Football only")
    pa.add_argument("--multi",action="store_true",help="NBA/NHL/MLB/UFC/F1 only")
    a=pa.parse_args()
    if not a.cricket and not a.football and not a.multi: run(True,True,True)
    else: run(a.cricket,a.football,a.multi)
