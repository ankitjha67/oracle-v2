"""
Oracle V2 — Core Infrastructure Layer
Database, Rate Limiting, Advanced Ratings, Calibration, Backtesting, Monte Carlo
"""
from __future__ import annotations
import sqlite3, json, time, math, hashlib, threading, logging, os
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Callable
from contextlib import contextmanager
import numpy as np

logger = logging.getLogger("oracle.core")

DB_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "oracle.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATABASE LAYER — SQLite with write-ahead logging
# ═══════════════════════════════════════════════════════════════════════════

class OracleDB:
    """Persistent storage for matches, predictions, ratings, odds, features."""

    _local = threading.local()

    def __init__(self, path: str = str(DB_PATH)):
        self.path = path
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.path, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def transaction(self):
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self):
        with self.transaction() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id TEXT PRIMARY KEY,
                sport TEXT NOT NULL,
                format TEXT DEFAULT '',
                tournament TEXT DEFAULT '',
                stage TEXT DEFAULT 'group',
                team_a TEXT NOT NULL,
                team_b TEXT NOT NULL,
                venue TEXT DEFAULT '',
                city TEXT DEFAULT '',
                country TEXT DEFAULT '',
                date TEXT DEFAULT '',
                winner TEXT DEFAULT '',
                score_a TEXT DEFAULT '',
                score_b TEXT DEFAULT '',
                margin TEXT DEFAULT '',
                toss_winner TEXT DEFAULT '',
                toss_decision TEXT DEFAULT '',
                is_neutral INTEGER DEFAULT 0,
                is_day_night INTEGER DEFAULT 0,
                weather_json TEXT DEFAULT '{}',
                extra_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_matches_sport ON matches(sport);
            CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(team_a, team_b);
            CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);

            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                match_id TEXT,
                sport TEXT,
                team_a TEXT,
                team_b TEXT,
                prob_a REAL,
                prob_b REAL,
                prob_draw REAL DEFAULT 0,
                predicted_winner TEXT,
                actual_winner TEXT DEFAULT '',
                confidence TEXT DEFAULT '',
                model_votes_json TEXT DEFAULT '{}',
                features_json TEXT DEFAULT '{}',
                odds_json TEXT DEFAULT '{}',
                is_correct INTEGER DEFAULT -1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(id)
            );
            CREATE INDEX IF NOT EXISTS idx_pred_sport ON predictions(sport);
            CREATE INDEX IF NOT EXISTS idx_pred_correct ON predictions(is_correct);

            CREATE TABLE IF NOT EXISTS team_ratings (
                team TEXT NOT NULL,
                sport TEXT NOT NULL,
                rating_type TEXT NOT NULL DEFAULT 'elo',
                rating REAL DEFAULT 1500,
                rd REAL DEFAULT 350,
                volatility REAL DEFAULT 0.06,
                mu REAL DEFAULT 25,
                sigma REAL DEFAULT 8.333,
                matches_played INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (team, sport, rating_type)
            );

            CREATE TABLE IF NOT EXISTS player_stats (
                player_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                team TEXT DEFAULT '',
                sport TEXT DEFAULT '',
                role TEXT DEFAULT '',
                batting_avg REAL DEFAULT 0,
                batting_sr REAL DEFAULT 0,
                bowling_avg REAL DEFAULT 0,
                bowling_econ REAL DEFAULT 0,
                bowling_sr REAL DEFAULT 0,
                fielding_catches INTEGER DEFAULT 0,
                matches INTEGER DEFAULT 0,
                runs INTEGER DEFAULT 0,
                wickets INTEGER DEFAULT 0,
                impact_rating REAL DEFAULT 50,
                form_json TEXT DEFAULT '[]',
                venue_stats_json TEXT DEFAULT '{}',
                vs_team_stats_json TEXT DEFAULT '{}',
                extra_json TEXT DEFAULT '{}',
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_player_team ON player_stats(team, sport);

            CREATE TABLE IF NOT EXISTS odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id TEXT,
                sport TEXT,
                team_a TEXT,
                team_b TEXT,
                bookmaker TEXT DEFAULT '',
                odds_a REAL,
                odds_b REAL,
                odds_draw REAL DEFAULT 0,
                implied_prob_a REAL,
                implied_prob_b REAL,
                implied_prob_draw REAL DEFAULT 0,
                line_type TEXT DEFAULT 'snapshot',
                hours_before_start REAL DEFAULT -1,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_odds_match ON odds_history(match_id);
            CREATE INDEX IF NOT EXISTS idx_odds_line_type ON odds_history(line_type);

            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT,
                ttl_hours REAL DEFAULT 6,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT,
                details_json TEXT DEFAULT '{}',
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """)

    # ── Match CRUD ──
    def insert_match(self, match: dict) -> str:
        mid = match.get("id") or hashlib.md5(
            f"{match['team_a']}_{match['team_b']}_{match.get('date','')}".encode()
        ).hexdigest()[:16]
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO matches
                (id,sport,format,tournament,stage,team_a,team_b,venue,city,country,
                 date,winner,score_a,score_b,margin,toss_winner,toss_decision,
                 is_neutral,is_day_night,weather_json,extra_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (mid, match.get("sport",""), match.get("format",""),
                  match.get("tournament",""), match.get("stage","group"),
                  match["team_a"], match["team_b"],
                  match.get("venue",""), match.get("city",""), match.get("country",""),
                  match.get("date",""), match.get("winner",""),
                  match.get("score_a",""), match.get("score_b",""),
                  match.get("margin",""), match.get("toss_winner",""),
                  match.get("toss_decision",""),
                  int(match.get("is_neutral", False)),
                  int(match.get("is_day_night", False)),
                  json.dumps(match.get("weather", {})),
                  json.dumps(match.get("extra", {}))))
        return mid

    def get_matches(self, sport: str = None, team: str = None,
                    limit: int = 1000, stage: str = None) -> list[dict]:
        conn = self._get_conn()
        query = "SELECT * FROM matches WHERE 1=1"
        params = []
        if sport:
            query += " AND sport=?"
            params.append(sport)
        if team:
            query += " AND (team_a=? OR team_b=?)"
            params.extend([team, team])
        if stage:
            query += " AND stage=?"
            params.append(stage)
        query += " ORDER BY date DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_h2h(self, team_a: str, team_b: str, sport: str = None) -> dict:
        conn = self._get_conn()
        q = """SELECT * FROM matches WHERE
               ((team_a=? AND team_b=?) OR (team_a=? AND team_b=?))
               AND winner != ''"""
        params = [team_a, team_b, team_b, team_a]
        if sport:
            q += " AND sport=?"
            params.append(sport)
        q += " ORDER BY date DESC"
        rows = conn.execute(q, params).fetchall()
        matches = [dict(r) for r in rows]
        a_wins = sum(1 for m in matches if m["winner"] == team_a)
        b_wins = sum(1 for m in matches if m["winner"] == team_b)
        draws = len(matches) - a_wins - b_wins
        recent = matches[:5]
        recent_a = sum(1 for m in recent if m["winner"] == team_a)
        recent_b = sum(1 for m in recent if m["winner"] == team_b)

        # Trend detection: compare recent form vs all-time
        h2h_trend = 0.0
        if len(matches) >= 5:
            all_time_rate = a_wins / len(matches) if matches else 0.5
            recent_rate = recent_a / len(recent) if recent else 0.5
            h2h_trend = recent_rate - all_time_rate  # positive = A improving

        # Streak detection: consecutive wins by same team
        streak_team = None
        streak_len = 0
        for m in matches:
            if streak_team is None:
                streak_team = m["winner"]
                streak_len = 1
            elif m["winner"] == streak_team:
                streak_len += 1
            else:
                break

        return {
            "total": len(matches), "a_wins": a_wins, "b_wins": b_wins,
            "draws": draws, "recent_5_a": recent_a, "recent_5_b": recent_b,
            "last_match": matches[0] if matches else None,
            "h2h_trend": round(h2h_trend, 3),
            "streak_team": streak_team, "streak_len": streak_len,
            "recent_win_rate_a": recent_a / len(recent) if recent else 0.5,
        }

    # ── Predictions ──
    def insert_prediction(self, pred: dict) -> str:
        pid = pred.get("id") or hashlib.md5(
            f"{pred['team_a']}_{pred['team_b']}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                (id,match_id,sport,team_a,team_b,prob_a,prob_b,prob_draw,
                 predicted_winner,actual_winner,confidence,model_votes_json,
                 features_json,odds_json,is_correct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (pid, pred.get("match_id",""), pred.get("sport",""),
                  pred["team_a"], pred["team_b"],
                  pred.get("prob_a",0.5), pred.get("prob_b",0.5),
                  pred.get("prob_draw",0),
                  pred.get("predicted_winner",""), pred.get("actual_winner",""),
                  pred.get("confidence",""),
                  json.dumps(pred.get("model_votes",{}), default=str),
                  json.dumps(pred.get("features",{}), default=str),
                  json.dumps(pred.get("odds",{}), default=str),
                  pred.get("is_correct", -1)))
        return pid

    def get_prediction_accuracy(self, sport: str = None, n: int = 100) -> dict:
        conn = self._get_conn()
        q = "SELECT * FROM predictions WHERE is_correct >= 0"
        params = []
        if sport:
            q += " AND sport=?"
            params.append(sport)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(n)
        rows = conn.execute(q, params).fetchall()
        preds = [dict(r) for r in rows]
        if not preds:
            return {"accuracy": 0, "total": 0, "correct": 0}
        correct = sum(1 for p in preds if p["is_correct"] == 1)
        return {
            "accuracy": correct / len(preds),
            "total": len(preds),
            "correct": correct,
            "brier_scores": [p["prob_a"] for p in preds],
        }

    # ── Ratings ──
    def get_rating(self, team: str, sport: str, rtype: str = "elo") -> dict:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM team_ratings WHERE team=? AND sport=? AND rating_type=?",
            (team, sport, rtype)).fetchone()
        if row:
            return dict(row)
        return {"team": team, "sport": sport, "rating_type": rtype,
                "rating": 1500, "rd": 350, "volatility": 0.06,
                "mu": 25, "sigma": 8.333, "matches_played": 0}

    def update_rating(self, team: str, sport: str, rtype: str, **kwargs):
        with self.transaction() as conn:
            existing = self.get_rating(team, sport, rtype)
            existing.update(kwargs)
            existing["last_updated"] = datetime.now().isoformat()
            conn.execute("""
                INSERT OR REPLACE INTO team_ratings
                (team,sport,rating_type,rating,rd,volatility,mu,sigma,
                 matches_played,last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (team, sport, rtype, existing.get("rating",1500),
                  existing.get("rd",350), existing.get("volatility",0.06),
                  existing.get("mu",25), existing.get("sigma",8.333),
                  existing.get("matches_played",0), existing["last_updated"]))

    # ── Player Stats ──
    def upsert_player(self, player: dict):
        pid = player.get("player_id") or hashlib.md5(
            player["name"].encode()).hexdigest()[:16]
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO player_stats
                (player_id,name,team,sport,role,batting_avg,batting_sr,
                 bowling_avg,bowling_econ,bowling_sr,fielding_catches,
                 matches,runs,wickets,impact_rating,form_json,
                 venue_stats_json,vs_team_stats_json,extra_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (pid, player["name"], player.get("team",""),
                  player.get("sport",""), player.get("role",""),
                  player.get("batting_avg",0), player.get("batting_sr",0),
                  player.get("bowling_avg",0), player.get("bowling_econ",0),
                  player.get("bowling_sr",0), player.get("fielding_catches",0),
                  player.get("matches",0), player.get("runs",0),
                  player.get("wickets",0), player.get("impact_rating",50),
                  json.dumps(player.get("form",[])),
                  json.dumps(player.get("venue_stats",{})),
                  json.dumps(player.get("vs_team_stats",{})),
                  json.dumps(player.get("extra",{}))))

    def get_team_players(self, team: str, sport: str = "") -> list[dict]:
        conn = self._get_conn()
        q = "SELECT * FROM player_stats WHERE team=?"
        params = [team]
        if sport:
            q += " AND sport=?"
            params.append(sport)
        return [dict(r) for r in conn.execute(q, params).fetchall()]

    # ── Odds History ──
    def insert_odds(self, odds: dict):
        with self.transaction() as conn:
            conn.execute("""
                INSERT INTO odds_history
                (match_id,sport,team_a,team_b,bookmaker,
                 odds_a,odds_b,odds_draw,
                 implied_prob_a,implied_prob_b,implied_prob_draw,
                 line_type,hours_before_start)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (odds.get("match_id",""), odds.get("sport",""),
                  odds["team_a"], odds["team_b"], odds.get("bookmaker",""),
                  odds.get("odds_a",0), odds.get("odds_b",0),
                  odds.get("odds_draw",0),
                  odds.get("implied_prob_a",0.5), odds.get("implied_prob_b",0.5),
                  odds.get("implied_prob_draw",0),
                  odds.get("line_type","snapshot"),
                  odds.get("hours_before_start",-1)))

    def get_odds_history(self, match_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM odds_history WHERE match_id=? ORDER BY timestamp",
            (match_id,)).fetchall()
        return [dict(r) for r in rows]

    # ── Cache ──
    def cache_get(self, key: str) -> Optional[Any]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data, ttl_hours, created_at FROM api_cache WHERE cache_key=?",
            (key,)).fetchone()
        if row:
            created = datetime.fromisoformat(row["created_at"])
            if (datetime.now() - created).total_seconds() < row["ttl_hours"] * 3600:
                return json.loads(row["data"])
        return None

    def cache_set(self, key: str, data: Any, ttl_hours: float = 6):
        with self.transaction() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO api_cache (cache_key, data, ttl_hours, created_at)
                VALUES (?, ?, ?, ?)
            """, (key, json.dumps(data, default=str), ttl_hours,
                  datetime.now().isoformat()))

    # ── Audit ──
    def log_event(self, event_type: str, details: dict = None):
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO audit_log (event_type, details_json) VALUES (?,?)",
                (event_type, json.dumps(details or {})))

    def get_stats(self) -> dict:
        conn = self._get_conn()
        return {
            "matches": conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
            "players": conn.execute("SELECT COUNT(*) FROM player_stats").fetchone()[0],
            "ratings": conn.execute("SELECT COUNT(*) FROM team_ratings").fetchone()[0],
            "odds_records": conn.execute("SELECT COUNT(*) FROM odds_history").fetchone()[0],
            "cache_entries": conn.execute("SELECT COUNT(*) FROM api_cache").fetchone()[0],
        }


# ═══════════════════════════════════════════════════════════════════════════
# 2. RATE LIMITER — Token bucket with per-API limits
# ═══════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """Thread-safe token bucket rate limiter per API."""

    _instances: dict[str, "RateLimiter"] = {}
    _lock = threading.Lock()

    LIMITS = {
        "open_meteo":      (10, 1.0),    # 10 requests per second
        "odds_api":        (1, 7.2),     # 500/month ≈ 1 per 7.2 seconds
        "thesportsdb":     (5, 1.0),     # 5 per second
        "espn":            (5, 1.0),
        "football_data":   (10, 60.0),   # 10 per minute
        "balldontlie":     (30, 60.0),   # 30 per minute
        "nba_api":         (5, 2.0),     # conservative
        "ergast":          (4, 1.0),
        "newsdata":        (1, 432.0),   # 200/day
        "nominatim":       (1, 1.0),     # 1 per second (OSM policy)
        "cricsheet":       (1, 60.0),    # be gentle
        "github":          (60, 60.0),   # 60/min unauthenticated
        "default":         (5, 1.0),
    }

    def __init__(self, api_name: str):
        limit = self.LIMITS.get(api_name, self.LIMITS["default"])
        self.max_tokens, self.period = limit
        self.tokens = float(self.max_tokens)
        self.last_refill = time.monotonic()
        self._lock_instance = threading.Lock()

    @classmethod
    def get(cls, api_name: str) -> "RateLimiter":
        with cls._lock:
            if api_name not in cls._instances:
                cls._instances[api_name] = RateLimiter(api_name)
            return cls._instances[api_name]

    def acquire(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock_instance:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + elapsed * (self.max_tokens / self.period)
                )
                self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    @classmethod
    def wait(cls, api_name: str) -> bool:
        return cls.get(api_name).acquire()


def rate_limited_request(api_name: str, func: Callable, *args, **kwargs) -> Any:
    """Execute a function with rate limiting and graceful fallback."""
    if not RateLimiter.wait(api_name):
        logger.warning(f"Rate limit timeout for {api_name}")
        return None
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"API call failed [{api_name}]: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. ADVANCED RATING SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════

class RatingEngine:
    """Elo, Glicko-2, TrueSkill, MOV-Elo, Surface-specific Elo."""

    def __init__(self, db: OracleDB):
        self.db = db

    # Stage-dependent K-factor: knockouts matter more than group stage
    STAGE_K_FACTORS = {
        "group": 20, "league": 20, "super_8": 24, "super8": 24,
        "quarter_final": 28, "quarter": 28,
        "semi_final": 32, "semi": 32,
        "final": 40, "dead_rubber": 12,
    }

    def elo_update(self, winner: str, loser: str, sport: str,
                   K: float = 32, margin: float = 0,
                   stage: str = "") -> tuple[float, float]:
        w = self.db.get_rating(winner, sport, "elo")
        l = self.db.get_rating(loser, sport, "elo")
        expected = 1.0 / (1.0 + 10 ** ((l["rating"] - w["rating"]) / 400))
        # Dynamic K-factor based on match importance
        if stage:
            K = self.STAGE_K_FACTORS.get(stage.lower().replace(" ", "_"), K)
        # MOV adjustment
        mov_mult = 1.0
        if margin > 0:
            mov_mult = 1 + math.log(max(margin, 1) + 1) * 0.4
        effective_K = K * mov_mult
        new_w = w["rating"] + effective_K * (1 - expected)
        new_l = l["rating"] + effective_K * (0 - (1 - expected))
        self.db.update_rating(winner, sport, "elo",
                              rating=new_w, matches_played=w["matches_played"]+1)
        self.db.update_rating(loser, sport, "elo",
                              rating=new_l, matches_played=l["matches_played"]+1)
        return new_w, new_l

    def glicko2_update(self, winner: str, loser: str, sport: str):
        w = self.db.get_rating(winner, sport, "glicko2")
        l = self.db.get_rating(loser, sport, "glicko2")
        try:
            import glicko2
            pw = glicko2.Player(rating=w["rating"], rd=w["rd"], vol=w["volatility"])
            pl = glicko2.Player(rating=l["rating"], rd=l["rd"], vol=l["volatility"])
            pw.update_player([l["rating"]], [l["rd"]], [1.0])
            pl.update_player([w["rating"]], [w["rd"]], [0.0])
            self.db.update_rating(winner, sport, "glicko2",
                                  rating=pw.getRating(), rd=pw.getRd(),
                                  volatility=pw.vol,
                                  matches_played=w["matches_played"]+1)
            self.db.update_rating(loser, sport, "glicko2",
                                  rating=pl.getRating(), rd=pl.getRd(),
                                  volatility=pl.vol,
                                  matches_played=l["matches_played"]+1)
        except ImportError:
            self.elo_update(winner, loser, sport)

    def trueskill_update(self, winner: str, loser: str, sport: str):
        w = self.db.get_rating(winner, sport, "trueskill")
        l = self.db.get_rating(loser, sport, "trueskill")
        try:
            import trueskill
            pw = trueskill.Rating(mu=w["mu"], sigma=w["sigma"])
            pl = trueskill.Rating(mu=l["mu"], sigma=l["sigma"])
            new_w, new_l = trueskill.rate_1vs1(pw, pl)
            self.db.update_rating(winner, sport, "trueskill",
                                  mu=new_w.mu, sigma=new_w.sigma,
                                  matches_played=w["matches_played"]+1)
            self.db.update_rating(loser, sport, "trueskill",
                                  mu=new_l.mu, sigma=new_l.sigma,
                                  matches_played=l["matches_played"]+1)
        except ImportError:
            self.elo_update(winner, loser, sport)

    def surface_elo_update(self, winner: str, loser: str, sport: str,
                           surface: str, K: float = 32):
        """Surface-specific Elo (critical for tennis: clay/grass/hard)."""
        rtype = f"elo_{surface}"
        w = self.db.get_rating(winner, sport, rtype)
        l = self.db.get_rating(loser, sport, rtype)
        expected = 1.0 / (1.0 + 10 ** ((l["rating"] - w["rating"]) / 400))
        new_w = w["rating"] + K * (1 - expected)
        new_l = l["rating"] + K * (0 - (1 - expected))
        self.db.update_rating(winner, sport, rtype,
                              rating=new_w, matches_played=w["matches_played"]+1)
        self.db.update_rating(loser, sport, rtype,
                              rating=new_l, matches_played=l["matches_played"]+1)
        return new_w, new_l

    def update_all(self, winner: str, loser: str, sport: str,
                   margin: float = 0, surface: str = "", stage: str = ""):
        """Update all rating systems at once."""
        stage_K = 32
        self.elo_update(winner, loser, sport, stage_K, margin, stage=stage)
        self.glicko2_update(winner, loser, sport)
        self.trueskill_update(winner, loser, sport)
        if surface:
            self.surface_elo_update(winner, loser, sport, surface)

    def get_all_ratings(self, team: str, sport: str) -> dict:
        return {
            "elo": self.db.get_rating(team, sport, "elo"),
            "glicko2": self.db.get_rating(team, sport, "glicko2"),
            "trueskill": self.db.get_rating(team, sport, "trueskill"),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. PROBABILITY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

class ProbabilityCalibrator:
    """Platt scaling, isotonic regression, and reliability diagrams."""

    def __init__(self):
        self._platt_a = 0.0
        self._platt_b = 0.0
        self._fitted = False
        self._iso_bins = None

    def fit_platt(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray):
        """Fit Platt scaling (sigmoid calibration)."""
        from scipy.optimize import minimize
        def neg_log_likelihood(params):
            a, b = params
            calibrated = 1.0 / (1.0 + np.exp(-(a * predicted_probs + b)))
            calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)
            return -np.mean(actual_outcomes * np.log(calibrated) +
                            (1 - actual_outcomes) * np.log(1 - calibrated))
        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self._platt_a, self._platt_b = result.x
        self._fitted = True

    def calibrate(self, prob: float) -> float:
        """Apply Platt scaling to a probability."""
        if not self._fitted:
            return prob
        return 1.0 / (1.0 + np.exp(-(self._platt_a * prob + self._platt_b)))

    def fit_isotonic(self, predicted_probs: np.ndarray,
                     actual_outcomes: np.ndarray, n_bins: int = 10):
        """Fit isotonic regression (binned calibration)."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        self._iso_bins = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                actual_rate = actual_outcomes[mask].mean()
                pred_center = predicted_probs[mask].mean()
                self._iso_bins.append((pred_center, actual_rate, mask.sum()))
            else:
                center = (bins[i] + bins[i+1]) / 2
                self._iso_bins.append((center, center, 0))

    def reliability_diagram(self) -> dict:
        """Return data for reliability diagram."""
        if self._iso_bins is None:
            return {}
        return {
            "bins": [{"predicted": b[0], "actual": b[1], "count": b[2]}
                     for b in self._iso_bins]
        }

    def brier_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Lower is better. Perfect = 0, worst = 1."""
        return np.mean((predicted - actual) ** 2)

    def log_loss_score(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
        return -np.mean(actual * np.log(predicted) +
                        (1 - actual) * np.log(1 - predicted))


# ═══════════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    total_matches: int = 0
    correct: int = 0
    wrong: int = 0
    accuracy: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    roi_pct: float = 0.0  # if betting at fair odds
    calibration_error: float = 0.0
    by_stage: dict = field(default_factory=dict)
    by_confidence: dict = field(default_factory=dict)
    predictions: list = field(default_factory=list)
    clv: float = 0.0  # closing line value


class WalkForwardBacktester:
    """
    Walk-forward backtesting: train on matches [0..N], predict match N+1.
    Simulates real production usage where you only know the past.
    """

    def __init__(self, model_factory: Callable, feature_extractor: Callable,
                 min_training_size: int = 10):
        self.model_factory = model_factory
        self.feature_extractor = feature_extractor
        self.min_training = min_training_size

    def run(self, matches: list[dict], odds_data: dict = None) -> BacktestResult:
        """
        matches: list of dicts with keys: team_a, team_b, winner, features, stage, odds_prob
        """
        result = BacktestResult()
        all_predicted = []
        all_actual = []
        all_probs = []

        for i in range(self.min_training, len(matches)):
            train = matches[:i]
            test = matches[i]

            # Build training data
            X_train = np.array([m["features"] for m in train])
            y_train = np.array([1 if m["winner"] == m["team_a"] else 0 for m in train])

            # Train model
            model = self.model_factory()
            try:
                model.fit(X_train, y_train)
            except Exception:
                continue

            # Predict
            X_test = np.array(test["features"]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[0]
                prob_a = proba[1] if len(proba) > 1 else proba[0]
            else:
                pred = model.predict(X_test)[0]
                prob_a = 0.9 if pred == 1 else 0.1

            predicted_winner = test["team_a"] if prob_a > 0.5 else test["team_b"]
            actual_winner = test["winner"]
            is_correct = predicted_winner == actual_winner

            result.total_matches += 1
            if is_correct:
                result.correct += 1
            else:
                result.wrong += 1

            actual_label = 1 if actual_winner == test["team_a"] else 0
            all_predicted.append(prob_a)
            all_actual.append(actual_label)

            # ROI calculation (if betting at model probability)
            if odds_data and test.get("match_id") in odds_data:
                market_prob = odds_data[test["match_id"]]
                if prob_a > market_prob + 0.05:  # value threshold
                    odds = 1 / market_prob
                    if is_correct:
                        result.roi_pct += (odds - 1)
                    else:
                        result.roi_pct -= 1

            result.predictions.append({
                "match": i, "team_a": test["team_a"], "team_b": test["team_b"],
                "prob_a": prob_a, "predicted": predicted_winner,
                "actual": actual_winner, "correct": is_correct,
                "stage": test.get("stage", ""),
            })

            # By stage
            stage = test.get("stage", "unknown")
            if stage not in result.by_stage:
                result.by_stage[stage] = {"correct": 0, "total": 0}
            result.by_stage[stage]["total"] += 1
            if is_correct:
                result.by_stage[stage]["correct"] += 1

        # Aggregate metrics
        if result.total_matches > 0:
            result.accuracy = result.correct / result.total_matches
            cal = ProbabilityCalibrator()
            predicted = np.array(all_predicted)
            actual = np.array(all_actual)
            result.brier_score = cal.brier_score(predicted, actual)
            result.log_loss = cal.log_loss_score(predicted, actual)

            # Calibration error (ECE) — 15 bins for finer resolution
            from analytics import EvaluationSuite
            cal_data = EvaluationSuite.calibration_data(predicted, actual, n_bins=15)
            result.calibration_error = cal_data["ece"]

        return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO TOURNAMENT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """Simulate entire tournaments N times to get outcome distributions."""

    def __init__(self, predict_fn: Callable, n_simulations: int = 10000,
                 seed: int = 42):
        """
        predict_fn: callable(team_a, team_b) -> (prob_a, prob_b)
        """
        self.predict_fn = predict_fn
        self.n_sims = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_match(self, team_a: str, team_b: str) -> str:
        """Simulate a single match outcome."""
        prob_a, _ = self.predict_fn(team_a, team_b)
        return team_a if self.rng.random() < prob_a else team_b

    def simulate_group(self, teams: list[str],
                       fixtures: list[tuple[str, str]]) -> list[tuple[str, int]]:
        """Simulate round-robin group, return sorted standings."""
        points = {t: 0 for t in teams}
        for ta, tb in fixtures:
            winner = self.simulate_match(ta, tb)
            points[winner] += 2
        return sorted(points.items(), key=lambda x: -x[1])

    def simulate_knockout(self, teams: list[str]) -> str:
        """Simulate single-elimination bracket."""
        remaining = list(teams)
        while len(remaining) > 1:
            next_round = []
            for i in range(0, len(remaining), 2):
                if i + 1 < len(remaining):
                    winner = self.simulate_match(remaining[i], remaining[i+1])
                    next_round.append(winner)
                else:
                    next_round.append(remaining[i])
            remaining = next_round
        return remaining[0] if remaining else ""

    def simulate_tournament(self, groups: dict[str, list[str]],
                            n_qualify: int = 2) -> dict:
        """
        Full tournament simulation.
        groups: {"A": ["team1", "team2", ...], "B": [...], ...}
        """
        results = defaultdict(lambda: {"champion": 0, "finalist": 0,
                                       "semifinalist": 0, "qualified": 0})

        for _ in range(self.n_sims):
            # Group stage
            qualifiers = []
            for group_name, teams in groups.items():
                fixtures = [(teams[i], teams[j])
                            for i in range(len(teams))
                            for j in range(i+1, len(teams))]
                standings = self.simulate_group(teams, fixtures)
                for team, pts in standings[:n_qualify]:
                    qualifiers.append(team)
                    results[team]["qualified"] += 1

            # Knockout
            if len(qualifiers) >= 4:
                # Semis
                sf1_winner = self.simulate_match(qualifiers[0], qualifiers[3])
                sf2_winner = self.simulate_match(qualifiers[1], qualifiers[2])
                results[sf1_winner]["semifinalist"] += 1
                results[sf2_winner]["semifinalist"] += 1

                # Final
                champion = self.simulate_match(sf1_winner, sf2_winner)
                runner_up = sf2_winner if champion == sf1_winner else sf1_winner
                results[champion]["champion"] += 1
                results[champion]["finalist"] += 1
                results[runner_up]["finalist"] += 1

        # Normalize
        output = {}
        for team, counts in results.items():
            output[team] = {
                "champion_pct": counts["champion"] / self.n_sims * 100,
                "finalist_pct": counts["finalist"] / self.n_sims * 100,
                "semifinalist_pct": counts["semifinalist"] / self.n_sims * 100,
                "qualified_pct": counts["qualified"] / self.n_sims * 100,
                "raw_counts": dict(counts),
            }

        return dict(sorted(output.items(),
                           key=lambda x: -x[1]["champion_pct"]))

    def simulate_series(self, team_a: str, team_b: str,
                        best_of: int = 7) -> dict:
        """Simulate a best-of-N series (NBA playoffs, Test series)."""
        wins_needed = (best_of + 1) // 2
        a_series_wins = 0

        for _ in range(self.n_sims):
            a_wins = b_wins = 0
            while a_wins < wins_needed and b_wins < wins_needed:
                winner = self.simulate_match(team_a, team_b)
                if winner == team_a:
                    a_wins += 1
                else:
                    b_wins += 1
            if a_wins == wins_needed:
                a_series_wins += 1

        return {
            team_a: round(a_series_wins / self.n_sims * 100, 1),
            team_b: round((self.n_sims - a_series_wins) / self.n_sims * 100, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. BIAS AUDITOR
# ═══════════════════════════════════════════════════════════════════════════

class BiasAuditor:
    """Detect systematic biases in the prediction model."""

    @staticmethod
    def audit(predictions: list[dict]) -> dict:
        """
        predictions: list of {team_a, team_b, prob_a, predicted_winner,
                              actual_winner, is_correct, stage, ...}
        """
        if not predictions:
            return {"error": "no predictions to audit"}

        scored = [p for p in predictions if p.get("actual_winner")]
        if not scored:
            return {"error": "no scored predictions"}

        total = len(scored)
        correct = sum(1 for p in scored if p.get("is_correct"))

        # Home team bias
        home_preds = [p for p in scored if p.get("is_home_a")]
        home_correct = sum(1 for p in home_preds if p.get("is_correct"))

        # Favorite bias (did we always pick the favorite?)
        fav_picked = sum(1 for p in scored if p.get("prob_a", 0.5) > 0.6)
        fav_correct = sum(1 for p in scored
                          if p.get("prob_a", 0.5) > 0.6 and p.get("is_correct"))

        # Underdog detection
        upsets = [p for p in scored
                  if not p.get("is_correct") and p.get("prob_a", 0.5) > 0.65]

        # Confidence calibration
        high_conf = [p for p in scored if p.get("prob_a", 0.5) > 0.7 or
                     p.get("prob_a", 0.5) < 0.3]
        high_conf_correct = sum(1 for p in high_conf if p.get("is_correct"))

        # Stage breakdown
        stage_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for p in scored:
            stage = p.get("stage", "unknown")
            stage_acc[stage]["total"] += 1
            if p.get("is_correct"):
                stage_acc[stage]["correct"] += 1

        return {
            "overall_accuracy": correct / total if total else 0,
            "total_predictions": total,
            "favorite_bias": {
                "favorites_picked": fav_picked,
                "favorites_correct": fav_correct,
                "favorites_accuracy": fav_correct / fav_picked if fav_picked else 0,
                "assessment": "HIGH BIAS" if fav_picked / total > 0.7 else "MODERATE"
                    if fav_picked / total > 0.5 else "LOW",
            },
            "upset_detection": {
                "missed_upsets": len(upsets),
                "upset_examples": [{"teams": f"{u['team_a']} vs {u['team_b']}",
                                   "predicted": u.get("predicted_winner"),
                                   "actual": u.get("actual_winner")}
                                  for u in upsets[:5]],
            },
            "confidence_calibration": {
                "high_confidence_predictions": len(high_conf),
                "high_confidence_accuracy": high_conf_correct / len(high_conf)
                    if high_conf else 0,
            },
            "stage_accuracy": {k: {"accuracy": v["correct"] / v["total"] if v["total"] else 0,
                                   **v}
                               for k, v in stage_acc.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_core():
    print("=" * 80)
    print("  ORACLE V2 CORE — Testing all components")
    print("=" * 80)

    # Database
    db = OracleDB()
    mid = db.insert_match({
        "sport": "cricket", "format": "t20", "team_a": "India",
        "team_b": "New Zealand", "venue": "Ahmedabad", "stage": "final",
        "date": "2026-03-08", "winner": "",
    })
    print(f"  ✅ DB: Match inserted (id={mid})")
    stats = db.get_stats()
    print(f"  ✅ DB stats: {stats}")

    # Rate limiter
    assert RateLimiter.wait("default"), "Rate limiter failed"
    print(f"  ✅ Rate limiter: working")

    # Rating engine
    rating_engine = RatingEngine(db)
    rating_engine.update_all("India", "England", "cricket", margin=7)
    r = rating_engine.get_all_ratings("India", "cricket")
    print(f"  ✅ Ratings: Elo={r['elo']['rating']:.0f}, "
          f"Glicko={r['glicko2']['rating']:.0f}")

    # Calibration
    cal = ProbabilityCalibrator()
    probs = np.array([0.7, 0.6, 0.8, 0.55, 0.9, 0.65, 0.75, 0.5, 0.85, 0.6])
    actuals = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0])
    cal.fit_platt(probs, actuals)
    calibrated = cal.calibrate(0.7)
    brier = cal.brier_score(probs, actuals)
    print(f"  ✅ Calibration: 0.70 → {calibrated:.3f}, Brier={brier:.4f}")

    # Monte Carlo
    def mock_predict(a, b):
        elo = {"India": 2150, "NZ": 2040, "SA": 2070, "ENG": 2080,
               "AUS": 2015, "WI": 1950, "PAK": 1920, "SL": 1854}
        ea = elo.get(a, 1500)
        eb = elo.get(b, 1500)
        pa = 1 / (1 + 10**((eb - ea) / 400))
        return pa, 1 - pa

    mc = MonteCarloSimulator(mock_predict, n_simulations=5000)
    series = mc.simulate_series("India", "NZ", best_of=5)
    print(f"  ✅ Monte Carlo (5-match series): {series}")

    # Bias auditor
    mock_preds = [
        {"team_a": "A", "team_b": "B", "prob_a": 0.7, "predicted_winner": "A",
         "actual_winner": "A", "is_correct": True, "stage": "group"},
        {"team_a": "A", "team_b": "C", "prob_a": 0.65, "predicted_winner": "A",
         "actual_winner": "C", "is_correct": False, "stage": "semi"},
    ]
    audit = BiasAuditor.audit(mock_preds)
    print(f"  ✅ Bias audit: overall accuracy={audit['overall_accuracy']:.1%}")

    print(f"\n  All core components operational!")
    print("=" * 80)


if __name__ == "__main__":
    test_core()
