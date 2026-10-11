"""Database layer: Postgres when DATABASE_URL is set, otherwise a local SQLite file.
Also holds the rate limiter (stored in the DB so limits survive restarts and are shared by all instances)."""
import os, random, sqlite3, time
from contextlib import contextmanager
from datetime import datetime, timezone

from pathlib import Path

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", os.getenv("NEXT_PUBLIC_SUPABASE_URL", "https://kdlpsqejqymofmiuelur.supabase.co"))
SUPABASE_KEY = os.getenv("SUPABASE_KEY", os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "sb_publishable_1bjb0Nv72r5ECuQdCmHIZg_4yAOqNYt"))
NEXT_PUBLIC_SUPABASE_URL = SUPABASE_URL
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY = SUPABASE_KEY
_supabase_client = None


def get_supabase():
    """Returns an initialized Supabase Python client instance."""
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception:
            _supabase_client = None
    return _supabase_client

# Helper: build Postgres connection string if password provided and DATABASE_URL not set
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD", "")
SUPABASE_PROJECT_REF = ""
if "supabase.co" in NEXT_PUBLIC_SUPABASE_URL:
    try:
        SUPABASE_PROJECT_REF = NEXT_PUBLIC_SUPABASE_URL.split("://")[-1].split(".")[0]
    except Exception:
        pass

DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL and SUPABASE_DB_PASSWORD and SUPABASE_PROJECT_REF:
    DATABASE_URL = f"postgresql://postgres:{SUPABASE_DB_PASSWORD}@db.{SUPABASE_PROJECT_REF}.supabase.co:5432/postgres"

USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))
SQLITE_PATH = os.getenv("DB_PATH", "bank.db")
_pool = None


def now_s(offset=0.0) -> str:
    return datetime.fromtimestamp(time.time() + offset, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class Conn:
    """Same call style for both databases: execute(sql_with_?, params)."""
    def __init__(self, raw, pg):
        self.raw, self.pg = raw, pg

    def execute(self, sql, params=()):
        sql = sql.replace("?", "%s") if self.pg else sql.replace(" FOR UPDATE", "")
        return self.raw.execute(sql, params)


def _get_pool():
    global _pool
    if _pool is None:
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=int(os.getenv("DB_POOL_MAX", "10")),
                               kwargs={"row_factory": dict_row, "prepare_threshold": None},  # works with poolers (Neon)
                               check=ConnectionPool.check_connection, open=True)
    return _pool


_initializing_schema = False
_schema_initialized = False


def ensure_schema():
    global _schema_initialized, _initializing_schema
    if not _schema_initialized and not _initializing_schema:
        _initializing_schema = True
        try:
            init_schema()
            _schema_initialized = True
        finally:
            _initializing_schema = False


@contextmanager
def db():
    """One atomic transaction: commits on success, rolls back on any error."""
    ensure_schema()
    if USE_PG:
        with _get_pool().connection() as raw:
            yield Conn(raw, True)
        return
    raw = sqlite3.connect(SQLITE_PATH, isolation_level=None, timeout=10)
    raw.row_factory = sqlite3.Row
    try:
        raw.execute("BEGIN IMMEDIATE")
        yield Conn(raw, False)
        raw.execute("COMMIT")
    except BaseException:
        try:
            raw.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        raw.close()


def integrity_errors():
    if USE_PG:
        import psycopg
        return (psycopg.errors.UniqueViolation,)
    return (sqlite3.IntegrityError,)


def init_schema():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    real = "DOUBLE PRECISION" if USE_PG else "REAL"
    stmts = [
        f"CREATE TABLE IF NOT EXISTS users(id {pk}, username TEXT UNIQUE NOT NULL, full_name TEXT NOT NULL, "
        f"email TEXT NOT NULL, pw_hash TEXT NOT NULL, balance {real} NOT NULL, created_at TEXT)",
        f"CREATE TABLE IF NOT EXISTS transactions(id {pk}, sender TEXT, receiver TEXT, amount {real}, status TEXT, "
        f"fraud_score {real}, ml_score {real}, rule_score {real}, reasons TEXT, ai_note TEXT, created_at TEXT, reviewed_at TEXT)",
        f"CREATE TABLE IF NOT EXISTS rate_events(key TEXT NOT NULL, ts {real} NOT NULL)",
        "CREATE INDEX IF NOT EXISTS idx_rate ON rate_events(key, ts)",
        "CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(sender, created_at)",
    ]
    with db() as c:
        for s in stmts:
            c.execute(s)
        if not USE_PG:  # upgrade databases created by older versions of this app
            for tbl, col, typ in [("users", "created_at", "TEXT"), ("transactions", "ml_score", "REAL"),
                                  ("transactions", "rule_score", "REAL"), ("transactions", "reasons", "TEXT"),
                                  ("transactions", "ai_note", "TEXT"), ("transactions", "reviewed_at", "TEXT")]:
                if col not in [r["name"] for r in c.execute(f"PRAGMA table_info({tbl})")]:
                    c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}")
    _schema_initialized = True


# ---------------- rate limiting ----------------
class RateLimited(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = max(1, int(retry_after))


def lim(name: str, default: int) -> int:
    return int(os.getenv("RL_" + name, default))


def hit(key: str, limit: int, window: int):
    """Count this request; raise RateLimited if more than `limit` in the last `window` seconds."""
    now = time.time()
    with db() as c:
        rows = c.execute("SELECT ts FROM rate_events WHERE key=? AND ts>? ORDER BY ts", (key, now - window)).fetchall()
        if len(rows) >= limit:
            raise RateLimited(rows[0]["ts"] + window - now + 1)
        c.execute("INSERT INTO rate_events(key, ts) VALUES(?,?)", (key, now))
        if random.random() < .01:
            c.execute("DELETE FROM rate_events WHERE ts<?", (now - 86400,))


def record(*keys: str):
    now = time.time()
    with db() as c:
        for k in keys:
            c.execute("INSERT INTO rate_events(key, ts) VALUES(?,?)", (k, now))


def clear(key: str):
    with db() as c:
        c.execute("DELETE FROM rate_events WHERE key=?", (key,))


def check_lock(key: str, limit: int, window: int):
    """Locked once `limit` failures happened within `window`; unlocks `window` seconds after the last one."""
    now = time.time()
    with db() as c:
        rows = c.execute("SELECT ts FROM rate_events WHERE key=? AND ts>? ORDER BY ts", (key, now - window)).fetchall()
    if len(rows) >= limit:
        raise RateLimited(rows[-1]["ts"] + window - now + 1)
