import hashlib, hmac, json, os, secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.sessions import SessionMiddleware

from app import db as D
from app.db import RateLimited, db, hit, lim, now_s
from fraud import model as fm
from fraud.explain import llm_note, template_note
from fraud.rules import decide, rule_check

SECRET = os.getenv("SECRET_KEY", "dev-only-change-me")
START_BALANCE = 1000.0  # simulation: every new account gets a demo balance
FAIL_WINDOW = 900       # 15 minutes


@asynccontextmanager
async def lifespan(app):
    D.init_schema()
    fm.load_or_train()
    yield


app = FastAPI(title="SafeBank", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET, max_age=3600, same_site="lax",
                   https_only=os.getenv("HTTPS_ONLY") == "1")


def too_many(e: RateLimited):
    return JSONResponse({"detail": f"Too many requests. Try again in {e.retry_after} seconds."},
                        status_code=429, headers={"Retry-After": str(e.retry_after)})


@app.exception_handler(RateLimited)
async def _rl_handler(request, e):
    return too_many(e)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"  # run uvicorn with --proxy-headers behind a proxy


@app.middleware("http")
async def api_rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        try:
            await run_in_threadpool(hit, f"api:{client_ip(request)}", lim("API", 300), 60)
        except RateLimited as e:
            return too_many(e)
    return await call_next(request)


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    return f"{salt}${hashlib.pbkdf2_hmac('sha256', pw.encode(), salt.encode(), 200_000).hex()}"


DUMMY = hash_pw("dummy-password")  # keeps timing equal when the username doesn't exist


def check_pw(pw, stored):
    return hmac.compare_digest(hash_pw(pw, stored.split("$")[0]), stored)


def current(request: Request) -> str:
    u = request.session.get("u")
    if not u:
        raise HTTPException(401, "Please log in")
    return u


class Register(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[\w\.\-@]{3,40}$")
    full_name: str = Field(min_length=2, max_length=60)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=100)

    @field_validator("username", "full_name", "email", mode="before")
    @classmethod
    def strip_whitespace(cls, v):
        return v.strip() if isinstance(v, str) else v


class Login(BaseModel):
    username: str = Field(max_length=40)
    password: str = Field(max_length=100)

    @field_validator("username", mode="before")
    @classmethod
    def strip_username(cls, v):
        return v.strip() if isinstance(v, str) else v


class Transfer(BaseModel):
    receiver: str = Field(max_length=40)
    amount: float = Field(ge=0.01, le=1_000_000)


class Delete(BaseModel):
    password: str


@app.post("/api/register", status_code=201)
def register(b: Register, request: Request):
    hit(f"register:{client_ip(request)}", lim("REGISTER", 5), 3600)
    try:
        with db() as c:
            c.execute("INSERT INTO users(username,full_name,email,pw_hash,balance,created_at) VALUES(?,?,?,?,?,?)",
                      (b.username.lower(), b.full_name, b.email, hash_pw(b.password), START_BALANCE, now_s()))
    except D.integrity_errors():
        raise HTTPException(409, "Username already taken")
    request.session["u"] = b.username.lower()
    return {"ok": True}


@app.post("/api/login")
def login(b: Login, request: Request):
    ip, name = client_ip(request), b.username.lower()
    hit(f"login-ip:{ip}", lim("LOGIN_IP", 10), 60)                                   # brute-force speed limit per IP
    D.check_lock(f"fail:{name}:{ip}", lim("LOCK_FAILS", 5), FAIL_WINDOW)             # lockout for this user from this IP
    D.check_lock(f"fail-user:{name}", lim("LOCK_FAILS_USER", 20), FAIL_WINDOW)       # distributed-attack guard
    with db() as c:
        u = c.execute("SELECT pw_hash FROM users WHERE username=?", (name,)).fetchone()
    if not check_pw(b.password, u["pw_hash"] if u else DUMMY) or not u:
        D.record(f"fail:{name}:{ip}", f"fail-user:{name}")
        raise HTTPException(401, "Wrong username or password")
    D.clear(f"fail:{name}:{ip}")
    request.session["u"] = name
    return {"ok": True}


@app.post("/api/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@app.get("/api/me")
def me(request: Request):
    u = current(request)
    with db() as c:
        r = c.execute("SELECT username,full_name,balance FROM users WHERE username=?", (u,)).fetchone()
    if not r:
        request.session.clear()
        raise HTTPException(401, "Please log in")
    return dict(r)


@app.get("/api/transactions")
def transactions(request: Request):
    u = current(request)
    with db() as c:
        rows = c.execute("SELECT sender,receiver,amount,status,created_at FROM transactions "
                         "WHERE sender=? OR (receiver=? AND status='COMPLETED') ORDER BY id DESC LIMIT 20",
                         (u, u)).fetchall()
    return [dict(r) for r in rows]


def age_hours(created_at):
    if not created_at:
        return 9999.0
    t = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t).total_seconds() / 3600


def context(c, me, to, amount, balance, hour):
    one = lambda sql, *a: c.execute(sql, a).fetchone()
    comp = one("SELECT COUNT(*) n, COALESCE(AVG(amount),0) a FROM transactions WHERE sender=? AND status='COMPLETED'", me)
    return dict(
        amount=amount, balance=balance, hour=hour, n_completed=comp["n"], avg_amount=comp["a"],
        n_last_hour=one("SELECT COUNT(*) n FROM transactions WHERE sender=? AND created_at>=?", me, now_s(-3600))["n"],
        blocked_24h=one("SELECT COUNT(*) n FROM transactions WHERE sender=? AND status='BLOCKED' AND created_at>=?", me, now_s(-86400))["n"],
        new_receiver=one("SELECT COUNT(*) n FROM transactions WHERE sender=? AND receiver=? AND status='COMPLETED'", me, to)["n"] == 0,
        receiver_age_h=age_hours(one("SELECT created_at FROM users WHERE username=?", to)["created_at"]))


def move(c, sender, receiver, amount):
    c.execute("UPDATE users SET balance=balance-? WHERE username=?", (amount, sender))
    c.execute("UPDATE users SET balance=balance+? WHERE username=?", (amount, receiver))


def lock_balances(c, a, b):
    """Lock both accounts (in a fixed order, so two transfers can't deadlock) and return their balances."""
    rows = c.execute("SELECT username,balance FROM users WHERE username IN (?,?) ORDER BY username FOR UPDATE", (a, b)).fetchall()
    return {r["username"]: r["balance"] for r in rows}


def upgrade_note(tid, facts):
    note = llm_note(facts)
    if note:
        with db() as c:
            c.execute("UPDATE transactions SET ai_note=? WHERE id=?", (note, tid))


MESSAGES = {"COMPLETED": "Transfer successful",
            "HELD": "Transfer held for a quick security review. No money has moved yet; check your history for the result.",
            "BLOCKED": "Transfer blocked for your safety. No money was moved."}


@app.post("/api/transfer")
def transfer(t: Transfer, request: Request, bg: BackgroundTasks):
    me_, to = current(request), t.receiver.lower()
    if to == me_:
        raise HTTPException(400, "You cannot send money to yourself")
    hit(f"transfer:{me_}", lim("TRANSFER", 10), 60)
    amount, hour = round(t.amount, 2), datetime.now(timezone.utc).hour
    with db() as c:
        bal = lock_balances(c, me_, to)
        if me_ not in bal:
            raise HTTPException(401, "Please log in")
        if to not in bal:
            raise HTTPException(404, "Receiver not found")
        if amount > bal[me_]:
            raise HTTPException(400, "Insufficient balance")
        ml = fm.risk(amount, bal[me_], bal[me_] - amount, bal[to], hour)
        rs, why = rule_check(context(c, me_, to, amount, bal[me_], hour))
        status = decide(ml, rs)
        reasons = ml["reasons"] + why + (["behaviour is statistically unusual (anomaly detector)"] if ml["anomaly"] >= .995 else [])
        if status == "COMPLETED":
            move(c, me_, to, amount)
        tid = c.execute("INSERT INTO transactions(sender,receiver,amount,status,fraud_score,ml_score,rule_score,reasons,ai_note,created_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?) RETURNING id",
                        (me_, to, amount, status, max(ml["score"], rs), ml["score"], rs, json.dumps(reasons),
                         template_note(me_, to, amount, status, reasons), now_s())).fetchone()["id"]
    if status != "COMPLETED" and os.getenv("ANTHROPIC_API_KEY"):
        bg.add_task(upgrade_note, tid, f"Amount ${amount:,.2f}; sender balance ${bal[me_]:,.2f}; "
                    f"ML fraud score {ml['score']}; rule score {rs}; signals: {'; '.join(reasons) or 'none'}")
    return {"status": status.lower(), "message": MESSAGES[status]}


@app.delete("/api/account")
def delete_account(b: Delete, request: Request):
    u = current(request)
    hit(f"delete:{u}", 5, 3600)
    with db() as c:
        r = c.execute("SELECT pw_hash FROM users WHERE username=?", (u,)).fetchone()
        if not r or not check_pw(b.password, r["pw_hash"]):
            raise HTTPException(403, "Wrong password")
        c.execute("DELETE FROM users WHERE username=?", (u,))
    request.session.clear()
    return {"ok": True}


@app.get("/health")
def health():
    with db() as c:
        c.execute("SELECT 1")
    return {
        "status": "ok",
        "database": "postgres" if D.USE_PG else "sqlite",
        "details": "Connected to PostgreSQL" if D.USE_PG else "Using local SQLite (bank.db). To connect to Supabase PostgreSQL, set your DB password in .env (DATABASE_URL or SUPABASE_DB_PASSWORD)."
    }


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


class Review(BaseModel):
    id: int
    action: str = Field(pattern="^(approve|reject)$")


def admin(request: Request):
    hit(f"admin:{client_ip(request)}", lim("ADMIN", 60), 60)
    tok = os.getenv("ADMIN_TOKEN")
    if not tok or not hmac.compare_digest(request.headers.get("x-admin-token", ""), tok):
        D.record(f"admin-fail:{client_ip(request)}")
        D.check_lock(f"admin-fail:{client_ip(request)}", 10, FAIL_WINDOW)
        raise HTTPException(403, "Admin access denied (set ADMIN_TOKEN and send it as x-admin-token)")


@app.get("/api/admin/queue")
def admin_queue(request: Request):
    admin(request)
    with db() as c:
        rows = c.execute("SELECT id,sender,receiver,amount,ml_score,rule_score,reasons,ai_note,created_at "
                         "FROM transactions WHERE status='HELD' ORDER BY id").fetchall()
    return [{**dict(r), "reasons": json.loads(r["reasons"] or "[]")} for r in rows]


@app.get("/api/admin/stats")
def admin_stats(request: Request):
    admin(request)
    with db() as c:
        rows = c.execute("SELECT status, COUNT(*) n, COALESCE(SUM(amount),0) amt FROM transactions GROUP BY status").fetchall()
        users = c.execute("SELECT COUNT(*) n FROM users").fetchone()["n"]
    if not fm._state:
        fm.load_or_train()
    return {"users": users, "by_status": {r["status"]: {"count": r["n"], "amount": r["amt"]} for r in rows},
            "model": fm._state.get("metrics", {})}


@app.post("/api/admin/review")
def admin_review(b: Review, request: Request):
    admin(request)
    with db() as c:
        t = c.execute("SELECT * FROM transactions WHERE id=? AND status='HELD' FOR UPDATE", (b.id,)).fetchone()
        if not t:
            raise HTTPException(404, "No held transfer with that id")
        if b.action == "approve":
            bal = lock_balances(c, t["sender"], t["receiver"])
            if t["sender"] not in bal or t["receiver"] not in bal or bal[t["sender"]] < t["amount"]:
                raise HTTPException(409, "Cannot approve: an account is gone or the balance is too low now")
            move(c, t["sender"], t["receiver"], t["amount"])
        c.execute("UPDATE transactions SET status=?, reviewed_at=? WHERE id=?",
                  ("COMPLETED" if b.action == "approve" else "REJECTED", now_s(), b.id))
    return {"ok": True}


@app.get("/admin")
def admin_page():
    return FileResponse(Path(__file__).parent.parent / "static" / "admin.html")
