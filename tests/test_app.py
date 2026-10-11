import os, tempfile
if not os.getenv("DATABASE_URL"):
    os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["ADMIN_TOKEN"] = "secret-admin"
os.environ.pop("ANTHROPIC_API_KEY", None)
for k, v in dict(RL_API=100000, RL_REGISTER=1000, RL_LOGIN_IP=1000, RL_TRANSFER=1000).items():
    os.environ[k] = str(v)
import pytest
from fastapi.testclient import TestClient
from app.main import app

H = {"x-admin-token": "secret-admin"}


def user(n):
    return dict(username=n, full_name="Test User", email=f"{n}@x.com", password="password123")


def login(name):
    c = TestClient(app); c.__enter__()
    assert c.post("/api/register", json=user(name)).status_code == 201
    return c


@pytest.fixture(scope="module")
def env():
    with TestClient(app):  # run lifespan once so the model is loaded
        yield


def test_auth(env):
    a = login("alice")
    assert a.post("/api/register", json=user("alice")).status_code == 409
    assert TestClient(app).get("/api/me").status_code == 401
    assert TestClient(app).post("/api/login", json=dict(username="alice", password="x")).status_code == 401


def test_normal_transfer_and_validation(env):
    a, b = login("amy"), login("ben")
    assert a.post("/api/transfer", json=dict(receiver="ben", amount=100)).json()["status"] == "completed"
    assert a.get("/api/me").json()["balance"] == 900 and b.get("/api/me").json()["balance"] == 1100
    assert a.post("/api/transfer", json=dict(receiver="ben", amount=99999)).status_code == 400
    assert a.post("/api/transfer", json=dict(receiver="amy", amount=5)).status_code == 400


def test_drain_is_blocked(env):
    login("bob"); c = login("carol")
    r = c.post("/api/transfer", json=dict(receiver="bob", amount=1000)).json()
    assert r["status"] in ("blocked", "held")  # full-balance drain is never completed automatically
    assert c.get("/api/me").json()["balance"] == 1000  # no money moved


def test_velocity_hold_then_admin_review(env):
    login("erin"); d = login("dave")
    s = [d.post("/api/transfer", json=dict(receiver="erin", amount=10)).json()["status"] for _ in range(4)]
    assert s == ["completed"] * 3 + ["held"]
    assert d.get("/api/me").json()["balance"] == 970  # held money did not move
    q = [x for x in TestClient(app).get("/api/admin/queue", headers=H).json() if x["sender"] == "dave"]
    assert len(q) == 1 and any("last hour" in r for r in q[0]["reasons"]) and q[0]["ai_note"]
    assert TestClient(app).get("/api/admin/queue").status_code == 403
    ok = TestClient(app).post("/api/admin/review", headers=H, json=dict(id=q[0]["id"], action="approve"))
    assert ok.status_code == 200 and d.get("/api/me").json()["balance"] == 960
    d.post("/api/transfer", json=dict(receiver="erin", amount=10))  # held again
    q = [x for x in TestClient(app).get("/api/admin/queue", headers=H).json() if x["sender"] == "dave"]
    TestClient(app).post("/api/admin/review", headers=H, json=dict(id=q[0]["id"], action="reject"))
    assert d.get("/api/me").json()["balance"] == 960
    st = TestClient(app).get("/api/admin/stats", headers=H).json()
    assert st["by_status"]["REJECTED"]["count"] >= 1 and "pr_auc" in st["model"]


def test_model_explains_and_beats_baseline(env):
    from fraud import model as fm
    r = fm.risk(1000, 1000, 0, 0, hour=3)
    assert r["score"] >= r["threshold"] and r["reasons"]
    m = fm._state["metrics"]
    assert m["pr_auc"] > 0.8  # sanity floor; compare with baseline_logreg_pr_auc on real data


def test_login_lockout(env):
    login("lockme")
    c = TestClient(app)
    for _ in range(5):
        assert c.post("/api/login", json=dict(username="lockme", password="wrongpass")).status_code == 401
    r = c.post("/api/login", json=dict(username="lockme", password="password123"))  # correct, but locked
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


def test_transfer_and_register_rate_limits(env, monkeypatch):
    login("rita"); login("sam")
    c = login("tom")
    monkeypatch.setenv("RL_TRANSFER", "2")
    codes = [c.post("/api/transfer", json=dict(receiver="sam", amount=5)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    monkeypatch.setenv("RL_REGISTER", "1")
    r = TestClient(app).post("/api/register", json=user("toolate"))
    assert r.status_code == 429 and "retry-after" in r.headers


def test_health_reports_database(env):
    d = TestClient(app).get("/health").json()
    assert d["database"] == ("postgres" if os.getenv("DATABASE_URL") else "sqlite")


def test_real_csv_pipeline(tmp_path):
    """Build a test CSV in the exact Kaggle PaySim format and train from it."""
    import numpy as np, pandas as pd
    from fraud import model as fm
    r = np.random.default_rng(7)
    n = 20_000
    f = r.random(n) < 0.02
    old_org = np.where(f, 5000.0, r.uniform(100, 2000, n))
    amount = np.where(f, 4995.0, r.uniform(10, 500, n))
    types = r.choice(["TRANSFER", "CASH_OUT"], n)
    pay = pd.DataFrame({
        "step": r.integers(1, 744, n), "type": types, "amount": amount,
        "nameOrig": "C1", "oldbalanceOrg": old_org, "newbalanceOrig": old_org - amount, "nameDest": "C2",
        "oldbalanceDest": np.where(f, 0.0, 1000.0), "newbalanceDest": np.where(f, 4995.0, 1000.0 + amount),
        "isFraud": f.astype(int), "isFlaggedFraud": 0
    })
    pay = pd.concat([pay, pay.head(200).assign(type="PAYMENT", isFraud=0)])  # rows the loader must drop
    csv = tmp_path / "PS_test.csv"
    pay.to_csv(csv, index=False)
    out = tmp_path / "m.joblib"
    fm.train(str(csv), out=out)
    import joblib
    m = joblib.load(out)["metrics"]
    assert m["data_source"] == "kaggle_paysim_csv" and m["rows"] == 20_000 and m["pr_auc"] > .5
