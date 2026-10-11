# SafeBank v3: persistent database, rate limiting, Kaggle-trained fraud model

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Bhargav-Sadineni/fraud-detection-in-money-transfer)

## 1. Persistent database (Postgres)
The app uses **Postgres when `DATABASE_URL` is set**, otherwise a local SQLite file (`bank.db`).
Tables are created automatically on startup. Rate-limit counters live in the same database.

Render's own free Postgres is **deleted after 30 days**, so use a free host that doesn't expire, e.g.
**Neon** (neon.tech) or **Supabase**:
1. Create a project and copy its **connection string**
   (looks like `postgresql://user:password@host/dbname?sslmode=require`).
2. Local test:
   You can place your credentials in `.env`:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=https://kdlpsqejqymofmiuelur.supabase.co
   NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=sb_publishable_1bjb0Nv72r5ECuQdCmHIZg_4yAOqNYt
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.kdlpsqejqymofmiuelur.supabase.co:5432/postgres
   ADMIN_TOKEN=demo-admin
   ```
   Or via PowerShell:
   ```powershell
   $env:DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.kdlpsqejqymofmiuelur.supabase.co:5432/postgres"
   $env:ADMIN_TOKEN="demo-admin"
   python -m uvicorn app.main:app --port 8001
   ```
   Open http://127.0.0.1:8001/health: it must show `"database":"postgres"`. Register, stop the server, start it
   again: your account and balance are still there.
3. Render: Dashboard -> your service -> **Environment** -> add `DATABASE_URL` with the same string -> Save (it redeploys).

## 2. Rate limiting and login lockout (all stored in the DB, shared across instances)
| What | Limit | Response |
|---|---|---|
| Any `/api/*` call | 300 per minute per IP | 429 + `Retry-After` |
| Login attempts | 10 per minute per IP | 429 |
| Wrong passwords | 5 per 15 min per username+IP -> locked 15 min; 20 per 15 min for one username from any IP | 429 |
| Register | 5 per hour per IP | 429 |
| Transfers | 10 per minute per user | 429 |
| Delete account | 5 per hour per user | 429 |
| Admin API | 60 per minute per IP; 10 wrong tokens -> locked 15 min | 429 / 403 |

Override any limit with env vars `RL_API`, `RL_LOGIN_IP`, `RL_LOCK_FAILS`, `RL_LOCK_FAILS_USER`, `RL_REGISTER`, `RL_TRANSFER`, `RL_ADMIN`.
The Docker image runs uvicorn with `--proxy-headers` so limits apply to the real client IP behind Render.
Also: transfers lock both accounts while they run (`SELECT ... FOR UPDATE` on Postgres), so two simultaneous
transfers can't spend the same money.

## 3. Replace the demo model with the real Kaggle dataset
The demo model was trained on generated data. To train on **PaySim** (Kaggle `ealaxi/paysim1`, ~6.3M rows):
1. `pip install -r requirements-train.txt`
2. Get a Kaggle token: kaggle.com -> your profile -> **Settings** -> **API** -> **Create New Token**. Save the
   downloaded `kaggle.json` to `C:\Users\<you>\.kaggle\kaggle.json`.
3. Run `python -m scripts.sync_kaggle`. It will:
   - download the dataset into `data/` (or use a CSV already there; `--csv path` to point at one),
   - train a **candidate** model with a time-based split (steps 1-600 train, 601-680 validate, 681-743 test),
   - print old vs new metrics (PR-AUC, precision, recall, logistic-regression baseline),
   - **promote** it to `models/bank_model.joblib` only if it passes sanity checks (>=50 test frauds, PR-AUC >= 0.5;
     `--force` overrides),
   - pin `scikit-learn` in `requirements.txt` to the version used for training.
   No Kaggle CLI? Download the CSV by hand from https://www.kaggle.com/datasets/ealaxi/paysim1 into `data/`.
4. Deploy it: `git add models requirements.txt && git commit -m "Kaggle model" && git push`. Render rebuilds and
   the Docker image uses the committed model. Confirm in `/admin` -> Model: source = `kaggle_paysim_csv`.

Training uses a lot of memory (roughly 8 GB; I could not measure it here). PaySim is itself simulated, so good scores
on it do not guarantee performance on real bank data.

## 4. Run / deploy summary
```powershell
pip install -r requirements.txt
python -m fraud.model            # demo model (skip if you ran sync_kaggle)
$env:ADMIN_TOKEN="demo-admin"
python -m uvicorn app.main:app --reload --port 8001     # app, and /admin
pytest -q                        # runs on SQLite; set DATABASE_URL to test against Postgres
```
Deploy: push to GitHub -> Render **New -> Blueprint** (`render.yaml`), then set `DATABASE_URL` in Environment.
`SECRET_KEY` and `ADMIN_TOKEN` are generated for you.

## Known limits
Balances are stored as floating-point numbers (use integer cents or NUMERIC for real money). No email/OTP
verification, no password reset, no audit log beyond the `transactions` table, no model-drift monitoring.
