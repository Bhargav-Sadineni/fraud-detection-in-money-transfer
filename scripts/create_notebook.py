"""Generates PaySim_Fraud_Detection_Model.ipynb containing the complete training and evaluation pipeline."""
import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown(content):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")]
    })

def add_code(content):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in content.split("\n")]
    })

# Section 1
add_markdown("""# 🏦 SafeBank: PaySim AI Fraud Detection Model
### End-to-End Training, Isolation Forest Anomaly Detection, Calibration & Export

This notebook provides the complete reproducible training pipeline for the SafeBank fraud defense system trained on the Kaggle **PaySim** dataset (`ealaxi/paysim1`).

---
### 🔍 Pipeline Architecture:
1. **PaySim Ingestion**: Filtering to transaction types where fraud exists (`TRANSFER` and `CASH_OUT`)
2. **Feature Engineering**:
   - `frac`: Fraction of balance drained (`amount / old_balance`)
   - `drain`: Binary flag for transferring $\\ge 99.9\\%$ of total funds
   - `dest_empty`: Binary flag for recipient account starting with zero funds (mule wallet indicator)
   - `is_night`: Binary flag for transactions between 00:00 and 05:00 UTC
   - `log_amount`: Non-linear scaling $\\log(1 + amount)$
3. **Temporal Train/Validation/Test Split**: Realistic time-based partition
4. **Supervised Classifier**: `HistGradientBoostingClassifier` with balanced class weights
5. **Anomaly Detector**: `IsolationForest` trained on legitimate transactions
6. **Calibrated Thresholds**: Optimizing PR-AUC and blocking thresholds
7. **Explainable AI (Feature Occlusion)**: Generating plain-English reasons for flagged transfers
8. **Export**: Serializing to `models/bank_model.joblib` for real-time web inference""")

# Section 2
add_markdown("## 1. Environment & Dependencies")
add_code("""import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"Pandas Version:       {pd.__version__}")
print(f"NumPy Version:        {np.__version__}")""")

# Section 3
add_markdown("## 2. Feature Engineering Definitions")
add_code("""GROUPS = {
    "it empties (or nearly empties) the sender's account": ["frac", "drain", "new_org"],
    "the receiving account was empty before": ["dest_empty", "old_dest"],
    "the amount is unusually large": ["amount", "log_amount"],
    "it happens at an unusual hour": ["hour", "is_night"],
}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Extract engineered fraud signals from financial transaction data.\"\"\"
    return pd.DataFrame({
        "amount": df.amount,
        "old_org": df.old_org,
        "new_org": df.new_org,
        "old_dest": df.old_dest,
        "frac": df.amount / df.old_org.clip(lower=1),
        "drain": ((df.old_org > 0) & (df.amount >= df.old_org * 0.999)).astype(int),
        "dest_empty": (df.old_dest == 0).astype(int),
        "hour": df.hour,
        "is_night": (df.hour <= 5).astype(int),
        "is_transfer": df.is_transfer,
        "log_amount": np.log1p(df.amount),
    })""")

# Section 4
add_markdown("## 3. Data Ingestion (PaySim CSV / Synthetic Generator)")
add_code("""def load_paysim_csv(path: str) -> pd.DataFrame:
    \"\"\"Load Kaggle PaySim CSV (ealaxi/paysim1). Only TRANSFER and CASH_OUT contain fraud.\"\"\"
    cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "isFraud"]
    print(f"Loading {path}...")
    df = pd.read_csv(path, usecols=cols)
    df = df[df.type.isin(["TRANSFER", "CASH_OUT"])].rename(columns={
        "oldbalanceOrg": "old_org",
        "newbalanceOrig": "new_org",
        "oldbalanceDest": "old_dest",
        "isFraud": "y"
    })
    df["hour"] = df.step % 24
    df["is_transfer"] = (df.type == "TRANSFER").astype(int)
    return df.drop(columns="type").reset_index(drop=True)


def generate_synthetic_data(n=100_000, seed=1) -> pd.DataFrame:
    \"\"\"Synthetic demo transaction generator matching PaySim distributions.\"\"\"
    r = np.random.default_rng(seed)
    f = r.random(n) < 0.012
    old_org = np.where(f, r.lognormal(8.5, 1.0, n), r.lognormal(7.5, 1.2, n))
    drain = f & (r.random(n) < 0.75)
    frac = np.where(drain, 1.0, np.where(f, r.uniform(.4, 1, n),
                    np.where(r.random(n) < .004, 1.0, r.beta(1.5, 8, n))))
    amount = old_org * frac
    old_dest = np.where(f, np.where(r.random(n) < .6, 0, r.lognormal(7, 1, n)),
                        np.where(r.random(n) < .12, 0, r.lognormal(7.5, 1.3, n)))
    hour = np.where(f & (r.random(n) < .5), r.integers(0, 6, n), r.integers(6, 24, n))
    hour = np.where(~f & (r.random(n) < .05), r.integers(0, 6, n), hour)
    return pd.DataFrame(dict(step=r.integers(1, 744, n), amount=amount, old_org=old_org,
                             new_org=old_org - amount, old_dest=old_dest, hour=hour,
                             is_transfer=r.integers(0, 2, n), y=f.astype(int)))


# Locate CSV in data/ or fallback to synthetic
data_dir = Path("data")
csv_files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []

if csv_files:
    print(f"Found PaySim CSV: {csv_files[0]}")
    df = load_paysim_csv(str(csv_files[0]))
else:
    print("No CSV in data/ -> Generating 100,000 synthetic PaySim transactions for simulation")
    df = generate_synthetic_data(100_000)

print(f"Total Transactions: {len(df):,}")
print(f"Fraud Rate:         {df.y.mean():.4%}")
df.head()""")

# Section 5
add_markdown("## 4. Temporal Split & Feature Matrix")
add_code("""X = extract_features(df)
y = df.y.values
step = df.step.values

# Temporal Train (<= 600) / Validation (600-680) / Test (> 680)
tr = step <= 600
va = (step > 600) & (step <= 680)
te = step > 680

print(f"Training set:   {tr.sum():,} rows ({y[tr].sum():,} frauds)")
print(f"Validation set: {va.sum():,} rows ({y[va].sum():,} frauds)")
print(f"Test set:       {te.sum():,} rows ({y[te].sum():,} frauds)")""")

# Section 6
add_markdown("## 5. Threshold Calibration Helper")
add_code("""def find_threshold(y_true, y_prob, min_precision=0.8):
    \"\"\"Finds optimal decision threshold balancing precision and recall.\"\"\"
    pr, rc, th = precision_recall_curve(y_true, y_prob)
    ok = np.where(pr[:-1] >= min_precision)[0]
    if len(ok):
        return float(th[ok[np.argmax(rc[:-1][ok])]])
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(th[np.argmax(f1)])""")

# Section 7
add_markdown("## 6. Model Training (`HistGradientBoostingClassifier`)")
add_code("""print("Fitting HistGradientBoostingClassifier on training set...")
clf = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.08,
    max_depth=6,
    class_weight="balanced",
    random_state=1
).fit(X[tr], y[tr])

print("Fitting Baseline Logistic Regression for benchmark...")
baseline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=300, class_weight="balanced")
).fit(X[tr], y[tr])

# Calibrate decision threshold on validation set
val_probs = clf.predict_proba(X[va])[:, 1]
thr = find_threshold(y[va], val_probs)
print(f"Optimized Blocking Threshold: {thr:.4f}")""")

# Section 8
add_markdown("## 7. Anomaly Detector Training (`IsolationForest`)")
add_code("""print("Fitting IsolationForest on legitimate baseline transactions...")
legit = X[tr & (y == 0)]
legit_sample = legit.sample(min(50_000, len(legit)), random_state=1)

iso = IsolationForest(n_estimators=150, random_state=1).fit(legit_sample)
iso_ref = np.sort(-iso.score_samples(legit_sample))
test_anom = np.searchsorted(iso_ref, -iso.score_samples(X[te])) / len(iso_ref)

print("Anomaly detector trained.")""")

# Section 9
add_markdown("## 8. Test Set Evaluation & Metrics")
add_code("""p = clf.predict_proba(X[te])[:, 1]
pred = (p >= thr).astype(int)

pr_auc = average_precision_score(y[te], p)
roc_auc = roc_auc_score(y[te], p)
base_pr_auc = average_precision_score(y[te], baseline.predict_proba(X[te])[:, 1])
tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()

metrics = {
    "rows": int(len(df)),
    "fraud_rate": round(float(y.mean()), 5),
    "test_rows": int(te.sum()),
    "test_frauds": int(y[te].sum()),
    "pr_auc": round(pr_auc, 4),
    "roc_auc": round(roc_auc, 4),
    "baseline_logreg_pr_auc": round(base_pr_auc, 4),
    "block_threshold": round(thr, 4),
    "precision": round(precision_score(y[te], pred, zero_division=0), 4),
    "recall": round(recall_score(y[te], pred), 4),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "anomaly_detector_recall_at_top1pct": round(float((test_anom[y[te] == 1] >= 0.99).mean()), 4),
    "sklearn": sklearn.__version__,
}

print("=== Model Metrics ===")
print(json.dumps(metrics, indent=2))""")

# Section 10
add_markdown("## 9. Feature Occlusion Explainability")
add_code("""def explain_transaction(amount, old_org, new_org, old_dest, hour=12, is_transfer=1):
    \"\"\"Scores an individual transfer and explains why it was flagged.\"\"\"
    X_single = extract_features(pd.DataFrame([dict(
        amount=amount, old_org=old_org, new_org=new_org,
        old_dest=old_dest, hour=hour, is_transfer=is_transfer
    )]))
    score = float(clf.predict_proba(X_single)[:, 1][0])
    anomaly = float(np.searchsorted(iso_ref, -iso.score_samples(X_single)[0]) / len(iso_ref))
    
    reasons = []
    baseline_median = legit_sample.median().to_dict()
    if score >= 0.2:
        found = []
        for name, cols in GROUPS.items():
            X_mod = X_single.copy()
            for c in cols:
                X_mod[c] = baseline_median[c]
            drop = score - float(clf.predict_proba(X_mod)[:, 1][0])
            if drop >= 0.05:
                found.append((drop, name))
        reasons = [name for _, name in sorted(found, reverse=True)[:3]]
        
    decision = "BLOCKED" if score >= thr else ("HELD" if score >= 0.6 * thr or anomaly >= 0.995 else "COMPLETED")
    return {
        "score": round(score, 4),
        "decision": decision,
        "anomaly": round(anomaly, 4),
        "reasons": reasons
    }

print("Normal Transfer:")
print(explain_transaction(amount=50.0, old_org=1000.0, new_org=950.0, old_dest=250.0, hour=14))

print("\\nFraudulent Account Drain:")
print(explain_transaction(amount=1000.0, old_org=1000.0, new_org=0.0, old_dest=0.0, hour=3))""")

# Section 11
add_markdown("## 10. Serializing the Model Bundle for Production")
add_code("""models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
save_path = models_dir / "bank_model.joblib"

bundle = {
    "model": clf,
    "iso": iso,
    "iso_ref": iso_ref,
    "threshold": thr,
    "review_threshold": min(0.5, 0.6 * thr),
    "baseline": legit_sample.median().to_dict(),
    "metrics": metrics,
}

joblib.dump(bundle, save_path)
print(f"Bundle successfully saved to: {save_path.resolve()}")
print(f"File Size: {save_path.stat().st_size / 1024 / 1024:.2f} MB")""")

# Write notebook
out_path = Path("PaySim_Fraud_Detection_Model.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {out_path.resolve()}")
