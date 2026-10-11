"""Transfer-fraud model: gradient boosting + anomaly detector + explanations.
Trained strictly on the Kaggle PaySim dataset (ealaxi/paysim1).
"""
import argparse, json
from pathlib import Path

import joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PATH = Path("models/bank_model.joblib")
DATA_DIR = Path("data")
_state: dict = {}

GROUPS = {  # feature groups used for plain-English explanations
    "it empties (or nearly empties) the sender's account": ["frac", "drain", "new_org"],
    "the receiving account was empty before": ["dest_empty", "old_dest"],
    "the amount is unusually large": ["amount", "log_amount"],
    "it happens at an unusual hour": ["hour", "is_night"],
}


def feats(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "amount": d.amount, "old_org": d.old_org, "new_org": d.new_org, "old_dest": d.old_dest,
        "frac": d.amount / d.old_org.clip(lower=1),
        "drain": ((d.old_org > 0) & (d.amount >= d.old_org * 0.999)).astype(int),
        "dest_empty": (d.old_dest == 0).astype(int),
        "hour": d.hour, "is_night": (d.hour <= 5).astype(int),
        "is_transfer": d.is_transfer, "log_amount": np.log1p(d.amount)})


def find_default_csv() -> Path | None:
    """Finds the Kaggle PaySim CSV file inside data/."""
    if not DATA_DIR.exists():
        return None
    files = sorted(DATA_DIR.glob("*.csv"))
    return next((f for f in files if "PS_" in f.name or "paysim" in f.name.lower()), files[0] if files else None)


def load_csv(path):
    """Kaggle 'PaySim' (ealaxi/paysim1). Only TRANSFER and CASH_OUT contain fraud."""
    cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "isFraud"]
    df = pd.read_csv(path, usecols=cols)
    df = df[df.type.isin(["TRANSFER", "CASH_OUT"])].rename(columns={
        "oldbalanceOrg": "old_org", "newbalanceOrig": "new_org", "oldbalanceDest": "old_dest", "isFraud": "y"})
    df["hour"] = df.step % 24
    df["is_transfer"] = (df.type == "TRANSFER").astype(int)
    return df.drop(columns="type").reset_index(drop=True)


def _threshold(y, p, min_precision=.8):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= min_precision)[0]
    if len(ok):
        return float(th[ok[np.argmax(rc[:-1][ok])]])
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
    return float(th[np.argmax(f1)])


def train(csv=None, out=PATH):
    """Train the model strictly on the Kaggle PaySim dataset."""
    out = Path(out)
    if csv is None:
        csv = find_default_csv()
    if csv is None or not Path(csv).exists():
        raise FileNotFoundError(
            "Kaggle PaySim dataset not found in data/. "
            "Please run 'python -m scripts.sync_kaggle' to download and sync the Kaggle dataset."
        )

    print(f"Training on Kaggle PaySim dataset: {csv}")
    df = load_csv(csv)
    X, y, step = feats(df), df.y.values, df.step.values
    tr, va, te = step <= 600, (step > 600) & (step <= 680), step > 680  # time-based split
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=.08, max_depth=6,
                                       class_weight="balanced", random_state=1).fit(X[tr], y[tr])
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")).fit(X[tr], y[tr])
    thr = _threshold(y[va], m.predict_proba(X[va])[:, 1])
    p = m.predict_proba(X[te])[:, 1]
    pred = (p >= thr).astype(int)

    legit = X[tr & (y == 0)]
    legit = legit.sample(min(50_000, len(legit)), random_state=1)
    iso = IsolationForest(n_estimators=150, random_state=1).fit(legit)
    ref = np.sort(-iso.score_samples(legit))
    a = np.searchsorted(ref, -iso.score_samples(X[te])) / len(ref)

    tn, fp, fn, tp = confusion_matrix(y[te], pred).ravel()
    metrics = {
        "data_source": "kaggle_paysim_csv",
        "rows": int(len(df)),
        "fraud_rate": round(float(y.mean()), 5),
        "test_rows": int(te.sum()),
        "test_frauds": int(y[te].sum()),
        "pr_auc": round(average_precision_score(y[te], p), 4),
        "roc_auc": round(roc_auc_score(y[te], p), 4),
        "baseline_logreg_pr_auc": round(average_precision_score(y[te], base.predict_proba(X[te])[:, 1]), 4),
        "block_threshold": round(thr, 4),
        "precision": round(precision_score(y[te], pred, zero_division=0), 4),
        "recall": round(recall_score(y[te], pred), 4),
        "confusion": dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        "anomaly_detector_recall_at_top1pct": round(float((a[y[te] == 1] >= .99).mean()), 4),
        "sklearn": sklearn.__version__}

    out.parent.mkdir(exist_ok=True)
    joblib.dump({"model": m, "iso": iso, "iso_ref": ref, "threshold": thr,
                 "review_threshold": min(.5, .6 * thr), "baseline": legit.median().to_dict(),
                 "metrics": metrics}, out)
    out.with_name(out.stem + "_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Training finished successfully. Metrics:")
    print(json.dumps(metrics, indent=2))


def load_or_train():
    try:
        s = joblib.load(PATH) if PATH.exists() else None
    except Exception:
        s = None
    if s is None:
        print("No pre-trained model found. Initiating training on Kaggle PaySim dataset...")
        train()
        s = joblib.load(PATH)
    _state.update(s)


def risk(amount, old_org, new_org, old_dest, hour=12, is_transfer=1) -> dict:
    if not _state:
        load_or_train()
    S = _state
    X = feats(pd.DataFrame([dict(amount=amount, old_org=old_org, new_org=new_org,
                                 old_dest=old_dest, hour=hour, is_transfer=is_transfer)]))
    p = float(S["model"].predict_proba(X)[:, 1][0])
    anomaly = float(np.searchsorted(S["iso_ref"], -S["iso"].score_samples(X)[0]) / len(S["iso_ref"]))
    reasons = []
    if p >= .2:  # occlusion: neutralise one feature group at a time, see how much the score falls
        found = []
        for name, cols in GROUPS.items():
            X2 = X.copy()
            for c in cols:
                X2[c] = S["baseline"][c]
            d = p - float(S["model"].predict_proba(X2)[:, 1][0])
            if d >= .05:
                found.append((d, name))
        reasons = [n for _, n in sorted(found, reverse=True)[:3]]
    return dict(score=round(p, 4), anomaly=round(anomaly, 4), reasons=reasons,
                threshold=S["threshold"], review_threshold=S["review_threshold"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    train(ap.parse_args().csv)
