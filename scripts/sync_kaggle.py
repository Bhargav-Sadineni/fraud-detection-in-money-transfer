"""Replace the demo model with one trained on the real Kaggle PaySim dataset.

  python -m scripts.sync_kaggle                 # download (if needed), train, validate, promote
  python -m scripts.sync_kaggle --csv path.csv  # use a CSV you already downloaded
  python -m scripts.sync_kaggle --force         # promote even if the sanity checks fail
"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

import joblib, sklearn

from fraud import model as fm

DATASET, DATA = "ealaxi/paysim1", Path("data")
CANDIDATE = Path("models/candidate.joblib")


def find_csv():
    files = sorted(DATA.glob("*.csv"))
    return next((f for f in files if "PS_" in f.name or "paysim" in f.name.lower()), files[0] if files else None)


def download():
    DATA.mkdir(exist_ok=True)
    zip_path = DATA / "paysim1.zip"
    print("Downloading PaySim dataset from Kaggle...")
    try:
        import httpx, os, zipfile
        token = os.getenv("KAGGLE_API_TOKEN")
        if not token:
            token_file = Path.home() / ".kaggle" / "access_token"
            if token_file.exists():
                token = token_file.read_text().strip()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=300.0) as r:
            if r.status_code == 200:
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                print("Extracting dataset...")
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(DATA)
                zip_path.unlink(missing_ok=True)
                print("Download and extraction complete.")
                return
    except Exception as e:
        print(f"Direct download failed ({e}), checking Kaggle CLI...")

    if not shutil.which("kaggle"):
        sys.exit("The Kaggle CLI is not installed. Run: pip install kaggle\n"
                 "Then create a token at https://www.kaggle.com/settings -> API -> 'Create New Token' and put the\n"
                 r"downloaded kaggle.json in C:\Users\<you>\.kaggle\kaggle.json (Windows) or ~/.kaggle/kaggle.json."
                 "\nOr download the CSV manually from https://www.kaggle.com/datasets/ealaxi/paysim1 into data/ "
                 "and run this again.")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA), "--unzip"], check=True)


def pin_sklearn():
    req = Path("requirements.txt")
    txt = re.sub(r"(?m)^scikit-learn.*$", f"scikit-learn=={sklearn.__version__}", req.read_text())
    req.write_text(txt)
    print(f"requirements.txt pinned to scikit-learn=={sklearn.__version__} (the version the model was trained with)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    csv = Path(a.csv) if a.csv else find_csv()
    if csv is None:
        download()
        csv = find_csv()
    if csv is None or not csv.exists():
        sys.exit("Could not find the PaySim CSV in data/")
    print(f"Training on {csv} (several minutes, needs ~8 GB RAM)...")
    old = joblib.load(fm.PATH)["metrics"] if fm.PATH.exists() else {}
    fm.train(str(csv), out=CANDIDATE)
    new = joblib.load(CANDIDATE)["metrics"]

    print("\n              current model   NEW model")
    for k in ["data_source", "rows", "test_frauds", "pr_auc", "baseline_logreg_pr_auc", "precision", "recall"]:
        print(f"{k:<26}{str(old.get(k, '-')):<16}{new[k]}")
    ok = new["data_source"] == "kaggle_paysim_csv" and new["test_frauds"] >= 50 and new["pr_auc"] >= .5
    if not ok and not a.force:
        sys.exit("\nNot promoted: the new model failed the sanity checks (>=50 test frauds and PR-AUC >= 0.5). Use --force to override.")
    shutil.copy(CANDIDATE, fm.PATH)
    shutil.copy(CANDIDATE.with_name("candidate_metrics.json"), fm.PATH.with_name("bank_model_metrics.json"))
    pin_sklearn()
    print("\nDone. models/bank_model.joblib is now the Kaggle-trained model.\n"
          "Next: git add models requirements.txt && git commit -m 'Kaggle model' && git push  (Render redeploys)")


if __name__ == "__main__":
    main()
