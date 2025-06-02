"""
audit.py

Fairness audit for the ExponentiatedGradient‐wrapped XGBoost model
evaluated on the Adult Census Income dataset.

Outputs in ./audit/:
  • metrics_xgb.csv      subgroup and overall metrics table
  • roc_<attr>_xgb.png   ROC curves by sensitive attribute
  • rates_<attr>_xgb.png TPR/FPR bar charts by sensitive attribute
  • parity_gaps_xgb.png  DP & EO gap bar chart
  • audit_provenance.json SHA-256 hashes for reproducibility
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)

# ───── Configuration ─────────────────────────────────────────────────────
RESULTS_CSV = Path("adult_results.csv")
PRED_FILE   = Path("predictions") / "preds_XGB.csv"
OUT_DIR     = Path("audit")

# raw column names
SENSITIVE   = {"sex", "race", "age"}
AGE_BINS    = [17, 25, 35, 45, 55, 65, 100]
AGE_LABELS  = ["17–24", "25–34", "35–44", "45–54", "55–64", "65–99"]

sns.set_theme(style="whitegrid", palette="Set2")


# ───── Helpers ───────────────────────────────────────────────────────────
def sha8(path: Path) -> str:
    """Return first 8 hex chars of SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip whitespace from column names."""
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_adult() -> pd.DataFrame:
    """Load adult.csv or fetch via Fairlearn, with imputation."""
    ADULT_CSV = Path("adult.csv")
    if ADULT_CSV.exists():
        df = pd.read_csv(ADULT_CSV, na_values="?")
    else:
        from fairlearn.datasets import fetch_adult
        df = fetch_adult(as_frame=True).frame
    df = normalise(df)
    num = df.select_dtypes("number").columns
    cat = df.select_dtypes(exclude="number").columns
    if len(num):
        df[num] = df[num].fillna(df[num].median())
    if len(cat):
        df[cat] = df[cat].fillna(df[cat].mode().iloc[0])
    return df


# ───── Metrics collector ─────────────────────────────────────────────────
def collect_metrics(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["age_band"] = pd.cut(data.age, bins=AGE_BINS, labels=AGE_LABELS, right=False)

    def avg_by(col, fn):
        return float(np.mean([fn(sub.y_true, sub.y_pred)
                              for _, sub in data.groupby(col, observed=True)]))

    results = {}
    for name, fn in [
        ("accuracy", accuracy_score),
        ("precision", lambda y, yhat: precision_score(y, yhat, zero_division=0)),
        ("recall", lambda y, yhat: recall_score(y, yhat, zero_division=0)),
        ("f1", lambda y, yhat: f1_score(y, yhat, zero_division=0)),
    ]:
        results[name] = {
            "sex": avg_by("sex", fn),
            "race": avg_by("race", fn),
            "age_band": avg_by("age_band", fn),
            "overall": fn(data.y_true, data.y_pred),
        }

    def group_auc(sub):
        if sub.y_true.nunique() < 2:
            return np.nan
        return roc_auc_score(sub.y_true, sub.y_prob)

    results["auc_roc"] = {
        "sex":      float(np.nanmean([group_auc(sub) for _, sub in data.groupby("sex", observed=True)])),
        "race":     float(np.nanmean([group_auc(sub) for _, sub in data.groupby("race", observed=True)])),
        "age_band": float(np.nanmean([group_auc(sub) for _, sub in data.groupby("age_band", observed=True)])),
        "overall":  roc_auc_score(data.y_true, data.y_prob),
    }

    dp_vals = {}
    eo_vals = {}
    for attr in ("sex", "race", "age_band"):
        dp_vals[attr] = demographic_parity_difference(
            y_true=data.y_true, y_pred=data.y_pred, sensitive_features=data[attr]
        )
        eo_vals[attr] = equalized_odds_difference(
            y_true=data.y_true, y_pred=data.y_pred, sensitive_features=data[attr]
        )

    results["dp_gap"] = {**dp_vals, "overall": max(dp_vals.values())}
    results["eo_gap"] = {**eo_vals, "overall": max(eo_vals.values())}

    return pd.DataFrame(results).T


# ───── Plotting ─────────────────────────────────────────────────────────
def roc_plot(df: pd.DataFrame, attr: str, out: Path):
    plt.figure(figsize=(6, 4.5))
    for grp, sub in df.groupby(attr, observed=True):
        if sub.y_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub.y_true, sub.y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{grp}  AUC={roc_auc_score(sub.y_true, sub.y_prob):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curves by {attr}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def rate_bars(df: pd.DataFrame, attr: str, out: Path):
    recs = []
    for grp, sub in df.groupby(attr, observed=True):
        tp = ((sub.y_true == 1) & (sub.y_pred == 1)).sum()
        fn = ((sub.y_true == 1) & (sub.y_pred == 0)).sum()
        fp = ((sub.y_true == 0) & (sub.y_pred == 1)).sum()
        tn = ((sub.y_true == 0) & (sub.y_pred == 0)).sum()
        recs.append({
            attr: grp,
            "TPR": tp / (tp + fn + 1e-9),
            "FPR": fp / (fp + tn + 1e-9),
        })
    df_rates = pd.DataFrame(recs).melt(attr, var_name="Rate", value_name="Score")
    plt.figure(figsize=(6, 4))
    sns.barplot(df_rates, x=attr, y="Score", hue="Rate", edgecolor="black")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title(f"TPR and FPR by {attr}")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def parity_bar(gaps: dict, out: Path):
    df_gap = pd.Series(gaps).rename_axis("Metric").reset_index(name="Gap")
    df_gap = df_gap.sort_values("Gap", ascending=False)
    plt.figure(figsize=(6, 3))
    sns.barplot(df_gap, y="Metric", x="Gap", edgecolor="black")
    plt.axvline(0.05, ls="--", color="grey", label="ε = 0.05")
    plt.xlim(0, 1)
    plt.xlabel("Absolute gap")
    plt.title("Demographic & EO gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


# ───── Main workflow ─────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)

    # load predictions for XGBoost + ExponentiatedGradient
    preds = normalise(pd.read_csv(PRED_FILE))

    # ensure sensitive cols present
    missing = SENSITIVE - set(preds.columns)
    if missing:
        adult = load_adult()
        preds = preds.merge(
            adult[list(missing)],
            left_on="id",
            right_index=True,
            how="left",
        )

    # compute and save metrics
    metrics = collect_metrics(preds)
    mpath = OUT_DIR / "metrics_xgb_fair.csv"
    metrics.to_csv(mpath)
    print(f"[SAVE] {mpath}")

    preds["age_band"] = pd.cut(preds.age, bins=AGE_BINS, labels=AGE_LABELS, right=False)

    for attr in ("sex", "race", "age_band"):
        roc_plot(preds, attr, OUT_DIR / f"roc_{attr}_xgb_fair.png")
        rate_bars(preds, attr, OUT_DIR / f"rates_{attr}_xgb_fair.png")

    gaps = {
        f"dp_gap_{a}": metrics.loc["dp_gap", a] for a in ("sex", "race", "age_band")
    }
    gaps["dp_gap_overall"] = metrics.loc["dp_gap", "overall"]
    gaps.update({
        f"eo_gap_{a}": metrics.loc["eo_gap", a] for a in ("sex", "race", "age_band")
    })
    gaps["eo_gap_overall"] = metrics.loc["eo_gap", "overall"]

    parity_bar(gaps, OUT_DIR / "parity_gaps_xgb_fair.png")

    prov = {
        "preds_sha": sha8(PRED_FILE),
        "metrics_sha": sha8(mpath),
        "script_sha": sha8(Path(__file__)),
    }
    with open(OUT_DIR / "audit_provenance_fair.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    print("✔ Audit complete — see ./audit/")


if __name__ == "__main__":
    main()
