"""
audit.py

Fairness audit for the Adult-Income benchmark.

Outputs in ./audit/:
  • metrics_<clf>.csv     9×4 table: metrics × [sex, race, age_band, overall]
  • roc_<attr>_<clf>.png  ROC curves per group
  • rates_<attr>_<clf>.png TPR/FPR bars per group
  • parity_gaps_<clf>.png  DP & EO gaps bar
  • audit_provenance.json  SHA-256 hashes for reproducibility
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List

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
ADULT_CSV   = Path("adult.csv")
PRED_DIR    = Path("predictions")
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


def pick_best(df: pd.DataFrame) -> str:
    """Select best model by Accuracy → AUC-ROC → F1 → Seconds."""
    auc_col = next(c for c in ("auc_roc", "auc-roc") if c in df.columns)
    ranked = df.sort_values(
    ["accuracy", auc_col, "f1", "seconds"],
    ascending=[False, False, False, True]
    ).reset_index(drop=True)
    best = ranked.loc[0, "classifier"]
    print(f"[INFO] best model → {best}")
    return best


def load_adult() -> pd.DataFrame:
    """Load adult.csv or fetch via Fairlearn fallback."""
    if ADULT_CSV.exists():
        df = pd.read_csv(ADULT_CSV, na_values="?")
    else:
        from fairlearn.datasets import fetch_adult
        df = fetch_adult(as_frame=True).frame
    normalise(df)
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
    data["age_band"] = pd.cut(
        data.age, bins=AGE_BINS, labels=AGE_LABELS, right=False
    )

    def avg_by(attr: str, fn):
        return float(np.mean([
            fn(sub.y_true, sub.y_pred)
            for _, sub in data.groupby(attr, observed=True)
        ]))

    results: Dict[str, Dict[str, float]] = {}

    for name, fn in [
        ("accuracy", accuracy_score),
        ("precision", lambda y, yhat: precision_score(y, yhat, zero_division=0)),
        ("recall", lambda y, yhat: recall_score(y, yhat, zero_division=0)),
        ("f1", lambda y, yhat: f1_score(y, yhat, zero_division=0)),
    ]:
        results[name] = {
            "sex":      avg_by("sex", fn),
            "race":     avg_by("race", fn),
            "age_band": avg_by("age_band", fn),
            "overall":  fn(data.y_true, data.y_pred),
        }

    def auc_group(sub):
        if sub.y_true.nunique() < 2:
            return np.nan
        return roc_auc_score(sub.y_true, sub.y_prob)

    results["auc_roc"] = {
        "sex":      float(np.nanmean([auc_group(sub) for _, sub in data.groupby("sex", observed=True)])),
        "race":     float(np.nanmean([auc_group(sub) for _, sub in data.groupby("race", observed=True)])),
        "age_band": float(np.nanmean([auc_group(sub) for _, sub in data.groupby("age_band", observed=True)])),
        "overall":  roc_auc_score(data.y_true, data.y_prob),
    }

    dp_vals = {}
    eo_vals = {}

    for attr in ("sex", "race", "age_band"):
        dp = demographic_parity_difference(
            y_true=data.y_true, y_pred=data.y_pred, sensitive_features=data[attr]
        )
        eo = equalized_odds_difference(
            y_true=data.y_true, y_pred=data.y_pred, sensitive_features=data[attr]
        )
        dp_vals[attr] = dp
        eo_vals[attr] = eo

    dp_overall = max(dp_vals.values())
    eo_overall = max(eo_vals.values())

    results["dp_gap"] = {
        **dp_vals,
        "overall": dp_overall,
    }
    results["eo_gap"] = {
        **eo_vals,
        "overall": eo_overall,
    }

    return pd.DataFrame(results).T


# ───── Plotting ─────────────────────────────────────────────────────────
def roc_plot(df: pd.DataFrame, attr: str, out: Path) -> None:
    plt.figure(figsize=(6, 4.5))
    for g, sub in df.groupby(attr, observed=True):
        if sub.y_true.nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(sub.y_true, sub.y_prob)
        auc = roc_auc_score(sub.y_true, sub.y_prob)
        plt.plot(fpr, tpr, lw=2, label=f"{g}  AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC by {attr}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def rate_bars(df: pd.DataFrame, attr: str, out: Path) -> None:
    recs: List[Dict] = []
    for g, sub in df.groupby(attr, observed=True):
        tp = ((sub.y_true == 1) & (sub.y_pred == 1)).sum()
        fn = ((sub.y_true == 1) & (sub.y_pred == 0)).sum()
        fp = ((sub.y_true == 0) & (sub.y_pred == 1)).sum()
        tn = ((sub.y_true == 0) & (sub.y_pred == 0)).sum()
        recs.append({
            attr: g,
            "TPR": tp / (tp + fn + 1e-9),
            "FPR": fp / (fp + tn + 1e-9),
        })
    bar = pd.DataFrame(recs).melt(attr, var_name="Rate", value_name="Score")
    plt.figure(figsize=(6, 4))
    sns.barplot(bar, x=attr, y="Score", hue="Rate", edgecolor="black")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.title(f"Error Rates by {attr}")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def parity_bar(gaps: Dict[str, float], out: Path) -> None:
    gdf = (
        pd.Series(gaps)
        .rename_axis("Metric")
        .reset_index(name="Gap")
        .sort_values("Gap", ascending=False)
    )
    plt.figure(figsize=(6, 3))
    sns.barplot(gdf, y="Metric", x="Gap", color="salmon", edgecolor="black")
    plt.axvline(0.05, ls="--", color="grey", label="ε = 0.05")
    plt.xlim(0, 1)
    plt.xlabel("Absolute Gap")
    plt.title("Parity Gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


# ───── Main workflow ─────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(exist_ok=True)

    # pick best model
    best = pick_best(normalise(pd.read_csv(RESULTS_CSV)))

    # load predictions
    preds = normalise(pd.read_csv(PRED_DIR / f"preds_{best}.csv"))

    # merge in sensitive attributes if missing
    missing = SENSITIVE - set(preds.columns)
    if missing:
        adult = load_adult()
        preds = preds.merge(
            adult[list(missing)],
            left_on="id",
            right_index=True,
            how="left",
        )

    # compute & save metrics
    metrics = collect_metrics(preds)
    mpath = OUT_DIR / f"metrics_{best}.csv"
    metrics.to_csv(mpath)
    print(f"[SAVE] {mpath}")

    preds["age_band"] = pd.cut(preds.age, bins=AGE_BINS, labels=AGE_LABELS, right=False)

    for attr in ("sex", "race", "age_band"):
        roc_plot(preds, attr, OUT_DIR / f"roc_{attr}_{best}.png")
        rate_bars(preds, attr, OUT_DIR / f"rates_{attr}_{best}.png")

    gap_vals = {
        f"dp_gap_{attr}": metrics.loc["dp_gap", attr] for attr in ("sex", "race", "age_band")
    }
    gap_vals["dp_gap_overall"] = metrics.loc["dp_gap", "overall"]
    gap_vals.update({
        f"eo_gap_{attr}": metrics.loc["eo_gap", attr] for attr in ("sex", "race", "age_band")
    })
    gap_vals["eo_gap_overall"] = metrics.loc["eo_gap", "overall"]

    parity_bar(gap_vals, OUT_DIR / f"parity_gaps_{best}.png")

    prov = {
        "best_model":   best,
        "preds_sha":    sha8(PRED_DIR / f"preds_{best}.csv"),
        "metrics_sha":  sha8(mpath),
        "script_sha":   sha8(Path(__file__)),
    }
    with open(OUT_DIR / "audit_provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)

    print("✔ Audit complete — see ./audit/")


if __name__ == "__main__":
    main()