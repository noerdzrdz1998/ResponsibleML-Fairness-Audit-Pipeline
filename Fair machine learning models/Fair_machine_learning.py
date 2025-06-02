"""
Fair_machine_learning.py
========================

Train a single XGBoost model on the Adult Census Income dataset, wrapped in Fairlearn’s ExponentiatedGradient 
reduction to enforce Equalized Odds across sex, race and age. Records both predictive performance and fairness metrics.

Outputs
-------
adult_results.csv
    Accuracy, Precision, Recall, F1, AUC-ROC and elapsed time for each outer fold.
predictions/preds_XGB.csv
    id, fold, y_true, y_pred, y_prob, sex, race, age (one row per test sample).
models/model_XGB.joblib
    Final ExponentiatedGradient-wrapped XGBoost model fitted on the full dataset.

Notes
-----
* Looks for `adult.csv` in the script folder; if not found, fetches the data via Fairlearn’s `fetch_adult`.
* Imputes missing values (median for numerics, most frequent for categoricals) and one-hot encodes all categorical features.
* Nested cross-validation: 5 outer folds, each with 3-fold inner Optuna tuning (30 trials, TPE sampler, seed=42).
* Uses Optuna to tune XGBoost hyperparameters, then applies ExponentiatedGradient with the Equalized Odds constraint.
"""
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds

from xgboost import XGBClassifier


###############################################################################
# Hyper-parameter spaces
###############################################################################
def _spaces() -> Dict[str, Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Any]] = {}
    if XGBClassifier is not None:
        spaces["XGB"] = {
            "model": XGBClassifier(
                eval_metric="logloss",
                objective="binary:logistic",
                random_state=42,
            ),
            "grid": {
                "n_estimators": [100, 200, 300, 400, 500, 600, 800, 1000, 1200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                "max_depth": [3, 4, 5, 6, 8, 10, 12, 15, 20],
                "min_child_weight": [1, 3, 5, 7, 9, 11],
                "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.3, 0.5, 1, 5, 10],
                "reg_alpha": [0, 0.01, 0.05, 0.1, 1],
                "reg_lambda": [0.5, 1.0, 1.5, 2.0, 3.0],
            },
        }
    return spaces

###############################################################################
# Data utilities
###############################################################################
def _impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    if not df[num_cols].empty:
        df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    if not df[cat_cols].empty:
        df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    return df


def _load_adult() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    csv_path = Path("adult.csv")
    if csv_path.exists():
        raw = pd.read_csv(csv_path, na_values="?")
        raw = _impute(raw)
        if "income" in raw.columns:
            y = (raw["income"].str.strip() == ">50K").astype(int)
            X_raw = raw.drop(columns=["income"])
        else:
            X_raw = raw.iloc[:, :-1]
            y = raw.iloc[:, -1].astype(int)
    else:
        from fairlearn.datasets import fetch_adult

        data = fetch_adult(as_frame=True)
        X_raw = _impute(data.data)
        y = (data.target == "> 50K").astype(int)

    X_enc = pd.get_dummies(X_raw, drop_first=True)
    return X_enc, X_raw, y


###############################################################################
# Bayesian tuner
###############################################################################
def _bayes(model, grid, X: np.ndarray, y: np.ndarray, trials: int = 30):
    if not grid:
        clone(model).fit(X, y)
        return model

    space_size = np.prod([len(v) for v in grid.values()]) if grid else 1
    trials = min(trials, int(space_size))

    def objective(trial: optuna.trial.Trial) -> float:
        params = {k: trial.suggest_categorical(k, v) for k, v in grid.items()}
        clf = clone(model).set_params(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        acc: List[float] = []
        for tr, va in cv.split(X, y):
            clf.fit(X[tr], y[tr])
            acc.append(accuracy_score(y[va], clf.predict(X[va])))
        return float(np.mean(acc))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = clone(model).set_params(**study.best_params)
    best.fit(X, y)
    return best


###############################################################################
# Nested CV evaluation with fairness reduction
###############################################################################
def _nested(
    name: str,
    info: Dict[str, Any],
    X_enc: pd.DataFrame,
    X_raw: pd.DataFrame,
    y: pd.Series,
    sens_cols: List[str],
):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics, pred_rows = [], []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X_enc, y)):
        # 1) Tune and train base learner
        base = _bayes(
            info["model"],
            info["grid"],
            X_enc.values[tr_idx],
            y.values[tr_idx],
        )

        # 2) Wrap in ExponentiatedGradient with Equalized Odds
        expgrad = ExponentiatedGradient(
            estimator=base,
            constraints=EqualizedOdds(),
            eps=0.005,
            max_iter=200,
        )
        expgrad.fit(
            X_enc.values[tr_idx],
            y.values[tr_idx],
            sensitive_features=X_raw.iloc[tr_idx][sens_cols],
        )

        # 3) Predict on test fold
        y_pred = expgrad.predict(X_enc.values[te_idx])
        pmf = expgrad._pmf_predict(X_enc.values[te_idx])
        y_score = pmf[:, 1]
        
        # 4) Compute metrics
        fold_metrics.append(
            [
                accuracy_score(y.values[te_idx], y_pred),
                precision_score(y.values[te_idx], y_pred, zero_division=0),
                recall_score(y.values[te_idx], y_pred, zero_division=0),
                f1_score(y.values[te_idx], y_pred, zero_division=0),
                roc_auc_score(y.values[te_idx], y_score)
                if not np.isnan(y_score).all()
                else np.nan,
            ]
        )

        # 5) Store per-sample predictions
        for i, idx in enumerate(te_idx):
            row = {
                "id": int(idx),
                "fold": fold,
                "y_true": int(y.values[idx]),
                "y_pred": int(y_pred[i]),
                "y_prob": float(y_score[i]),
            }
            for col in sens_cols:
                row[col] = X_raw.iloc[idx][col]
            pred_rows.append(row)

    return np.nanmean(fold_metrics, axis=0), pred_rows


###############################################################################
# Main
###############################################################################
def main() -> None:
    base_dir = Path(__file__).parent
    X_enc, X_raw, y = _load_adult()
    sens_cols = [c for c in X_raw.columns if c.lower() in {"sex", "race", "age"}]

    pred_dir = base_dir / "predictions"
    model_dir = base_dir / "models"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, info in _spaces().items():
        print(f"⏳ {name} (fair ExponentiatedGradient)")
        start = time.time()
        metrics, rows = _nested(name, info, X_enc, X_raw, y, sens_cols)
        elapsed = round(time.time() - start, 1)

        acc, prec, rec, f1, auc = metrics
        summary_rows.append(
            {
                "Classifier": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "AUC_ROC": auc,
                "Seconds": elapsed,
            }
        )

        pd.DataFrame(rows).to_csv(pred_dir / f"preds_{name}.csv", index=False)

        # Refit on full data and wrap in fairness reduction
        base_full = _bayes(
            info["model"],
            info["grid"],
            X_enc.values,
            y.values
        )
        expgrad_full = ExponentiatedGradient(
            estimator=base_full,
            constraints=EqualizedOdds(),
        )
        expgrad_full.fit(
            X_enc.values,
            y.values,
            sensitive_features=X_raw[sens_cols],
        )
        joblib.dump(expgrad_full, model_dir / f"model_{name}.joblib")

    pd.DataFrame(summary_rows).to_csv(base_dir / "adult_results.csv", index=False)

if __name__ == "__main__":
    main()