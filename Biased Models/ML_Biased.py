"""
ML_Biased.py
=======================

Train a collection of classifiers on the Adult Census Income data set,
recording both prediction quality and fairness-related information.

Outputs
-------
adult_results.csv
    Mean Accuracy, Precision, Recall, F1, AUC-ROC over an outer 5-fold CV.
predictions/preds_<clf>.csv
    id, fold, y_true, y_pred, y_prob, sex, race, age  (one file per model).
models/model_<clf>.joblib
    Final estimator fitted on the full data after hyper-parameter tuning.
Notes
-----
* Uses a local ``adult.csv`` if present, otherwise ``fairlearn.datasets.fetch_adult``.
* Optuna performs 30 Bayesian trials (TPE, seed=42) with an inner 3-fold CV.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


###############################################################################
# Hyper-parameter spaces
###############################################################################
def _spaces() -> Dict[str, Dict[str, Any]]:
    log_C_values = np.logspace(-4, 3, 15).tolist()

    skl_major, skl_minor = map(int, sklearn.__version__.split(".")[:2])
    dt_criteria = ["gini", "entropy"] + (["log_loss"] if (skl_major, skl_minor) >= (1, 1) else [])

    spaces: Dict[str, Dict[str, Any]] = {
        "LogReg": {
            "model": LogisticRegression(max_iter=100, random_state=42),
            "grid": {
                "solver": ["liblinear", "saga"],
                "penalty": ["l1", "l2"],
                "class_weight": [None, "balanced"],
                "C": log_C_values,
            },
        },
        "RF": {
            "model": RandomForestClassifier(random_state=42),
            "grid": {
                "n_estimators": list(range(100, 1001, 100)),
                "max_depth": [None, 5, 10, 20, 30, 40, 60, 80],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
            },
        },
        "ET": {
            "model": ExtraTreesClassifier(random_state=42),
            "grid": {
                "n_estimators": list(range(100, 1001, 100)),
                "max_depth": [None, 5, 10, 20, 30, 40, 60, 80],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "grid": {
                "n_neighbors": list(range(1, 51)),
                "weights": ["uniform", "distance"],
                "p": [1, 2], 
                "metric": ["minkowski", "manhattan"], 
                "leaf_size": list(range(10, 61, 10)),
            },
        },
        "HistGB": {
            "model": HistGradientBoostingClassifier(random_state=42, max_iter=100),
            "grid": {
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5],
                "max_depth": [None, 3, 5, 7, 9, 12, 15],
                "l2_regularization": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
                "min_samples_leaf": [1, 5, 10, 20, 30, 50],
            },
        },
        "SGD": {
            "model": SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
            "grid": {
                "loss": ["hinge", "log_loss", "modified_huber"],
                "alpha": np.logspace(-6, -1, 6).tolist(),
                "penalty": ["l2", "l1", "elasticnet"],
                "l1_ratio": [0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0],
                "learning_rate": ["optimal", "invscaling", "adaptive", "constant"],
                "eta0": np.logspace(-4, -2, 5).tolist(),
            },
        },
        "Ada": {
            "model": AdaBoostClassifier(random_state=42),
            "grid": {
                "n_estimators": [50, 100, 150, 200, 300, 400, 500, 600],
                "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
                "algorithm": ["SAMME", "SAMME.R"],
            },
        },
        "GNB": {"model": GaussianNB(), "grid": {
        "var_smoothing": np.logspace(-11, -7, 5),}},
        "BNB": {
            "model": BernoulliNB(),
            "grid": {"alpha": np.logspace(-3, 0, 9).tolist()},
        },
        "LDA": {
            "model": LinearDiscriminantAnalysis(),
            "grid": {"solver": ["svd"]},
        },
        "QDA": {
            "model": QuadraticDiscriminantAnalysis(),
            "grid": {"reg_param": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]},
        },
        "DT": {
            "model": DecisionTreeClassifier(random_state=42),
            "grid": {
                "max_depth": [None, 2, 5, 10, 20, 30, 40, 60, 80],
                "min_samples_split": [2, 5, 10, 20, 30],
                "min_samples_leaf": [1, 2, 4, 6, 8, 10],
                "criterion": dt_criteria,
                "ccp_alpha": [0.0, 0.0001, 0.001, 0.01],
            },
        },
        "Bag": {
            "model": BaggingClassifier(random_state=42),
            "grid": {
                "n_estimators": [10, 20, 50, 100, 200, 300, 500],
                "max_samples": [0.5, 0.6, 0.8, 1.0],
                "max_features": [0.5, 0.6, 0.8, 1.0],
                "bootstrap": [True, False],
            },
        },
    }

    if XGBClassifier is not None:
        spaces["XGB"] = {
            "model": XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
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
# Nested CV evaluation
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
        model = _bayes(
            info["model"],
            info["grid"],
            X_enc.values[tr_idx],
            y.values[tr_idx],
        )
        y_pred = model.predict(X_enc.values[te_idx])
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_enc.values[te_idx])[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_enc.values[te_idx])
        else:
            y_score = np.full_like(y_pred, np.nan, float)

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
    
    pred_dir  = base_dir / "predictions"
    model_dir = base_dir / "models"
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, info in _spaces().items():
        print(f"⏳ {name}")
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

        final_model = _bayes(
            info["model"],
            info["grid"],
            X_enc.values,
            y.values,
        )
        joblib.dump(final_model, model_dir / f"model_{name}.joblib")
        
    pd.DataFrame(summary_rows).to_csv(base_dir / "adult_results.csv", index=False)

if __name__ == "__main__":
    main()