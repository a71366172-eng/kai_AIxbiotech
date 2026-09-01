from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load WDBC and encode malignant as the positive class (1)."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    y = (raw.target == 0).astype(int).rename("malignant")
    return X, y


def split_data(test_size: float = 0.2) -> DataSplit:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(C=2, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows = []
    for name, model in candidate_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=None)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"cv_{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["cv_recall_mean", "cv_roc_auc_mean"], ascending=False
    ).reset_index(drop=True)


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    probability = model.predict_proba(X)[:, 1]
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "precision_malignant": precision_score(y, prediction, zero_division=0),
        "recall_malignant": recall_score(y, prediction, zero_division=0),
        "f1_malignant": f1_score(y, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def logistic_contributions(model: Pipeline, row: pd.DataFrame) -> pd.DataFrame:
    """Return standardized feature contributions to the logistic log-odds."""
    scaler = model.named_steps["scale"]
    estimator = model.named_steps["model"]
    standardized = scaler.transform(row)[0]
    contributions = standardized * estimator.coef_[0]
    result = pd.DataFrame(
        {
            "feature": row.columns,
            "value": row.iloc[0].values,
            "contribution": contributions,
        }
    )
    result["absolute_contribution"] = result["contribution"].abs()
    return result.sort_values("absolute_contribution", ascending=False)


def feature_ranges(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "min": X.min(),
            "q25": X.quantile(0.25),
            "median": X.median(),
            "q75": X.quantile(0.75),
            "max": X.max(),
        }
    )
