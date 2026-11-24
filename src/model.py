"""Popularity models for online news shared on social media.

Training uses older articles and testing the most recent fifth (by publish date), the
way a model would be used in practice. `timedelta` (days between publishing and data
collection) and the URL are not used: neither is a property of the article when it is
published.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET, feature_columns, time_split

RANDOM_STATE = 42
NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"
sns.set_theme(style="whitegrid")


def classifiers() -> dict:
    return {
        "Always predict the majority class": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(C=0.1, max_iter=3000)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, max_features=0.3, n_jobs=-1, random_state=RANDOM_STATE)),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                                            random_state=RANDOM_STATE),
    }


def final_classifier():
    return classifiers()["Gradient Boosting"]


def _cls_metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {"ROC_AUC": roc_auc_score(y, proba) if len(set(proba)) > 1 else 0.5,
            "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "F1": f1_score(y, pred)}


def compare_classifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    feats = feature_columns(df)
    train, test = time_split(df)
    rows, probas = [], {}
    for name, model in classifiers().items():
        model.fit(train[feats], train["popular"])
        p = model.predict_proba(test[feats])[:, 1]
        probas[name] = p
        rows.append({"model": name, **_cls_metrics(test["popular"], p)})
    return pd.DataFrame(rows).round(3), probas


def regression_result(df: pd.DataFrame) -> pd.DataFrame:
    feats = feature_columns(df)
    train, test = time_split(df)
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=RANDOM_STATE)
    model.fit(train[feats], train["log_shares"])
    pred = model.predict(test[feats])
    baseline = np.full(len(test), train["log_shares"].median())
    within_2x = (np.abs(pred - test["log_shares"]) <= np.log(2)).mean()
    return pd.DataFrame([
        {"model": "Predict the training median", "R2_log_shares": r2_score(test["log_shares"], baseline),
         "MAE_log_shares": mean_absolute_error(test["log_shares"], baseline),
         "within_2x_%": (np.abs(baseline - test["log_shares"]) <= np.log(2)).mean() * 100},
        {"model": "Gradient Boosting", "R2_log_shares": r2_score(test["log_shares"], pred),
         "MAE_log_shares": mean_absolute_error(test["log_shares"], pred), "within_2x_%": within_2x * 100},
    ]).round(3)


def validation_check(df: pd.DataFrame) -> pd.DataFrame:
    """Random 5-fold cross-validation vs training on older articles and testing on newer ones."""
    feats = feature_columns(df)
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    p = cross_val_predict(final_classifier(), df[feats], df["popular"], cv=cv, method="predict_proba")[:, 1]
    train, test = time_split(df)
    pt = final_classifier().fit(train[feats], train["popular"]).predict_proba(test[feats])[:, 1]
    return pd.DataFrame([
        {"validation": "Random 5-fold cross-validation", "ROC_AUC": roc_auc_score(df["popular"], p)},
        {"validation": "Train on older 80%, test on newest 20%", "ROC_AUC": roc_auc_score(test["popular"], pt)},
    ]).round(3)


def feature_importance(df: pd.DataFrame, sample: int = 4000) -> pd.DataFrame:
    feats = feature_columns(df)
    train, test = time_split(df)
    model = final_classifier().fit(train[feats], train["popular"])
    s = test.sample(min(sample, len(test)), random_state=RANDOM_STATE)
    r = permutation_importance(model, s[feats], s["popular"], n_repeats=5, random_state=RANDOM_STATE,
                               scoring="roc_auc")
    return (pd.DataFrame({"feature": feats, "AUC_drop": r.importances_mean})
            .sort_values("AUC_drop", ascending=False).round(4).reset_index(drop=True))


def lift_table(y, proba, bins: int = 5) -> pd.DataFrame:
    """Popularity rate in each fifth of the test articles, ranked by predicted probability."""
    q = pd.qcut(pd.Series(proba).rank(method="first"), bins, labels=[f"{i}" for i in range(bins, 0, -1)])
    t = pd.DataFrame({"group": q.values, "popular": np.asarray(y)})
    out = t.groupby("group", observed=True)["popular"].agg(articles="size", popular_rate="mean").reset_index()
    out = out.iloc[::-1].reset_index(drop=True)
    out["group"] = ["Top 20% by score", "20-40%", "40-60%", "60-80%", "Bottom 20%"][:len(out)]
    out["popular_rate"] = (out["popular_rate"] * 100).round(1)
    return out


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_roc(y, probas: dict):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    colors = {"Logistic Regression": "#7f9bb5", "Random Forest": BLUE, "Gradient Boosting": RED}
    for name, p in probas.items():
        if name in colors:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=colors[name], linewidth=2, label=f"{name} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right")
    return _finish(ax, "ROC curves on the newest 20% of articles")


def plot_lift(lift: pd.DataFrame, base_rate: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift["group"], lift["popular_rate"], color=[RED, BLUE, BLUE, BLUE, "#9ecae1"])
    for i, v in enumerate(lift["popular_rate"]):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10)
    ax.axhline(base_rate * 100, color="#555", linestyle="--", linewidth=1, label=f"All test articles ({base_rate * 100:.0f}%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Actually popular (%)")
    ax.legend(frameon=False)
    return _finish(ax, "Test articles ranked by the model's score")


def plot_feature_importance(imp: pd.DataFrame, top: int = 12):
    t = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(t["feature"], t["AUC_drop"], color=NAVY)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    return _finish(ax, f"Top {top} features (Gradient Boosting, test articles)")
