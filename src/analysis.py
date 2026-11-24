"""Summaries and charts of social media shares."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from src.data import POPULAR_THRESHOLD, TARGET

NAVY, BLUE, RED = "#0f2a44", "#3d6d99", "#b2182b"
WEEK_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.set_theme(style="whitegrid")


def shares_by(df: pd.DataFrame, col: str, bins=None, labels=None) -> pd.DataFrame:
    key = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True) if bins is not None else df[col]
    g = df.groupby(key, observed=True)
    out = pd.DataFrame({"articles": g.size(), "median_shares": g[TARGET].median(),
                        "popular_%": (g["popular"].mean() * 100).round(1)})
    out.index.name = col
    return out.reset_index()


def weekend_test(df: pd.DataFrame) -> dict:
    wk = df.loc[df["is_weekend"] == 1, TARGET]
    wd = df.loc[df["is_weekend"] == 0, TARGET]
    return {"median_weekend": wk.median(), "median_weekday": wd.median(), "p_value": mannwhitneyu(wk, wd).pvalue}


def rank_correlations(df: pd.DataFrame, cols: list[str], top: int = 15) -> pd.DataFrame:
    corr = df[cols + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    out = pd.DataFrame({"feature": corr.index, "spearman": corr.values})
    return out.reindex(out["spearman"].abs().sort_values(ascending=False).index).head(top).round(3).reset_index(drop=True)


# ---------------------------------------------------------------- charts


def _finish(ax, title):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    sns.despine(ax=ax)
    ax.figure.tight_layout()
    return ax.figure


def plot_shares_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[TARGET].clip(upper=20000), bins=60, color=BLUE, ax=axes[0])
    axes[0].set_title(f"Shares (capped at 20k for display, skewness {df[TARGET].skew():.0f})", loc="left",
                      fontweight="bold", fontsize=11)
    axes[0].set_xlabel("Shares")
    sns.histplot(df["log_shares"], bins=60, color=NAVY, ax=axes[1])
    axes[1].axvline(np.log1p(POPULAR_THRESHOLD), color=RED, linestyle="--", linewidth=2,
                    label=f"Popular: {POPULAR_THRESHOLD:,}+ shares (median)")
    axes[1].set_title("log(1 + shares)", loc="left", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("log(1 + shares)")
    axes[1].legend(frameon=False)
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def _popular_bars(ax, t, col, title):
    overall = 50
    ax.bar(t[col].astype(str), t["popular_%"], color=[RED if v >= 60 else (BLUE if v >= 45 else "#9ecae1") for v in t["popular_%"]])
    for i, (v, m) in enumerate(zip(t["popular_%"], t["median_shares"])):
        ax.text(i, v + 1, f"{v:.0f}%\n{m:,.0f}", ha="center", fontsize=8)
    ax.axhline(overall, color="#555", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(t["popular_%"].max() * 1.25, 60))
    ax.set_ylabel("Popular articles (%)")
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    sns.despine(ax=ax)


def plot_popularity_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    specs = [
        ("channel", None, None, "Channel"),
        ("weekday", None, None, "Day published"),
        ("num_imgs", [0, 0.5, 1.5, 5, 10, 200], ["0", "1", "2-5", "6-10", "11+"], "Number of images"),
        ("num_videos", [0, 0.5, 1.5, 3, 100], ["0", "1", "2-3", "4+"], "Number of videos"),
        ("n_tokens_content", [-1, 0, 300, 600, 1000, 10000], ["no text", "1-300", "301-600", "601-1000", "1000+"],
         "Words in the article"),
        ("kw_avg_avg", [0, 2000, 2500, 3000, 4000, 50000], ["<2k", "2-2.5k", "2.5-3k", "3-4k", "4k+"],
         "Average shares of the article's keywords"),
    ]
    for ax, (col, bins, labels, title) in zip(axes.flat, specs):
        t = shares_by(df, col, bins, labels)
        if col == "weekday":
            t = t.set_index("weekday").loc[WEEK_ORDER].reset_index()
        if col == "channel":
            t = t.sort_values("popular_%", ascending=False)
        _popular_bars(ax, t, col, title)
    fig.suptitle(f"Share of popular articles ({POPULAR_THRESHOLD:,}+ shares) by group; labels show % popular and median shares",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_monthly(df: pd.DataFrame):
    m = df.set_index("publish_date").resample("MS").agg(articles=(TARGET, "size"), median_shares=(TARGET, "median"),
                                                          popular=("popular", "mean"))
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.bar(m.index, m["articles"], width=20, color="#cfe2f3", label="Articles published")
    ax1.set_ylabel("Articles per month")
    ax2 = ax1.twinx()
    ax2.plot(m.index, m["popular"] * 100, color=RED, marker="o", linewidth=2, label="Popular (%)")
    ax2.set_ylabel("Popular (%)")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(lines, ["Articles published", "Popular (%)"], frameon=False, loc="upper left")
    return _finish(ax1, "Articles and share of popular articles by month")
