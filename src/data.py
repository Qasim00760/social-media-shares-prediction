"""Loading and cleaning the Online News Popularity dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "online_news_popularity.csv.gz"
TARGET = "shares"
POPULAR_THRESHOLD = 1400  # median shares in the full dataset

CHANNELS = {
    "data_channel_is_lifestyle": "Lifestyle",
    "data_channel_is_entertainment": "Entertainment",
    "data_channel_is_bus": "Business",
    "data_channel_is_socmed": "Social Media",
    "data_channel_is_tech": "Tech",
    "data_channel_is_world": "World",
}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
KEYWORD_COLUMNS = ["kw_min_min", "kw_max_min", "kw_avg_min", "kw_min_max", "kw_max_max", "kw_avg_max",
                   "kw_min_avg", "kw_max_avg", "kw_avg_avg"]

# Not known when an article is published, or not a property of the article.
EXCLUDED = ["url", "timedelta", TARGET]


def load_raw(path=RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the cleaned articles and a log of what was changed."""
    df = raw.copy()
    log = []

    df["publish_date"] = pd.to_datetime(df["url"].str.extract(r"mashable\.com/(\d{4}/\d{2}/\d{2})")[0])
    df["channel"] = "Other"
    for col, name in CHANNELS.items():
        df.loc[df[col] == 1, "channel"] = name
    df["weekday"] = "monday"
    for day in WEEKDAYS:
        df.loc[df[f"weekday_is_{day}"] == 1, "weekday"] = day
    df["weekday"] = df["weekday"].str.capitalize()

    # Ratios of unique and non-stop words must be between 0 and 1.
    bad_ratio = df["n_unique_tokens"] > 1
    log.append(("Article with a unique-word ratio above 1 (impossible), removed", int(bad_ratio.sum())))
    df = df[~bad_ratio].copy()

    # Keyword share statistics use -1 and other negative values for "no data".
    negative = (df[KEYWORD_COLUMNS] < 0)
    log.append(("Negative keyword share values (used as 'no data'), set to missing", int(negative.sum().sum())))
    df[KEYWORD_COLUMNS] = df[KEYWORD_COLUMNS].mask(negative)

    no_text = df["n_tokens_content"] == 0
    df["no_text"] = no_text.astype(int)
    log.append(("Articles with no body text (mostly video posts), kept and flagged", int(no_text.sum())))

    df["log_shares"] = np.log1p(df[TARGET])
    df["popular"] = (df[TARGET] >= POPULAR_THRESHOLD).astype(int)
    df = df.sort_values("publish_date").reset_index(drop=True)
    return df, pd.DataFrame(log, columns=["step", "rows or values"])


def load_clean(path=RAW) -> pd.DataFrame:
    return clean(load_raw(path))[0]


def feature_columns(df: pd.DataFrame) -> list[str]:
    extra = {"publish_date", "channel", "weekday", "log_shares", "popular"}
    return [c for c in df.columns if c not in set(EXCLUDED) | extra]


def time_split(df: pd.DataFrame, train_share: float = 0.8):
    """Older articles for training, the most recent ones for testing."""
    cut = int(len(df) * train_share)
    return df.iloc[:cut], df.iloc[cut:]
