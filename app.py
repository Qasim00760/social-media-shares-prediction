"""Streamlit app: will an article be popular on social media?

Run locally:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import WEEK_ORDER, plot_monthly, plot_popularity_overview, plot_shares_distribution, shares_by, weekend_test
from src.data import CHANNELS, POPULAR_THRESHOLD, TARGET, WEEKDAYS, clean, feature_columns, load_raw, time_split
from src.model import final_classifier

RESULTS = pd.DataFrame(
    [
        ["Always predict the majority class", 0.500, 0.479, 0.479, 1.000],
        ["Logistic Regression", 0.714, 0.658, 0.636, 0.671],
        ["Random Forest", 0.733, 0.670, 0.642, 0.707],
        ["Gradient Boosting", 0.736, 0.669, 0.643, 0.698],
    ],
    columns=["model", "ROC-AUC", "accuracy", "precision", "recall"],
)
LIFT = pd.DataFrame(
    [["Top 20% by score", 76.4], ["20-40%", 61.8], ["40-60%", 46.9], ["60-80%", 35.4], ["Bottom 20%", 19.2]],
    columns=["test articles ranked by score", "actually popular (%)"],
)

st.set_page_config(page_title="Social Media Shares Prediction", layout="wide")


@st.cache_data
def get_data():
    return clean(load_raw())


@st.cache_resource
def get_model(_df):
    feats = feature_columns(_df)
    train, test = time_split(_df)
    model = final_classifier().fit(train[feats], train["popular"])
    test_scores = model.predict_proba(test[feats])[:, 1]
    return model, feats, np.sort(test_scores)


df, cleaning_log = get_data()
with st.spinner("Training the model (first load only)..."):
    model, feats, test_scores = get_model(df)

st.title("Social Media Shares Prediction")
st.write(
    f"Will a news article be popular on social media? Analysis of {len(df):,} Mashable articles (2013 to 2014) "
    f"and a model that estimates the chance an article reaches {POPULAR_THRESHOLD:,}+ shares, the median, "
    "using only what is known when it is published."
)

tab_predict, tab_explore, tab_model, tab_data = st.tabs(["Score an article", "What gets shared", "Model", "Data"])

# ---------------------------------------------------------------- scoring
with tab_predict:
    st.subheader("Chance of 1,400+ shares")
    c1, c2, c3 = st.columns(3)
    channel = c1.selectbox("Channel", ["Social Media", "Tech", "Lifestyle", "Business", "Entertainment", "World", "Other"])
    weekday = c1.selectbox("Day published", WEEK_ORDER, index=2)
    title_words = c1.slider("Words in the title", 2, 20, 10)
    content_words = c2.slider("Words in the article", 0, 3000, 500, step=50)
    images = c2.slider("Images", 0, 50, 1)
    videos = c2.slider("Videos", 0, 20, 0)
    links = c2.slider("Links", 0, 100, 10)
    keywords = c3.slider("Keywords", 1, 10, 7)
    kw_avg = c3.slider("Average past shares of the article's keywords", 500, 10000, 2800, step=100,
                       help="kw_avg_avg in the data: how many shares earlier articles with the same keywords got on average.")
    ref_avg = c3.number_input("Average shares of linked Mashable articles (0 if none)", 0, 50000, 2000, step=100)
    subjectivity = c3.slider("Subjectivity of the text (0 objective, 1 opinionated)", 0.0, 1.0, 0.45, step=0.05)

    # Start from the typical article in the same channel, then apply the inputs.
    same = df[df["channel"] == channel]
    row = same[feats].median(numeric_only=True).to_frame().T
    for col, name in CHANNELS.items():
        row[col] = int(name == channel)
    for day in WEEKDAYS:
        row[f"weekday_is_{day}"] = int(day == weekday.lower())
    row["is_weekend"] = int(weekday in ("Saturday", "Sunday"))
    row["n_tokens_title"] = title_words
    row["n_tokens_content"] = content_words
    row["no_text"] = int(content_words == 0)
    row["num_imgs"] = images
    row["num_videos"] = videos
    row["num_hrefs"] = links
    row["num_keywords"] = keywords
    row["kw_avg_avg"] = kw_avg
    for col in ["self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_sharess"]:
        row[col] = ref_avg
    row["num_self_hrefs"] = 0 if ref_avg == 0 else max(1, int(row["num_self_hrefs"].iloc[0]))
    row["global_subjectivity"] = subjectivity
    row = row[feats]

    p = float(model.predict_proba(row)[0, 1])
    rank = (np.searchsorted(test_scores, p) / len(test_scores)) * 100
    m1, m2 = st.columns(2)
    m1.metric("Chance of 1,400+ shares", f"{p * 100:.0f}%")
    m2.metric("Ranking among recent articles", f"higher than {rank:.0f}%")
    st.caption(
        "Inputs not shown here (topic mix, sentiment details, word statistics) are set to the median for the chosen "
        "channel. The model ranks articles well but cannot promise shares: among the 20% of recent articles it scored "
        "highest, 76% were popular, against 19% for the lowest 20%."
    )

# ---------------------------------------------------------------- explore
with tab_explore:
    w = weekend_test(df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Articles", f"{len(df):,}")
    k2.metric("Median shares", f"{df[TARGET].median():,.0f}")
    k3.metric("Weekend median", f"{w['median_weekend']:,.0f}")
    k4.metric("Weekday median", f"{w['median_weekday']:,.0f}")
    st.pyplot(plot_popularity_overview(df))
    left, right = st.columns(2)
    left.dataframe(shares_by(df, "channel").sort_values("popular_%", ascending=False), hide_index=True, width="stretch")
    right.dataframe(shares_by(df, "weekday").set_index("weekday").loc[WEEK_ORDER].reset_index(), hide_index=True,
                    width="stretch")
    st.pyplot(plot_shares_distribution(df))
    st.pyplot(plot_monthly(df))

# ---------------------------------------------------------------- model
with tab_model:
    st.markdown(
        "Every model is trained on the oldest 80% of articles (January 2013 to August 2014) and tested on the newest "
        "20% (August to December 2014), the way it would be used for new articles."
    )
    st.dataframe(RESULTS, hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.dataframe(LIFT, hide_index=True, width="stretch")
    right.markdown(
        "- **Gradient Boosting** is the final model (ROC-AUC 0.736). Random 5-fold cross-validation gives the same "
        "score (0.737), so the result is stable over time.\n"
        "- Predicting the exact number of shares is much harder: R2 of 0.15 on log(shares).\n"
        "- The model relies most on the keywords' past shares, the shares of linked Mashable articles, and whether "
        "the article is published at the weekend."
    )
    st.subheader("Why this project uses a different dataset")
    st.write(
        "The first version used a social media dataset in which likes, shares, comments and impressions were "
        "independent random numbers, and engagement rate was exactly (likes + shares + comments) / impressions. "
        "Its 96% accuracy came from giving the model those counts. With only information available before posting, "
        "no model did better than chance (ROC-AUC 0.50), so this version uses real articles with real share counts."
    )

# ---------------------------------------------------------------- data
with tab_data:
    st.markdown(
        "Source: [Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity), UCI Machine "
        "Learning Repository (Fernandes, Vinagre, Cortez and Sernadela, 2015). Mashable owns the articles; the dataset "
        "holds statistics about them, not their text."
    )
    st.dataframe(cleaning_log, hide_index=True, width="stretch")
    st.markdown(
        "- `url` and `timedelta` (days between publishing and data collection) are not model inputs.\n"
        "- Keyword share columns use -1 and other negative values when there is no data; these are treated as missing."
    )
    st.dataframe(df.drop(columns=["log_shares"]).head(500), hide_index=True, width="stretch")
