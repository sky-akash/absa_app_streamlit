"""Streamlit app: aspect-based sentiment analysis of customer reviews with
Hugging Face transformers.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from absa import insights as ins
from absa import models as mdl
from absa import viz
from absa.analysis import DEFAULT_THEMES, DEFAULT_VALUE_LABELS, AnalysisConfig, Models, run_analysis
from absa.data_io import DataError, load_reviews

SAMPLE_FILE = Path(__file__).parent / "sample_data" / "sample_reviews.csv"

st.set_page_config(page_title="Aspect-Based Sentiment Analysis", page_icon="💬", layout="wide")


# --------------------------------------------------------------------------- cached models

@st.cache_resource(show_spinner="Loading sentiment model…")
def get_sentiment(name: str):
    return mdl.load_text_classifier(name)


@st.cache_resource(show_spinner="Loading aspect-sentiment model…")
def get_absa(name: str):
    return mdl.load_text_classifier(name)


@st.cache_resource(show_spinner="Loading zero-shot model…")
def get_zero_shot(name: str):
    return mdl.load_zero_shot(name)


@st.cache_resource(show_spinner="Loading LLM for aspect extraction…")
def get_llm(name: str):
    return mdl.load_llm(name)


def _split_list(text: str) -> list[str]:
    items = [x.strip() for x in text.replace("\n", ",").split(",")]
    return list(dict.fromkeys(x for x in items if x))


# --------------------------------------------------------------------------- sidebar

st.sidebar.header("1. Data")
uploaded = st.sidebar.file_uploader("Upload reviews (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                    help="Two columns: id and comment.")
use_sample = st.sidebar.toggle("Use the sample dataset", value=uploaded is None)

st.sidebar.header("2. Analysis")
mode_label = st.sidebar.radio(
    "Aspect analysis",
    ["Automatic: let the LLM find aspects", "Custom: analyse my aspects", "Both"],
    help="Automatic extracts aspects from the comments themselves. Custom checks the aspects you list.",
)
mode = {"Automatic": "auto", "Custom": "user", "Both": "both"}[mode_label.split(":")[0]]

user_aspects: list[str] = []
threshold = 0.6
if mode in ("user", "both"):
    user_aspects = _split_list(st.sidebar.text_area(
        "Aspects to analyse (comma separated)", "price, quality, customer service, delivery, ease of use"))
    threshold = st.sidebar.slider("Detection threshold", 0.3, 0.95, 0.6, 0.05,
                                  help="How sure the zero-shot model must be that a sentence talks about an aspect.")

group_themes = False
themes = list(DEFAULT_THEMES)
max_aspects = 6
if mode in ("auto", "both"):
    max_aspects = st.sidebar.slider("Max aspects per review", 1, 10, 6)
    group_themes = st.sidebar.checkbox("Group extracted aspects into themes", value=True,
                                       help="Maps each extracted aspect to a broader theme with zero-shot classification.")
    if group_themes:
        themes = _split_list(st.sidebar.text_area("Themes", ", ".join(DEFAULT_THEMES)))

with st.sidebar.expander("Customer values & needs"):
    analyse_values = st.checkbox("Map reviews to customer values", value=True)
    value_labels = _split_list(st.text_area("Value labels", ", ".join(DEFAULT_VALUE_LABELS)))
    value_threshold = st.slider("Value threshold", 0.3, 0.95, 0.6, 0.05)
    extract_needs = st.checkbox("Extract each customer's underlying need with the LLM (slower)", value=False)

def _pick(label: str, options: dict, preset_key: str, preset: dict) -> str:
    keys = list(options)
    return options[st.selectbox(label, keys, index=keys.index(preset[preset_key]), key=f"{preset_key}_{preset_name}")]


with st.sidebar.expander("Models"):
    preset_name = st.radio("Preset", list(mdl.PRESETS))
    preset = mdl.PRESETS[preset_name]
    sentiment_name = _pick("Overall sentiment", mdl.SENTIMENT_MODELS, "sentiment", preset)
    absa_name = _pick("Aspect sentiment", mdl.ABSA_MODELS, "absa", preset)
    zsc_name = _pick("Zero-shot (custom aspects, values, themes)", mdl.ZERO_SHOT_MODELS, "zero_shot", preset)
    llm_name = _pick("LLM (automatic aspects, needs)", mdl.LLM_MODELS, "llm", preset)
    batch_size = st.select_slider("Batch size", [4, 8, 16, 32], value=16)

max_rows = st.sidebar.number_input("Max reviews to analyse", 10, 100_000, 500, step=50,
                                   help="Transformers on CPU take roughly 0.2 to 2 seconds per review.")


# --------------------------------------------------------------------------- data

st.title("💬 Aspect-Based Sentiment Analysis")
st.caption("Understand what customers talk about, how they feel about each aspect, and what they value, "
           "using pretrained Hugging Face transformers and a small LLM. No training needed.")

df: pd.DataFrame | None = None
try:
    if uploaded is not None and not use_sample:
        df = load_reviews(uploaded.name, uploaded.getvalue())
    elif use_sample:
        df = load_reviews(SAMPLE_FILE.name, SAMPLE_FILE.read_bytes())
except DataError as exc:
    st.error(str(exc))

if df is None:
    st.info("Upload a CSV or Excel file with `id` and `comment` columns, or switch on the sample dataset.")
    st.stop()

if len(df) > max_rows:
    st.warning(f"The file has {len(df):,} reviews; analysing the first {int(max_rows):,}. Raise the limit in the sidebar.")
    df = df.head(int(max_rows))

with st.expander(f"Preview data ({len(df):,} reviews)"):
    st.dataframe(df, width="stretch", hide_index=True)

if mode in ("user", "both") and not user_aspects:
    st.warning("Add at least one aspect to analyse in the sidebar.")
    st.stop()

if st.button("Run analysis", type="primary"):
    config = AnalysisConfig(
        mode=mode, user_aspects=user_aspects, detection_threshold=threshold,
        max_aspects_per_review=max_aspects, group_into_themes=group_themes, themes=themes,
        analyse_values=analyse_values, value_labels=value_labels, value_threshold=value_threshold,
        extract_needs=extract_needs, batch_size=batch_size,
    )
    models = Models(
        sentiment=lambda: get_sentiment(sentiment_name),
        absa=lambda: get_absa(absa_name),
        zero_shot=lambda: get_zero_shot(zsc_name),
        llm=lambda: get_llm(llm_name),
    )
    bar = st.progress(0.0, "Starting…")
    try:
        result = run_analysis(df, models, config, progress=lambda p, msg: bar.progress(min(max(p, 0.0), 1.0), msg))
    except Exception as exc:  # surface model download / memory errors in the UI
        bar.empty()
        st.exception(exc)
        st.stop()
    bar.empty()
    st.session_state["result"] = result

result = st.session_state.get("result")
if result is None:
    st.stop()


# --------------------------------------------------------------------------- results

reviews, aspects, values, cfg = result.reviews, result.aspects, result.values, result.config

controls = st.columns(3)
source_options = {"auto": "Automatic aspects", "user": "My aspects"}
sources = [s for s in ("auto", "user") if s in set(aspects["source"])]
source = controls[0].radio("Aspect set", sources, format_func=source_options.get, horizontal=True) if len(sources) > 1 else (sources[0] if sources else None)
level = "aspect"
if source == "auto" and cfg.group_into_themes:
    level = "theme" if controls[1].radio("Level", ["Aspects", "Themes"], horizontal=True) == "Themes" else "aspect"
min_mentions = controls[2].number_input("Min mentions per aspect", 1, 1000, 2 if len(reviews) >= 30 else 1)

view = aspects[aspects["source"] == source] if source else aspects.iloc[0:0]
if level == "theme":
    view = view.assign(aspect=view["theme"])
table = ins.aspect_summary(view, reviews, min_mentions=int(min_mentions))
values_table = ins.value_summary(values, reviews, cfg.value_threshold) if cfg.analyse_values else None
summary = ins.overall_summary(reviews)

tab_overview, tab_aspects, tab_insights, tab_values, tab_explore = st.tabs(
    ["Overview", "Aspects", "Insights", "Customer values & needs", "Review explorer"])

with tab_overview:
    k = st.columns(5)
    k[0].metric("Reviews", f"{summary['reviews']:,}")
    k[1].metric("Positive", f"{summary['pct_positive']:.0%}")
    k[2].metric("Negative", f"{summary['pct_negative']:.0%}")
    k[3].metric("Net sentiment", f"{summary['net_sentiment']:+.2f}")
    k[4].metric("Distinct aspects", f"{view['aspect'].nunique():,}")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.sentiment_donut(summary), width="stretch", key="chart_1")
    c2.plotly_chart(viz.polarity_histogram(reviews), width="stretch", key="chart_2")
    if not table.empty:
        st.plotly_chart(viz.aspect_sentiment_bars(table), width="stretch", key="chart_3")

with tab_aspects:
    if table.empty:
        st.info("No aspects found with the current settings. Lower the minimum mentions or detection threshold.")
    else:
        top_n = st.slider("Aspects to show", 5, 40, 15)
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.aspect_frequency(table, top_n), width="stretch", key="chart_4")
        c2.plotly_chart(viz.aspect_sentiment_bars(table, top_n), width="stretch", key="chart_5")
        st.plotly_chart(viz.priority_matrix(table.head(top_n)), width="stretch", key="chart_6")
        st.caption("**Core strength**: frequent and loved. **Fix first**: frequent and disliked. "
                   "**Hidden gem**: rare but loved. **Watch list**: rare but disliked.")
        top = table.head(min(top_n, 20))["aspect"].tolist()
        if len(top) > 1:
            st.plotly_chart(viz.cooccurrence_heatmap(ins.cooccurrence(view, top)), width="stretch", key="chart_7")
        if source == "auto" and cfg.group_into_themes and level == "aspect":
            st.plotly_chart(viz.theme_sunburst(aspects[aspects["source"] == "auto"]), width="stretch", key="chart_8")
        st.subheader("Aspect table")
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={
                "share_of_reviews": st.column_config.ProgressColumn("Share of reviews", format="percent", min_value=0, max_value=1),
                "pct_positive": st.column_config.NumberColumn("% positive", format="percent"),
                "pct_negative": st.column_config.NumberColumn("% negative", format="percent"),
                "net_sentiment": st.column_config.NumberColumn("Net sentiment", format="%+.2f"),
                "avg_polarity": st.column_config.NumberColumn("Avg polarity", format="%+.2f"),
                "impact": st.column_config.NumberColumn("Impact on overall", format="%+.2f"),
            },
        )

with tab_insights:
    st.subheader("Key findings")
    for line in ins.generate_insights(reviews, view, table, values_table, min_mentions=int(min_mentions)):
        st.markdown(f"- {line}")
    if not table.empty:
        st.plotly_chart(viz.impact_bars(table), width="stretch", key="chart_9")
        pairs = ins.negative_pairs(view)
        if not pairs.empty:
            st.subheader("Complaints that come together")
            st.dataframe(pairs, width="stretch", hide_index=True)
        st.subheader("What customers say")
        pick = st.selectbox("Aspect", table["aspect"].tolist())
        quotes = view[view["aspect"] == pick]
        qc = st.columns(2)
        for col, label in zip(qc, ["positive", "negative"]):
            col.markdown(f"**{label.title()} mentions**")
            sample = quotes[quotes["aspect_sentiment"] == label].sort_values("aspect_confidence", ascending=False).head(5)
            if sample.empty:
                col.caption("None")
            for ctx in sample["context"]:
                col.markdown(f"> {ctx}")
    mixed = ins.mixed_reviews(reviews, view)
    if len(mixed):
        with st.expander(f"Positive reviews that still raise a problem ({len(mixed)})"):
            st.dataframe(mixed[["id", "comment", "aspects"]], width="stretch", hide_index=True)

with tab_values:
    if values_table is None or values_table.empty:
        st.info("Customer value mapping is turned off. Enable it under **Customer values & needs** in the sidebar.")
    else:
        st.plotly_chart(viz.values_bars(values_table), width="stretch", key="chart_10")
        top = table.head(15)["aspect"].tolist() if not table.empty else []
        fig = viz.aspect_by_value_heatmap(view, values, top, cfg.value_threshold)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", key="chart_11")
        st.dataframe(values_table, width="stretch", hide_index=True,
                     column_config={"share_of_reviews": st.column_config.NumberColumn("Share", format="percent"),
                                    "pct_negative": st.column_config.NumberColumn("% negative", format="percent"),
                                    "avg_polarity": st.column_config.NumberColumn("Avg polarity", format="%+.2f")})
    if "need" in reviews:
        st.subheader("Underlying needs (LLM)")
        st.caption("Needs from negative and neutral reviews point to unmet expectations.")
        unmet = reviews[reviews["sentiment"] != "positive"][["id", "sentiment", "need", "comment"]]
        st.dataframe(unmet, width="stretch", hide_index=True)

with tab_explore:
    f1, f2 = st.columns(2)
    sent_filter = f1.multiselect("Overall sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])
    aspect_filter = f2.multiselect("Mentions aspect", sorted(view["aspect"].unique()))
    shown = reviews[reviews["sentiment"].isin(sent_filter)]
    if aspect_filter:
        shown = shown[shown["id"].isin(view.loc[view["aspect"].isin(aspect_filter), "id"])]
    st.dataframe(shown, width="stretch", hide_index=True,
                 column_config={"polarity": st.column_config.NumberColumn(format="%+.2f"),
                                "sentiment_confidence": st.column_config.NumberColumn(format="%.2f")})
    st.subheader("Aspect mentions")
    st.dataframe(view[view["id"].isin(shown["id"])], width="stretch", hide_index=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download review results (CSV)", reviews.to_csv(index=False), "review_sentiment.csv", "text/csv")
    d2.download_button("Download aspect mentions (CSV)", aspects.to_csv(index=False), "aspect_mentions.csv", "text/csv")
    d3.download_button("Download aspect summary (CSV)", table.to_csv(index=False), "aspect_summary.csv", "text/csv")
