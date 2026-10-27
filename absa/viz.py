"""Plotly figures for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SENTIMENT_COLORS = {"positive": "#2a78d6", "neutral": "#a8a7a0", "negative": "#e34948"}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
DIVERGING = [[0.0, "#e34948"], [0.5, "#d9d8d2"], [1.0, "#2a78d6"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#0d366b"]]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=90, b=10),
        title=dict(y=0.98, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        bargap=0.25,
    )
    return fig


def sentiment_donut(summary: dict) -> go.Figure:
    labels = ["positive", "neutral", "negative"]
    values = [summary[k] for k in labels]
    fig = go.Figure(go.Pie(
        labels=[l.title() for l in labels], values=values, hole=0.6, sort=False,
        marker=dict(colors=[SENTIMENT_COLORS[l] for l in labels], line=dict(color="white", width=2)),
        textinfo="percent", hovertemplate="%{label}: %{value} reviews (%{percent})<extra></extra>",
    ))
    fig.update_layout(title="Overall review sentiment",
                      annotations=[dict(text=f"{summary['net_sentiment']:+.2f}<br>net", showarrow=False, font_size=18)])
    return _layout(fig, 340)


def polarity_histogram(reviews: pd.DataFrame) -> go.Figure:
    fig = px.histogram(reviews, x="polarity", nbins=20, color="sentiment",
                       color_discrete_map=SENTIMENT_COLORS, category_orders={"sentiment": SENTIMENT_ORDER},
                       labels={"polarity": "Polarity (negative -1 to +1 positive)", "count": "Reviews"},
                       title="How strongly do customers feel?")
    fig.update_layout(yaxis_title="Reviews")
    return _layout(fig, 340)


def aspect_frequency(table: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = table.head(top_n).iloc[::-1]
    fig = px.bar(data, x="mentions", y="aspect", orientation="h", title=f"Most mentioned aspects (top {len(data)})",
                 labels={"mentions": "Reviews mentioning", "aspect": ""},
                 hover_data={"share_of_reviews": ":.0%"}, color_discrete_sequence=[CATEGORICAL[0]])
    return _layout(fig, max(320, 28 * len(data) + 80))


def aspect_sentiment_bars(table: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Diverging stacked bars: negative to the left, positive to the right."""
    data = table.head(top_n).sort_values("net_sentiment")
    fig = go.Figure()
    fig.add_bar(y=data["aspect"], x=-data["negative"], name="Negative", orientation="h",
                marker_color=SENTIMENT_COLORS["negative"], customdata=data["negative"],
                hovertemplate="%{y}: %{customdata} negative<extra></extra>")
    fig.add_bar(y=data["aspect"], x=data["neutral"], name="Neutral", orientation="h",
                marker_color=SENTIMENT_COLORS["neutral"], hovertemplate="%{y}: %{x} neutral<extra></extra>")
    fig.add_bar(y=data["aspect"], x=data["positive"], name="Positive", orientation="h",
                marker_color=SENTIMENT_COLORS["positive"], hovertemplate="%{y}: %{x} positive<extra></extra>")
    fig.update_layout(barmode="relative", title="Sentiment per aspect (sorted by net sentiment)",
                      xaxis_title="Mentions (negative ← | → positive / neutral)")
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    return _layout(fig, max(320, 28 * len(data) + 80))


def priority_matrix(table: pd.DataFrame) -> go.Figure:
    """Importance (how often mentioned) vs. performance (net sentiment)."""
    fig = px.scatter(
        table, x="share_of_reviews", y="net_sentiment", size="mentions", color="net_sentiment",
        color_continuous_scale=DIVERGING, range_color=[-1, 1], text="aspect",
        hover_data={"mentions": True, "quadrant": True, "share_of_reviews": ":.0%", "net_sentiment": ":+.2f"},
        labels={"share_of_reviews": "Importance: share of reviews mentioning", "net_sentiment": "Performance: net sentiment"},
        title="Priority matrix: what to fix, what to protect",
    )
    fig.update_traces(textposition="top center", marker=dict(line=dict(color="white", width=2), sizemin=8))
    x_mid = table["share_of_reviews"].median() if len(table) else 0
    fig.add_hline(y=0, line_dash="dot", line_color="#898781")
    fig.add_vline(x=x_mid, line_dash="dot", line_color="#898781")
    x_max = max(table["share_of_reviews"].max() * 1.1, x_mid * 2, 0.01) if len(table) else 1
    for x, y, label in [(x_max, 1.05, "Core strength"), (x_max, -1.05, "Fix first"), (0, 1.05, "Hidden gem"), (0, -1.05, "Watch list")]:
        fig.add_annotation(x=x, y=y, text=f"<b>{label}</b>", showarrow=False, font=dict(color="#52514e"),
                           xanchor="right" if x else "left")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(range=[-1.15, 1.15])
    fig.update_coloraxes(showscale=False)
    return _layout(fig, 520)


def impact_bars(table: pd.DataFrame, top_n: int = 15) -> go.Figure:
    data = table.dropna(subset=["impact"]).head(top_n).sort_values("impact")
    colors = [SENTIMENT_COLORS["positive"] if v >= 0 else SENTIMENT_COLORS["negative"] for v in data["impact"]]
    fig = go.Figure(go.Bar(x=data["impact"], y=data["aspect"], orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:+.2f} overall polarity<extra></extra>"))
    fig.update_layout(title="Satisfaction drivers: effect of mentioning an aspect on overall sentiment",
                      xaxis_title="Change in average overall polarity when mentioned")
    return _layout(fig, max(320, 28 * len(data) + 80))


def cooccurrence_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=SEQUENTIAL, aspect="auto",
                    labels=dict(color="Reviews"), title="Which aspects are discussed together?")
    return _layout(fig, max(380, 30 * len(matrix) + 120))


def theme_sunburst(aspects: pd.DataFrame) -> go.Figure:
    data = aspects.groupby(["theme", "aspect"], as_index=False).agg(
        mentions=("id", "nunique"), polarity=("aspect_polarity", "mean"))
    fig = px.sunburst(data, path=["theme", "aspect"], values="mentions", color="polarity",
                      color_continuous_scale=DIVERGING, range_color=[-1, 1],
                      title="Aspect themes (colour = average aspect sentiment)")
    return _layout(fig, 520)


def values_bars(values_table: pd.DataFrame) -> go.Figure:
    data = values_table.sort_values("share_of_reviews")
    fig = px.bar(data, x="share_of_reviews", y="value", orientation="h", color="avg_polarity",
                 color_continuous_scale=DIVERGING, range_color=[-1, 1],
                 labels={"share_of_reviews": "Share of reviews", "value": "", "avg_polarity": "Avg sentiment"},
                 hover_data={"reviews": True, "pct_negative": ":.0%"},
                 title="What customers value (colour = how satisfied they are)")
    fig.update_xaxes(tickformat=".0%")
    return _layout(fig, max(320, 34 * len(data) + 80))


def aspect_by_value_heatmap(aspects: pd.DataFrame, values: pd.DataFrame, top: list[str], threshold: float) -> go.Figure | None:
    labels = [c for c in values.columns if c != "id"]
    if not labels or aspects.empty or not top:
        return None
    flags = values.set_index("id")[labels] >= threshold
    sub = aspects[aspects["aspect"].isin(top)].drop_duplicates(["id", "aspect"])
    sub = sub.join(flags, on="id")
    matrix = sub.groupby("aspect")[labels].sum().reindex(top).fillna(0).astype(int)
    fig = px.imshow(matrix.T, text_auto=True, color_continuous_scale=SEQUENTIAL, aspect="auto",
                    labels=dict(color="Reviews", x="Aspect", y="Customer value"),
                    title="Which aspects carry which customer values?")
    return _layout(fig, max(380, 34 * len(labels) + 140))
