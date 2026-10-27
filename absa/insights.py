"""Aggregations and plain-language insights computed from the analysis results."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

QUADRANTS = {
    (True, True): "Core strength",
    (True, False): "Fix first",
    (False, True): "Hidden gem",
    (False, False): "Watch list",
}


def overall_summary(reviews: pd.DataFrame) -> dict:
    n = len(reviews)
    counts = reviews["sentiment"].value_counts()
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    return {
        "reviews": n,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "pct_positive": pos / n if n else 0.0,
        "pct_negative": neg / n if n else 0.0,
        "net_sentiment": (pos - neg) / n if n else 0.0,
        "avg_polarity": float(reviews["polarity"].mean()) if n else 0.0,
    }


def aspect_summary(aspects: pd.DataFrame, reviews: pd.DataFrame, key: str = "aspect", min_mentions: int = 1) -> pd.DataFrame:
    """One row per aspect: how often it is mentioned, how customers feel about
    it, and how much it moves overall review sentiment."""
    cols = ["aspect", "mentions", "share_of_reviews", "positive", "neutral", "negative",
            "pct_positive", "pct_negative", "net_sentiment", "avg_polarity", "impact", "quadrant"]
    if aspects.empty:
        return pd.DataFrame(columns=cols)

    per_review = aspects.groupby([key, "id"], as_index=False)["aspect_polarity"].mean()
    per_review["aspect_sentiment"] = np.select(
        [per_review["aspect_polarity"] > 0.2, per_review["aspect_polarity"] < -0.2],
        ["positive", "negative"], "neutral",
    )
    n_reviews = len(reviews)
    overall = reviews.set_index("id")["polarity"]

    rows = []
    for name, grp in per_review.groupby(key):
        mentions = len(grp)
        if mentions < min_mentions:
            continue
        counts = grp["aspect_sentiment"].value_counts()
        pos, neu, neg = (int(counts.get(k, 0)) for k in ("positive", "neutral", "negative"))
        mentioned = overall.index.isin(grp["id"])
        with_aspect = overall[mentioned].mean()
        without = overall[~mentioned].mean() if (~mentioned).any() else np.nan
        rows.append({
            "aspect": name,
            "mentions": mentions,
            "share_of_reviews": mentions / n_reviews if n_reviews else 0.0,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "pct_positive": pos / mentions,
            "pct_negative": neg / mentions,
            "net_sentiment": (pos - neg) / mentions,
            "avg_polarity": float(grp["aspect_polarity"].mean()),
            # Difference in average overall review polarity when the aspect is
            # mentioned vs. not: a simple, model-free "driver" signal.
            "impact": float(with_aspect - without) if not np.isnan(without) else np.nan,
        })
    table = pd.DataFrame(rows, columns=cols[:-1])
    if table.empty:
        return pd.DataFrame(columns=cols)
    freq_cut = table["share_of_reviews"].median()
    table["quadrant"] = [
        QUADRANTS[(share >= freq_cut, net >= 0)]
        for share, net in zip(table["share_of_reviews"], table["net_sentiment"])
    ]
    return table.sort_values("mentions", ascending=False).reset_index(drop=True)


def cooccurrence(aspects: pd.DataFrame, top: list[str], key: str = "aspect") -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=top, columns=top, dtype=int)
    if aspects.empty:
        return matrix
    sets = aspects[aspects[key].isin(top)].groupby("id")[key].apply(lambda s: sorted(set(s)))
    for items in sets:
        for a in items:
            matrix.loc[a, a] += 1
        for a, b in combinations(items, 2):
            matrix.loc[a, b] += 1
            matrix.loc[b, a] += 1
    return matrix


def negative_pairs(aspects: pd.DataFrame, key: str = "aspect", top_n: int = 8) -> pd.DataFrame:
    """Aspects that customers complain about together in the same review."""
    neg = aspects[aspects["aspect_sentiment"] == "negative"]
    sets = neg.groupby("id")[key].apply(lambda s: sorted(set(s)))
    counter: dict[tuple[str, str], int] = {}
    for items in sets:
        for pair in combinations(items, 2):
            counter[pair] = counter.get(pair, 0) + 1
    rows = [{"pair": f"{a} + {b}", "reviews": c} for (a, b), c in counter.items() if c >= 2]
    return pd.DataFrame(rows, columns=["pair", "reviews"]).sort_values("reviews", ascending=False).head(top_n)


def value_summary(values: pd.DataFrame, reviews: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cols = ["value", "share_of_reviews", "reviews", "avg_polarity", "pct_negative"]
    labels = [c for c in values.columns if c != "id"]
    if not labels:
        return pd.DataFrame(columns=cols)
    merged = values.merge(reviews[["id", "polarity", "sentiment"]], on="id")
    rows = []
    for label in labels:
        hit = merged[merged[label] >= threshold]
        rows.append({
            "value": label,
            "share_of_reviews": len(hit) / len(merged) if len(merged) else 0.0,
            "reviews": len(hit),
            "avg_polarity": float(hit["polarity"].mean()) if len(hit) else np.nan,
            "pct_negative": float((hit["sentiment"] == "negative").mean()) if len(hit) else np.nan,
        })
    return pd.DataFrame(rows, columns=cols).sort_values("share_of_reviews", ascending=False).reset_index(drop=True)


def mixed_reviews(reviews: pd.DataFrame, aspects: pd.DataFrame) -> pd.DataFrame:
    """Positive reviews that still contain a negative aspect: hidden friction."""
    if aspects.empty:
        return reviews.iloc[0:0]
    neg_ids = set(aspects.loc[aspects["aspect_sentiment"] == "negative", "id"])
    return reviews[(reviews["sentiment"] == "positive") & reviews["id"].isin(neg_ids)]


def _pct(x: float) -> str:
    return f"{x:.0%}"


def generate_insights(
    reviews: pd.DataFrame,
    aspects: pd.DataFrame,
    table: pd.DataFrame,
    values_table: pd.DataFrame | None = None,
    min_mentions: int = 3,
) -> list[str]:
    """Plain-language findings, most important first."""
    out: list[str] = []
    summary = overall_summary(reviews)
    out.append(
        f"**Overall mood:** {_pct(summary['pct_positive'])} of {summary['reviews']} reviews are positive "
        f"and {_pct(summary['pct_negative'])} negative (net sentiment {summary['net_sentiment']:+.2f})."
    )

    solid = table[table["mentions"] >= min_mentions] if not table.empty else table
    if solid.empty:
        solid = table
    if not solid.empty:
        most = solid.iloc[0]
        out.append(
            f"**Most discussed:** *{most['aspect']}* comes up in {_pct(most['share_of_reviews'])} of reviews, "
            f"so it is what customers notice most. Its net sentiment is {most['net_sentiment']:+.2f}."
        )
        best = solid.sort_values(["net_sentiment", "mentions"], ascending=False).iloc[0]
        if best["net_sentiment"] > 0:
            out.append(
                f"**Top strength:** *{best['aspect']}* ({_pct(best['pct_positive'])} positive across "
                f"{best['mentions']} mentions). Lean on it in messaging."
            )
        pain = solid.sort_values(["negative", "net_sentiment"], ascending=[False, True]).iloc[0]
        if pain["negative"] > 0:
            out.append(
                f"**Biggest pain point:** *{pain['aspect']}* draws {pain['negative']} negative mentions "
                f"({_pct(pain['pct_negative'])} of its mentions). This is the first thing to fix."
            )
        drivers = solid.dropna(subset=["impact"]).sort_values("impact")
        if not drivers.empty and drivers.iloc[0]["impact"] < -0.1:
            d = drivers.iloc[0]
            out.append(
                f"**Strongest drag on satisfaction:** reviews mentioning *{d['aspect']}* are "
                f"{abs(d['impact']):.2f} points less positive overall than those that don't."
            )
        if not drivers.empty and drivers.iloc[-1]["impact"] > 0.1:
            d = drivers.iloc[-1]
            out.append(
                f"**Strongest delight driver:** mentioning *{d['aspect']}* lifts overall review "
                f"polarity by {d['impact']:.2f} points."
            )

    if not table.empty:
        hidden = table[(table["quadrant"] == "Watch list") & (table["pct_negative"] >= 0.5) & (table["mentions"] >= 2)]
        if not hidden.empty:
            names = ", ".join(f"*{a}*" for a in hidden.sort_values("pct_negative", ascending=False)["aspect"].head(4))
            out.append(
                f"**Hidden pain points:** {names} are mentioned rarely but mostly negatively. They are easy "
                "to miss in averages and may signal emerging issues."
            )
        gems = table[(table["quadrant"] == "Hidden gem") & (table["pct_positive"] >= 0.7) & (table["mentions"] >= 2)]
        if not gems.empty:
            names = ", ".join(f"*{a}*" for a in gems.sort_values("pct_positive", ascending=False)["aspect"].head(4))
            out.append(f"**Hidden gems:** {names} delight the few who mention them; consider promoting them.")

    mixed = mixed_reviews(reviews, aspects)
    if len(mixed):
        top_friction = (
            aspects[aspects["id"].isin(mixed["id"]) & (aspects["aspect_sentiment"] == "negative")]["aspect"]
            .value_counts().head(3).index.tolist()
        )
        out.append(
            f"**Friction among happy customers:** {len(mixed)} positive reviews still criticise something, "
            f"most often {', '.join(f'*{a}*' for a in top_friction)}. Fixing these protects loyal customers."
        )

    pairs = negative_pairs(aspects) if not aspects.empty else pd.DataFrame()
    if not pairs.empty:
        p = pairs.iloc[0]
        out.append(f"**Linked complaints:** *{p['pair']}* are criticised together in {p['reviews']} reviews, which suggests a shared root cause.")

    if values_table is not None and not values_table.empty:
        top_vals = values_table[values_table["reviews"] > 0].head(3)
        if not top_vals.empty:
            names = ", ".join(f"*{v}* ({_pct(s)})" for v, s in zip(top_vals["value"], top_vals["share_of_reviews"]))
            out.append(f"**What customers value most:** {names}.")
        risky = values_table[(values_table["reviews"] >= 2)].sort_values("pct_negative", ascending=False)
        if not risky.empty and risky.iloc[0]["pct_negative"] >= 0.4:
            r = risky.iloc[0]
            out.append(
                f"**Unmet need:** customers who care about *{r['value']}* are the least satisfied "
                f"({_pct(r['pct_negative'])} negative). This expectation is not being met."
            )
    return out
