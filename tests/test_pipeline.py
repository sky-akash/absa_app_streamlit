"""Pipeline tests with small fake models, so they run without downloading
transformer weights. They check the plumbing, parsing and aggregation logic."""

from pathlib import Path

import pandas as pd
import pytest

from absa import insights as ins
from absa import viz
from absa.analysis import AnalysisConfig, Models, normalize_aspect, normalize_label, parse_aspect_list, run_analysis
from absa.data_io import DataError, load_reviews

ROOT = Path(__file__).resolve().parents[1]
NEG = ("terrible", "poor", "late", "damaged", "cheap", "crash", "overheat", "nightmare", "awful", "weak", "slow", "noisy", "blurry", "missing", "unhelpful", "confusing", "drains", "rejected", "expensive", "overpriced", "disappointed", "cracked", "scratches")
POS = ("amazing", "great", "love", "excellent", "fast", "easy", "friendly", "beautiful", "perfect", "fantastic", "happy", "solid", "durable", "stunning", "good", "impressed", "quick", "smooth", "polite")
VOCAB = ("battery", "screen", "delivery", "customer service", "price", "quality", "camera", "app", "design", "support", "packaging", "setup", "sound quality")


def _polarity(text):
    t = text.lower()
    score = sum(w in t for w in POS) - sum(w in t for w in NEG)
    pos, neg = (0.8, 0.1) if score > 0 else (0.1, 0.8) if score < 0 else (0.2, 0.2)
    return [{"label": "positive", "score": pos}, {"label": "negative", "score": neg}, {"label": "neutral", "score": 1 - pos - neg}]


def fake_sentiment(texts, **_):
    return [_polarity(t) for t in texts]


def fake_absa(inputs, **_):
    return [_polarity(x["text"]) for x in inputs]


def fake_llm(prompts, **_):
    out = []
    for chat in prompts:
        review = chat[-1]["content"].split('Review: "')[1].lower()
        terms = [v for v in VOCAB if v in review] + ["the product", "unicorn horn"]  # generic + hallucinated
        out.append([{"generated_text": ", ".join(terms) or "none"}])
    return out


def fake_zsc(texts, candidate_labels, **_):
    texts = [texts] if isinstance(texts, str) else texts
    res = []
    for t in texts:
        scores = [0.9 if lab.split()[0] in t.lower() else 0.05 for lab in candidate_labels]
        res.append({"sequence": t, "labels": list(candidate_labels), "scores": scores})
    return res if len(res) > 1 else res[0]


MODELS = Models(sentiment=lambda: fake_sentiment, absa=lambda: fake_absa, zero_shot=lambda: fake_zsc, llm=lambda: fake_llm)


@pytest.fixture
def sample():
    return load_reviews("sample_reviews.csv", (ROOT / "sample_data" / "sample_reviews.csv").read_bytes())


def test_load_csv_and_excel(tmp_path, sample):
    assert list(sample.columns) == ["id", "comment"] and len(sample) == 40
    xlsx = tmp_path / "r.xlsx"
    pd.DataFrame({"ID": [1, 1, 2], "Review": ["good", "bad", " "]}).to_excel(xlsx, index=False)
    df = load_reviews("r.xlsx", xlsx.read_bytes())
    assert df["id"].tolist() == ["1", "1_1"]  # blank dropped, duplicate id made unique
    with pytest.raises(DataError):
        load_reviews("r.json", b"{}")


def test_label_and_aspect_normalisation():
    assert normalize_label("5 stars") == "positive" and normalize_label("1 star") == "negative"
    assert normalize_label("3 stars") == "neutral" and normalize_label("NEGATIVE") == "negative"
    assert normalize_aspect("The Batteries") == "battery"
    assert normalize_aspect("my screens") == "screen"
    review = "The battery life is great but the screens are dim."
    assert parse_aspect_list("1. battery life\n2. screens, the product, unicorn horn", review, 5) == ["battery life", "screen"]


@pytest.mark.parametrize("mode", ["auto", "user", "both"])
def test_run_analysis_modes(sample, mode):
    cfg = AnalysisConfig(mode=mode, user_aspects=["price", "delivery", "battery"], group_into_themes=True, extract_needs=True)
    result = run_analysis(sample, MODELS, cfg)
    assert len(result.reviews) == 40
    assert set(result.aspects["source"]) == ({"auto", "user"} if mode == "both" else {mode})
    assert "unicorn horn" not in set(result.aspects["aspect"])  # hallucinations filtered
    assert result.aspects["aspect_sentiment"].isin(["positive", "neutral", "negative"]).all()

    table = ins.aspect_summary(result.aspects, result.reviews)
    assert not table.empty and table["mentions"].is_monotonic_decreasing
    values_table = ins.value_summary(result.values, result.reviews, cfg.value_threshold)
    findings = ins.generate_insights(result.reviews, result.aspects, table, values_table)
    assert findings[0].startswith("**Overall mood:**")

    # Every figure builds without error.
    summary = ins.overall_summary(result.reviews)
    top = table["aspect"].head(10).tolist()
    for fig in (viz.sentiment_donut(summary), viz.polarity_histogram(result.reviews), viz.aspect_frequency(table),
                viz.aspect_sentiment_bars(table), viz.priority_matrix(table), viz.impact_bars(table),
                viz.cooccurrence_heatmap(ins.cooccurrence(result.aspects, top)), viz.values_bars(values_table),
                viz.aspect_by_value_heatmap(result.aspects, result.values, top, cfg.value_threshold)):
        assert fig is not None


def test_user_mode_uses_sentence_context(sample):
    df = pd.DataFrame({"id": ["a"], "comment": ["The delivery was late. The price is great."]})
    result = run_analysis(df, MODELS, AnalysisConfig(mode="user", user_aspects=["delivery", "price"], analyse_values=False))
    got = result.aspects.set_index("aspect")["aspect_sentiment"].to_dict()
    assert got == {"delivery": "negative", "price": "positive"}
