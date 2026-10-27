"""Core analysis pipeline: overall sentiment, aspect extraction/detection and
aspect-level sentiment, all done with pretrained Hugging Face transformers.

Model objects are passed in (rather than created here) so the Streamlit app
can cache them and tests can swap in light-weight fakes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

ProgressFn = Callable[[float, str], None]

DEFAULT_VALUE_LABELS = [
    "price and value for money",
    "product quality",
    "customer service and support",
    "delivery and speed",
    "ease of use and convenience",
    "reliability and durability",
    "design and appearance",
    "trust and safety",
]

DEFAULT_THEMES = [
    "quality",
    "price",
    "customer service",
    "delivery",
    "usability",
    "performance",
    "design",
    "packaging",
]

ASPECT_HYPOTHESIS = "This customer review talks about {}."
VALUE_HYPOTHESIS = "The customer cares about {}."
THEME_HYPOTHESIS = "This is related to {}."

_GENERIC_ASPECTS = {
    "product", "products", "item", "items", "thing", "things", "it", "everything",
    "nothing", "none", "stuff", "experience", "overall", "review", "na", "n a",
    "aspect", "aspects", "one", "all", "time", "lot", "way",
}
_DETERMINERS = {"the", "a", "an", "my", "our", "their", "its", "this", "that", "these", "those", "your", "his", "her"}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")


@dataclass
class AnalysisConfig:
    mode: str = "auto"  # "auto", "user" or "both"
    user_aspects: list[str] = field(default_factory=list)
    detection_threshold: float = 0.6
    max_aspects_per_review: int = 6
    group_into_themes: bool = False
    themes: list[str] = field(default_factory=lambda: list(DEFAULT_THEMES))
    analyse_values: bool = True
    value_labels: list[str] = field(default_factory=lambda: list(DEFAULT_VALUE_LABELS))
    value_threshold: float = 0.6
    extract_needs: bool = False
    batch_size: int = 16


@dataclass
class Models:
    """Lazily supplies the transformer pipelines needed by the analysis."""

    sentiment: Callable[[], object]
    absa: Callable[[], object]
    zero_shot: Callable[[], object]
    llm: Callable[[], object]


@dataclass
class AnalysisResult:
    reviews: pd.DataFrame  # one row per review
    aspects: pd.DataFrame  # one row per (review, aspect) mention
    values: pd.DataFrame  # one row per review, one column per value label
    config: AnalysisConfig


# --------------------------------------------------------------------------- helpers

def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or [text.strip()]


def normalize_label(label: str) -> str:
    lab = label.lower()
    stars = re.match(r"(\d)\s*stars?", lab)  # e.g. nlptown "1 star" .. "5 stars"
    if stars:
        n = int(stars.group(1))
        return "negative" if n <= 2 else "positive" if n >= 4 else "neutral"
    if lab.startswith("pos") or lab == "label_2":
        return "positive"
    if lab.startswith("neg") or lab == "label_0":
        return "negative"
    return "neutral"


def _scores_to_polarity(scores: Sequence[dict]) -> tuple[str, float, float]:
    """Turn a list of {label, score} into (label, confidence, polarity in [-1, 1])."""
    probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in scores:
        probs[normalize_label(item["label"])] += float(item["score"])
    label = max(probs, key=probs.get)
    return label, probs[label], probs["positive"] - probs["negative"]


def _as_list(output) -> list:
    """Pipelines return a dict for one input and a list for many; normalise."""
    if isinstance(output, dict):
        return [output]
    return list(output)


def normalize_aspect(term: str) -> str:
    term = term.lower().strip()
    term = re.sub(r"[^a-z0-9\s\-&']", " ", term)
    term = re.sub(r"'s\b", "", term)
    words = [w for w in term.replace("-", " ").split() if w]
    while words and words[0] in _DETERMINERS:
        words = words[1:]
    if not words:
        return ""
    last = words[-1]
    if len(last) > 4 and last.endswith("ies"):
        last = last[:-3] + "y"
    elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is", "ics")):
        last = last[:-1]
    words[-1] = last
    return " ".join(words)


def _is_grounded(term: str, text_lower: str) -> bool:
    """True when every word of the aspect (or its stem) appears in the review.

    This keeps the LLM honest: aspects it invents but the customer never
    mentioned are dropped.
    """
    for word in term.split():
        stem = word[:-1] if len(word) > 4 else word
        if stem not in text_lower:
            return False
    return True


def parse_aspect_list(raw: str, review: str, limit: int) -> list[str]:
    raw = raw.strip()
    if not raw or raw.lower().startswith("none"):
        return []
    raw = raw.split("\n\n")[0]
    pieces = re.split(r"[,;\n]|\band\b", raw)
    text_lower = review.lower()
    seen: list[str] = []
    for piece in pieces:
        piece = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", piece).strip(" .\"'`:")
        term = normalize_aspect(piece)
        if (
            not term
            or term in _GENERIC_ASPECTS
            or len(term) < 3
            or len(term.split()) > 4
            or term in seen
            or not _is_grounded(term, text_lower)
        ):
            continue
        seen.append(term)
        if len(seen) >= limit:
            break
    return seen


def _context_for_term(term: str, review: str) -> str:
    """Sentences of the review that mention the aspect term (whole review if none)."""
    sentences = split_sentences(review)
    hits = [s for s in sentences if _is_grounded(term, s.lower())]
    return " ".join(hits) if hits else review


def _generated_text(output) -> str:
    item = output[0] if isinstance(output, list) else output
    text = item.get("generated_text", "") if isinstance(item, dict) else item
    if isinstance(text, list):  # chat format: list of messages
        text = text[-1].get("content", "") if text else ""
    return str(text)


def _chunks(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def _noop(_: float, __: str) -> None:
    return None


# --------------------------------------------------------------------------- model steps

def overall_sentiment(clf, texts: Sequence[str], batch_size: int = 16, progress: ProgressFn = _noop) -> pd.DataFrame:
    rows = []
    for start, batch in _chunks(list(texts), batch_size):
        outputs = clf(list(batch), top_k=None, truncation=True, max_length=512, batch_size=batch_size)
        for out in outputs:
            rows.append(_scores_to_polarity(_as_list(out)))
        progress(min(1.0, (start + len(batch)) / max(len(texts), 1)), "Overall sentiment")
    return pd.DataFrame(rows, columns=["sentiment", "sentiment_confidence", "polarity"])


EXTRACTION_PROMPT = (
    "Extract the aspects a customer comments on in the review below. Aspects are "
    "specific product features, service elements or parts of the experience "
    "(for example: battery life, delivery, staff, price, screen). Use short "
    "lowercase noun phrases of 1-3 words copied from the review wording. Reply "
    "with only a comma-separated list, or 'none' if no aspect is mentioned.\n\n"
    "Review: \"{review}\"\nAspects:"
)


def _chat(prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise assistant that analyses customer reviews."},
        {"role": "user", "content": prompt},
    ]


def extract_aspects_llm(
    generator,
    texts: Sequence[str],
    max_aspects: int = 6,
    batch_size: int = 8,
    progress: ProgressFn = _noop,
) -> list[list[str]]:
    results: list[list[str]] = []
    for start, batch in _chunks(list(texts), batch_size):
        prompts = [_chat(EXTRACTION_PROMPT.format(review=t[:1500])) for t in batch]
        outputs = generator(
            prompts,
            max_new_tokens=48,
            do_sample=False,
            return_full_text=False,
            batch_size=batch_size,
        )
        for review, out in zip(batch, outputs):
            results.append(parse_aspect_list(_generated_text(out), review, max_aspects))
        progress(min(1.0, (start + len(batch)) / max(len(texts), 1)), "Extracting aspects with the LLM")
    return results


def detect_aspects_zero_shot(
    zsc,
    texts: Sequence[str],
    aspects: Sequence[str],
    threshold: float = 0.6,
    batch_size: int = 16,
    progress: ProgressFn = _noop,
) -> list[dict[str, tuple[float, str]]]:
    """For each review return {aspect: (best score, supporting sentences)} for
    the user-given aspects the review talks about.

    Detection runs per sentence so a mixed review ("food was great, service
    was slow") gives each aspect only its own sentence as context.
    """
    sentence_index: list[tuple[int, str]] = []
    for i, text in enumerate(texts):
        for sentence in split_sentences(text):
            sentence_index.append((i, sentence))

    found: list[dict[str, list]] = [dict() for _ in texts]
    sentences = [s for _, s in sentence_index]
    for start, batch in _chunks(sentences, batch_size):
        outputs = _as_list(
            zsc(
                list(batch),
                candidate_labels=list(aspects),
                multi_label=True,
                hypothesis_template=ASPECT_HYPOTHESIS,
                batch_size=batch_size,
            )
        )
        for offset, out in enumerate(outputs):
            review_idx, sentence = sentence_index[start + offset]
            for label, score in zip(out["labels"], out["scores"]):
                if score >= threshold:
                    entry = found[review_idx].setdefault(label, [0.0, []])
                    entry[0] = max(entry[0], float(score))
                    entry[1].append(sentence)
        progress(min(1.0, (start + len(batch)) / max(len(sentences), 1)), "Detecting your aspects")

    return [{a: (v[0], " ".join(v[1])) for a, v in d.items()} for d in found]


def aspect_sentiment(
    absa_clf,
    pairs: Sequence[tuple[str, str]],
    batch_size: int = 16,
    progress: ProgressFn = _noop,
) -> list[tuple[str, float, float]]:
    """Sentiment towards each aspect in its context, as (label, confidence, polarity)."""
    results = []
    for start, batch in _chunks(list(pairs), batch_size):
        inputs = [{"text": ctx[:2000], "text_pair": aspect} for ctx, aspect in batch]
        outputs = absa_clf(inputs, top_k=None, truncation=True, batch_size=batch_size)
        if len(inputs) == 1 and outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        for out in outputs:
            results.append(_scores_to_polarity(_as_list(out)))
        progress(min(1.0, (start + len(batch)) / max(len(pairs), 1)), "Scoring aspect sentiment")
    return results


def classify_values(
    zsc,
    texts: Sequence[str],
    labels: Sequence[str],
    batch_size: int = 16,
    progress: ProgressFn = _noop,
) -> pd.DataFrame:
    rows = []
    for start, batch in _chunks(list(texts), batch_size):
        outputs = _as_list(
            zsc(
                list(batch),
                candidate_labels=list(labels),
                multi_label=True,
                hypothesis_template=VALUE_HYPOTHESIS,
                batch_size=batch_size,
            )
        )
        for out in outputs:
            rows.append(dict(zip(out["labels"], map(float, out["scores"]))))
        progress(min(1.0, (start + len(batch)) / max(len(texts), 1)), "Mapping customer values")
    return pd.DataFrame(rows, columns=list(labels)).fillna(0.0)


def group_terms_into_themes(zsc, terms: Sequence[str], themes: Sequence[str], min_score: float = 0.3) -> dict[str, str]:
    if not terms:
        return {}
    outputs = _as_list(zsc(list(terms), candidate_labels=list(themes), hypothesis_template=THEME_HYPOTHESIS))
    mapping = {}
    for term, out in zip(terms, outputs):
        mapping[term] = out["labels"][0] if out["scores"][0] >= min_score else "other"
    return mapping


NEEDS_PROMPT = (
    "Read this customer review and state, in one short sentence starting with "
    "'Wants', the underlying need or expectation of the customer.\n\n"
    "Review: \"{review}\"\nNeed:"
)


def extract_needs_llm(generator, texts: Sequence[str], batch_size: int = 8, progress: ProgressFn = _noop) -> list[str]:
    needs = []
    for start, batch in _chunks(list(texts), batch_size):
        prompts = [_chat(NEEDS_PROMPT.format(review=t[:1500])) for t in batch]
        outputs = generator(prompts, max_new_tokens=32, do_sample=False, return_full_text=False, batch_size=batch_size)
        for out in outputs:
            need = _generated_text(out).strip().split("\n")[0].strip(" \"'")
            needs.append(need)
        progress(min(1.0, (start + len(batch)) / max(len(texts), 1)), "Extracting customer needs")
    return needs


# --------------------------------------------------------------------------- orchestration

def run_analysis(
    df: pd.DataFrame,
    models: Models,
    config: AnalysisConfig,
    progress: ProgressFn = _noop,
) -> AnalysisResult:
    texts = df["comment"].tolist()
    ids = df["id"].tolist()
    bs = config.batch_size

    reviews = df[["id", "comment"]].copy().reset_index(drop=True)
    sent = overall_sentiment(models.sentiment(), texts, bs, progress)
    reviews = pd.concat([reviews, sent], axis=1)

    mentions: list[dict] = []
    if config.mode in ("auto", "both"):
        extracted = extract_aspects_llm(models.llm(), texts, config.max_aspects_per_review, max(1, bs // 2), progress)
        for rid, text, terms in zip(ids, texts, extracted):
            for term in terms:
                mentions.append({"id": rid, "aspect": term, "source": "auto", "context": _context_for_term(term, text), "detection_score": None})

    if config.mode in ("user", "both") and config.user_aspects:
        detected = detect_aspects_zero_shot(models.zero_shot(), texts, config.user_aspects, config.detection_threshold, bs, progress)
        for rid, found in zip(ids, detected):
            for aspect, (score, context) in found.items():
                mentions.append({"id": rid, "aspect": aspect, "source": "user", "context": context, "detection_score": score})

    aspects = pd.DataFrame(mentions, columns=["id", "aspect", "source", "context", "detection_score"])
    if not aspects.empty:
        scored = aspect_sentiment(models.absa(), list(zip(aspects["context"], aspects["aspect"])), bs, progress)
        aspects[["aspect_sentiment", "aspect_confidence", "aspect_polarity"]] = pd.DataFrame(scored, index=aspects.index)
    else:
        aspects = aspects.assign(aspect_sentiment=pd.Series(dtype=str), aspect_confidence=pd.Series(dtype=float), aspect_polarity=pd.Series(dtype=float))

    aspects["theme"] = aspects["aspect"]
    if config.group_into_themes and config.themes and (aspects["source"] == "auto").any():
        auto_terms = sorted(aspects.loc[aspects["source"] == "auto", "aspect"].unique())
        mapping = group_terms_into_themes(models.zero_shot(), auto_terms, config.themes)
        is_auto = aspects["source"] == "auto"
        aspects.loc[is_auto, "theme"] = aspects.loc[is_auto, "aspect"].map(mapping)

    values = pd.DataFrame(index=reviews.index)
    if config.analyse_values and config.value_labels:
        values = classify_values(models.zero_shot(), texts, config.value_labels, bs, progress)
        values.insert(0, "id", reviews["id"].values)

    if config.extract_needs:
        reviews["need"] = extract_needs_llm(models.llm(), texts, max(1, bs // 2), progress)

    counts = aspects.groupby("id")["aspect"].apply(lambda s: ", ".join(dict.fromkeys(s))) if not aspects.empty else pd.Series(dtype=str)
    reviews["aspects"] = reviews["id"].map(counts).fillna("")
    progress(1.0, "Done")
    return AnalysisResult(reviews=reviews, aspects=aspects.reset_index(drop=True), values=values, config=config)
