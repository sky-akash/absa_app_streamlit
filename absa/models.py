"""Hugging Face model registry and loaders.

Every model here is a pretrained transformer used as-is (no training needed):

* overall sentiment  -> RoBERTa fine-tuned for 3-class sentiment
* aspect sentiment   -> DeBERTa-v3 fine-tuned for aspect-based sentiment (ABSA)
* aspect detection   -> DeBERTa-v3 NLI model used for zero-shot classification
* aspect extraction  -> small instruction-tuned LLM (Qwen2.5 Instruct)
"""

from __future__ import annotations

from dataclasses import dataclass

SENTIMENT_MODELS = {
    "RoBERTa sentiment (cardiffnlp, 3-class)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "DistilBERT sentiment (lxyuan, 3-class, multilingual, fast)": "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "BERT review stars (nlptown, 1-5 stars, multilingual)": "nlptown/bert-base-multilingual-uncased-sentiment",
    "DistilBERT SST-2 (2-class, no neutral)": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
}

ABSA_MODELS = {
    "DeBERTa-v3 base ABSA (yangheng)": "yangheng/deberta-v3-base-absa-v1.1",
    "DeBERTa-v3 large ABSA (yangheng, more accurate)": "yangheng/deberta-v3-large-absa-v1.1",
}

ZERO_SHOT_MODELS = {
    "DeBERTa-v3 base zero-shot v2.0": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "DistilBERT MNLI (fastest, least accurate)": "typeform/distilbert-base-uncased-mnli",
    "BART large MNLI": "facebook/bart-large-mnli",
    "DeBERTa-v3 large zero-shot v2.0 (more accurate)": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
}

LLM_MODELS = {
    "Qwen2.5 0.5B Instruct (fast)": "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen2.5 1.5B Instruct (balanced)": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5 3B Instruct (best, needs more RAM)": "Qwen/Qwen2.5-3B-Instruct",
}


# Named presets the sidebar offers; values are keys of the dicts above.
# There is no well-maintained BERT/DistilBERT model for aspect-level
# sentiment, so both presets use the DeBERTa ABSA model for that step.
PRESETS = {
    "Accurate (RoBERTa / DeBERTa)": {
        "sentiment": "RoBERTa sentiment (cardiffnlp, 3-class)",
        "absa": "DeBERTa-v3 base ABSA (yangheng)",
        "zero_shot": "DeBERTa-v3 base zero-shot v2.0",
        "llm": "Qwen2.5 0.5B Instruct (fast)",
    },
    "Fast (BERT / DistilBERT)": {
        "sentiment": "DistilBERT sentiment (lxyuan, 3-class, multilingual, fast)",
        "absa": "DeBERTa-v3 base ABSA (yangheng)",
        "zero_shot": "DistilBERT MNLI (fastest, least accurate)",
        "llm": "Qwen2.5 0.5B Instruct (fast)",
    },
}


@dataclass
class ModelChoice:
    sentiment: str = SENTIMENT_MODELS["RoBERTa sentiment (cardiffnlp, 3-class)"]
    absa: str = ABSA_MODELS["DeBERTa-v3 base ABSA (yangheng)"]
    zero_shot: str = ZERO_SHOT_MODELS["DeBERTa-v3 base zero-shot v2.0"]
    llm: str = LLM_MODELS["Qwen2.5 0.5B Instruct (fast)"]


def _device():
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return -1


def load_text_classifier(model_name: str):
    from transformers import pipeline

    return pipeline("text-classification", model=model_name, device=_device())


def load_zero_shot(model_name: str):
    from transformers import pipeline

    return pipeline("zero-shot-classification", model=model_name, device=_device())


def load_llm(model_name: str):
    from transformers import pipeline

    generator = pipeline("text-generation", model=model_name, device=_device())
    # Left padding is required for batched generation with decoder-only models.
    generator.tokenizer.padding_side = "left"
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token
    return generator
