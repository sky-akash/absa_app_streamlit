# Aspect-Based Sentiment Analysis (Streamlit)

A Streamlit app that reads customer reviews and tells you:

- how customers feel overall (positive / neutral / negative, with a polarity score),
- which aspects they talk about (battery, delivery, price, staff, ...),
- how they feel about each aspect,
- what they value and where their needs are not being met.

Everything runs on pretrained Hugging Face transformers. No clustering and no training on your data.

## Two ways to find aspects

| Mode | How it works |
|---|---|
| **Automatic** | A small instruction-tuned LLM (Qwen2.5 Instruct) reads each review and lists the aspects mentioned. Aspects the review does not actually contain are dropped, and similar forms are merged ("batteries" → "battery"). Optionally, a zero-shot model groups them into broader themes. |
| **Custom** | You type the aspects you care about. A zero-shot NLI model checks each sentence for each aspect, so in "food was great, service was slow" each aspect gets only its own sentence. |
| **Both** | Runs both and lets you switch between them. |

The sentiment towards each aspect comes from an ABSA model that reads the sentence together with the aspect.

## Models

| Step | Default ("Accurate" preset) | "Fast" preset (BERT / DistilBERT) |
|---|---|---|
| Overall sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` |
| Aspect sentiment | `yangheng/deberta-v3-base-absa-v1.1` | same |
| Custom aspects, values, themes (zero-shot) | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `typeform/distilbert-base-uncased-mnli` |
| Automatic aspect extraction, needs (LLM) | `Qwen/Qwen2.5-0.5B-Instruct` | same |

Also selectable: `nlptown/bert-base-multilingual-uncased-sentiment` (BERT, 1 to 5 stars), DistilBERT SST-2, BART-large-MNLI, larger DeBERTa and Qwen models.

**Why not BERT/DistilBERT everywhere?** DistilBERT works well for overall sentiment and is about twice as fast. There is no well-maintained BERT or DistilBERT model for aspect-level sentiment, so both presets use the DeBERTa ABSA model, which belongs to the same BERT family. DistilBERT MNLI is noticeably weaker at zero-shot aspect detection (see [this discussion](https://discuss.huggingface.co/t/poor-performance-in-zero-shot-learning-when-using-the-model-typeform-distilbert-base-uncased-mnli/6374)), so DeBERTa is the default there.

## Dashboard

- **Overview:** sentiment split, polarity distribution, sentiment per aspect.
- **Aspects:** most mentioned aspects, diverging sentiment bars, a priority matrix (importance vs. performance: *Core strength*, *Fix first*, *Hidden gem*, *Watch list*), co-occurrence heatmap, theme sunburst and a full aspect table.
- **Insights:** plain-language findings: top strength, biggest pain point, drivers that raise or lower overall satisfaction, hidden pain points, friction among happy customers, complaints that come together, plus example quotes per aspect.
- **Customer values & needs:** what customers care about (value for money, quality, service, delivery, ...), how satisfied each group is, which aspects carry which values, and optionally each customer's underlying need written by the LLM.
- **Review explorer:** filter reviews by sentiment and aspect, and download the results as CSV.

## Input

A CSV or Excel file with two columns, `id` and `comment`. Common alternatives (`review`, `text`, `review_id`, ...) are recognised too. A sample is in `sample_data/sample_reviews.csv`.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Models download from Hugging Face on first use (about 2 GB for the default preset) and are cached afterwards. A GPU is used automatically when available. On a CPU, expect about 1 to 3 seconds per review with automatic extraction; start with a few hundred reviews.

## Tests

```bash
pytest
```

The tests use small fake models, so they run without downloading any weights.

## Project layout

```
app.py               Streamlit UI
absa/data_io.py      CSV/Excel loading and validation
absa/models.py       Model registry, presets and loaders
absa/analysis.py     Sentiment, aspect extraction/detection, ABSA, values, needs
absa/insights.py     Aggregations and generated findings
absa/viz.py          Plotly charts
tests/               Pipeline tests with fake models
```
