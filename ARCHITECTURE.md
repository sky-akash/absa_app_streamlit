# Architecture: how the ABSA pipeline works

This document explains every conceptual block of the app, from the uploaded file to the charts.
For each block you get what it does, why it is designed that way, which module implements it, and how to extend it.

## Contents

1. [The big picture](#1-the-big-picture)
2. [Design principles](#2-design-principles)
3. [Block 1: Ingestion](#3-block-1-ingestion)
4. [Block 2: Model components](#4-block-2-model-components)
5. [Block 3: Overall sentiment](#5-block-3-overall-sentiment)
6. [Block 4: Aspect finding, automatic mode](#6-block-4-aspect-finding-automatic-mode)
7. [Block 5: Aspect finding, custom mode](#7-block-5-aspect-finding-custom-mode)
8. [Block 6: Aspect-level sentiment](#8-block-6-aspect-level-sentiment)
9. [Block 7: Themes, customer values and needs](#9-block-7-themes-customer-values-and-needs)
10. [Block 8: Aggregation](#10-block-8-aggregation)
11. [Block 9: Findings](#11-block-9-findings)
12. [Block 10: Visualisation and the Streamlit UI](#12-block-10-visualisation-and-the-streamlit-ui)
13. [Data model](#13-data-model)
14. [Module map](#14-module-map)
15. [Extending the app](#15-extending-the-app)
16. [Limitations](#16-limitations)

---

## 1. The big picture

```
 CSV / Excel
     │
     ▼
┌──────────────┐   id, comment
│  Ingestion   │──────────────────────────────────────────────────────────┐
└──────────────┘                                                          │
                                                                          ▼
                         ┌──────────────────────── Pipeline (ordered steps) ───────────────────────┐
                         │                                                                          │
                         │  OverallSentimentStep ──► SentimentModel        (RoBERTa / DistilBERT)   │
                         │                                                                          │
                         │  FindAspectsStep ──┬──► LLMAspectExtractor      (Qwen chat LLM)          │
                         │                    └──► ZeroShotAspectDetector  (DeBERTa / DistilBERT NLI)│
                         │                                                                          │
                         │  AspectSentimentStep ──► AspectSentimentModel   (DeBERTa ABSA)           │
                         │                                                                          │
                         │  ThemeStep, CustomerValueStep ──► ZeroShotModel                          │
                         │  NeedsStep ──► TextGenerator                                             │
                         └────────────────────────────────┬─────────────────────────────────────────┘
                                                          │ reviews, mentions, values
                                                          ▼
                                   ┌────────────┐   ┌──────────┐   ┌─────────────────────┐
                                   │ Aggregation│──►│ Findings │──►│ Charts + Streamlit │
                                   └────────────┘   └──────────┘   └─────────────────────┘
```

The pipeline answers four questions, in this order:

1. **How does each customer feel overall?** (document-level sentiment)
2. **What does each customer talk about?** (aspects, found automatically or from the user's list)
3. **How do they feel about each of those things?** (aspect-level sentiment)
4. **What do they value and need?** (zero-shot value mapping, LLM need statements)

Everything after that (aggregation, findings, charts) is plain pandas and Plotly over the outputs. No model runs there.

## 2. Design principles

**Pretrained transformers instead of clustering.**
Classic topic or aspect discovery (LDA, k-means on embeddings) needs tuning for each dataset and produces clusters that someone has to name. 
Here every step uses a model that already understands the task: 
a sentiment classifier, an ABSA classifier, a natural-language-inference model for zero-shot labelling, and 
an instruction-tuned LLM. Nothing is trained on our data.

**Depend on interfaces, not on models.** 
The pipeline never imports `transformers`. It talks to five small abstract classes 
(`SentimentModel`, `AspectSentimentModel`, `ZeroShotModel`, `TextGenerator`, `AspectFinder`). 
Hugging Face adapters implement them for real use, and rule-based fakes implement them for tests. 
You can swap a model, or a whole provider, without touching the pipeline.

**Reuse the library, keep adapters thin.** 
Tokenisation, batching, truncation, the NLI zero-shot trick and chat templates are all done by `transformers.pipeline`. 
The adapters only reshape inputs and outputs (about 15 lines each).

**Configuration decides the steps.** 
The UI fills one `AnalysisConfig`, and `build_steps(config)` turns it into an ordered list of steps. 
Turning a feature off removes its step, so its model is never loaded.

**Models load lazily and once.** 
A `ModelProvider` hands out a component only when a step asks for it. 
The Streamlit app passes a cached loader, so each model downloads and loads once per server.

**Small, single-purpose modules.** 
Each file has one job: loading data, text utilities, prompts, interfaces, adapters, registry, steps, runner, aggregation, findings, charts, UI.

## 3. Block 1: Ingestion

**Module:** `absa/data_io.py` · **Entry point:** `load_reviews(file_name, data) -> DataFrame[id, comment]`

What it does:

- Reads `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`. CSV separators are sniffed, and encodings fall back from UTF-8 to UTF-8-BOM to Latin-1.
- Finds the columns case-insensitively. `id` can be `review_id` or `comment_id`, and `comment` can be `review`, `text`, `feedback` and so on. 
  A two-column file with other names is read as (id, comment). A file without an id column gets row numbers.
- Strips whitespace, drops empty comments and exact duplicates, and makes repeated ids unique (`17`, `17_1`), because every later join is on `id`.
- Raises `DataError` with a readable message, which the UI shows as is.

## 4. Block 2: Model components

**Modules:** `absa/components/base.py`, `hf.py`, `registry.py`

### 4.1 Interfaces (`base.py`)

| Interface | Input → output | Used for |
|---|---|---|
| `SentimentModel.predict` | texts → `Sentiment` per text | overall review sentiment |
| `AspectSentimentModel.predict` | (context, aspect) pairs → `Sentiment` per pair | sentiment towards an aspect |
| `ZeroShotModel.scores` | texts, labels, hypothesis template → {label: score} per text | custom aspects, values, themes |
| `TextGenerator.generate` | prompts → answer per prompt | automatic aspects, needs |
| `AspectFinder.find` | texts → list of `AspectMention` per text | the two analysis modes |

`Sentiment` is a small value object with `label` (positive / neutral / negative), `confidence` (probability of that label) and `polarity` = P(positive) − P(negative), a continuous score in [−1, 1]. Polarity is what the charts average, because it keeps the model's uncertainty that a hard label throws away.

### 4.2 Hugging Face adapters (`hf.py`)

| Adapter | transformers pipeline task | Notes |
|---|---|---|
| `HFSentimentModel` | `text-classification` | `top_k=None` returns every class. `normalize_label` maps any scheme (`positive`, `LABEL_2`, `5 stars`) onto three classes. |
| `HFAspectSentimentModel` | `text-classification` with sentence pairs | Sends `{"text": context, "text_pair": aspect}`, which is how ABSA models were trained. |
| `HFZeroShotModel` | `zero-shot-classification` | NLI under the hood: "premise = review, hypothesis = *This customer review talks about {label}.*" |
| `HFChatGenerator` | `text-generation` | Sends chat messages so the model's own chat template is applied. Greedy decoding gives stable output. Uses left padding for batching. |

Each adapter has a `load(model_id)` class method that builds the pipeline on GPU, Apple MPS or CPU, whichever is available.

### 4.3 Registry and presets (`registry.py`)

- `Role`: the four model slots (`SENTIMENT`, `ABSA`, `ZERO_SHOT`, `LLM`).
- `CATALOG`: a list of `ModelSpec(role, display name, hub id)`. The UI dropdowns are built from it.
- `PRESETS`: named choices of one model per role ("Accurate (RoBERTa / DeBERTa)" and "Fast (BERT / DistilBERT)").
- `ADAPTERS`: which adapter class serves which role.
- `load_component(role, model_id, batch_size)`: the factory.
- `ModelProvider(selection, batch_size, loader)`: what the pipeline receives. It calls `loader` lazily, so the app can pass a cached loader and tests a fake one.

### 4.4 Model choices

| Role | Accurate preset | Fast preset | Why |
|---|---|---|---|
| Overall sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | Both are 3-class, so neutral reviews are recognised. DistilBERT is about 2x faster and multilingual. |
| Aspect sentiment | `yangheng/deberta-v3-base-absa-v1.1` | same | The most widely used ABSA model on the Hub. There is no well-maintained BERT/DistilBERT ABSA model, and DeBERTa belongs to the BERT family. |
| Zero-shot | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `typeform/distilbert-base-uncased-mnli` | DeBERTa is far more reliable at zero-shot. DistilBERT MNLI is offered for speed. |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` | same | Small enough for a CPU, and follows "reply with a comma-separated list" well. Larger 1.5B and 3B variants can be selected. |

## 5. Block 3: Overall sentiment

**Step:** `OverallSentimentStep` · **Component:** `SentimentModel`

Each review is classified as a whole. The step writes three columns to `reviews`: `sentiment`, `sentiment_confidence` and `polarity`. Long reviews are truncated at 512 tokens by the pipeline.

This is the baseline the rest of the analysis is compared against. For example, the *impact* of an aspect (Block 8) is measured as the change in overall polarity.

## 6. Block 4: Aspect finding, automatic mode

**Step:** `FindAspectsStep(llm_finder)` · **Strategy:** `LLMAspectExtractor` · **Component:** `TextGenerator`

Goal: discover what customers talk about without being told what to look for.

1. **Prompt** (`prompts.ASPECT_EXTRACTION`): the LLM is asked for the aspects (product features, service elements, parts of the experience) as short noun phrases copied from the review, comma-separated, or `none`.
2. **Parse** (`text.parse_aspect_list`): the answer is split on commas, semicolons, new lines and "and", and bullets and numbering are stripped.
3. **Normalise** (`text.normalize_aspect`): lowercase, drop determiners and possessives, singularise the head noun. "The Batteries" and "battery" become one aspect.
4. **Filter**: generic words ("product", "everything", "experience") are dropped.
5. **Ground** (`text.is_grounded`): every word of the aspect, or its stem, must appear in the review. This removes aspects the LLM invented, which is the main failure mode of small LLMs.
6. **Locate context** (`text.context_for_term`): the sentences containing the aspect are kept as its context for Block 6.

Why an LLM rather than clustering or keyword extraction: an instruction-tuned model understands that "it died after two hours" is about *battery*, works on any domain without training, and returns human-readable names directly.

## 7. Block 5: Aspect finding, custom mode

**Step:** `FindAspectsStep(zero_shot_finder)` · **Strategy:** `ZeroShotAspectDetector` · **Component:** `ZeroShotModel`

Goal: check the aspects the business cares about, even when customers use different words ("took forever to arrive" → *delivery*).

1. Every review is split into sentences (`text.split_sentences`).
2. Every sentence is scored against every user aspect with the hypothesis *"This customer review talks about {aspect}."* `multi_label=True` scores each aspect independently, so a sentence can match several.
3. A review mentions an aspect when any sentence reaches the **detection threshold** (default 0.6, adjustable in the sidebar). The matching sentences become the context and the best score is kept as `detection_score`.

Working per sentence is what makes mixed reviews come out right: in "The food was great. Service was slow." the aspect *service* only sees "Service was slow."

Both modes return the same `AspectMention` objects, which is why the rest of the pipeline does not care which mode found them. In "Both" mode the two finders simply run one after the other, and each mention is tagged with its `source` (`auto` or `user`).

## 8. Block 6: Aspect-level sentiment

**Step:** `AspectSentimentStep` · **Component:** `AspectSentimentModel`

For every mention, the ABSA model reads the pair (context, aspect) and decides how the customer feels about that aspect specifically. "Great screen but the battery is awful" gives *screen* positive and *battery* negative, even though overall sentiment might be neutral. Output columns: `aspect_sentiment`, `aspect_confidence`, `aspect_polarity`.

## 9. Block 7: Themes, customer values and needs

These optional steps add the "why" behind the "what".

**Themes** (`ThemeStep`, zero-shot, single-label). Automatic extraction can return many specific terms ("battery life", "charger", "charging speed"). Each unique term is classified once into a user-editable list of themes (quality, price, delivery, ...). A term scoring below 0.3 on every theme becomes "other". The dashboard can then switch between aspect level and theme level, and shows a sunburst of themes and their aspects.

**Customer values** (`CustomerValueStep`, zero-shot, multi-label). Each review is scored against value statements such as *price and value for money*, *reliability and durability* and *trust and safety*, with the hypothesis *"The customer cares about {value}."* This captures motivations that are often implied rather than said. Values above the threshold count for that review.

**Needs** (`NeedsStep`, LLM, optional because it is slower). The LLM writes one sentence per review starting with "Wants ...", for example "Wants a charger that lasts more than a month". The UI lists them for negative and neutral reviews, where they point at unmet expectations.

## 10. Block 8: Aggregation

**Module:** `absa/insights/aggregation.py` (pure pandas, no models)

| Function | Output |
|---|---|
| `overall_summary` | counts, % positive / negative, net sentiment = (pos − neg) / n, mean polarity |
| `aspect_summary` | one row per aspect (or theme). Details below. |
| `cooccurrence` | aspect × aspect matrix of reviews mentioning both |
| `negative_pairs` | aspect pairs criticised together in the same review |
| `value_summary` | per value: share of reviews, mean polarity, % negative |
| `mixed_reviews` | positive reviews that still contain a negative aspect |

`aspect_summary` computes these columns:

- **mentions / share_of_reviews:** importance. A review mentioning an aspect twice counts once, with its polarities averaged.
- **positive / neutral / negative, pct_\*, net_sentiment:** performance.
- **impact:** mean overall polarity of reviews that mention the aspect minus that of reviews that do not. It is a simple driver analysis that shows which aspects pull satisfaction up or down. It is correlational, not causal.
- **quadrant:** an importance–performance analysis with the median share and zero net sentiment as cut-offs:
  - *Core strength:* frequent and positive. Protect it.
  - *Fix first:* frequent and negative. Top priority.
  - *Hidden gem:* rare and positive. Worth promoting.
  - *Watch list:* rare and negative. Possibly an emerging issue.

## 11. Block 9: Findings

**Module:** `absa/insights/findings.py`

The findings are a list of small **rule functions**, each registered with the `@rule` decorator. Each rule receives an `InsightInputs` bundle (reviews, mentions, aspect table, value table, minimum mentions) and returns one sentence, or `None` when it does not apply. `generate_insights` runs them in order.

Built-in rules: overall mood, most discussed aspect, top strength, biggest pain point, strongest drag on satisfaction, strongest delight driver, hidden pain points, hidden gems, friction among happy customers, linked complaints, what customers value most, unmet need.

Rules only consider aspects with at least *min mentions* (set in the UI), so a single angry review does not become a headline.

## 12. Block 10: Visualisation and the Streamlit UI

**Modules:** `absa/viz.py` (figure builders) and `app.py` (layout and state)

Every chart is a pure function from a DataFrame to a Plotly figure, so it can be reused outside Streamlit (notebooks, reports). Colours follow one rule set: positive is blue, negative is red, neutral is grey, and continuous sentiment uses a red–grey–blue diverging scale.

| Tab | Charts |
|---|---|
| Overview | KPI tiles, sentiment donut, polarity histogram, sentiment per aspect |
| Aspects | frequency bars, diverging sentiment bars, priority matrix, co-occurrence heatmap, theme sunburst, aspect table |
| Insights | findings, driver (impact) bars, linked complaints, positive and negative quotes per aspect, mixed reviews |
| Customer values & needs | value share coloured by satisfaction, aspect × value heatmap, value table, LLM needs |
| Review explorer | filter by sentiment and aspect, CSV downloads |

`app.py` holds no analysis logic. It collects settings into an `AnalysisConfig`, builds a `ModelProvider` with a `st.cache_resource` loader, calls `run_analysis` with a progress bar, keeps the `AnalysisResult` in `st.session_state` so filters do not re-run the models, and renders the tabs.

## 13. Data model

`run_analysis` returns an `AnalysisResult` with three tables:

**`reviews`** (one row per review)

| column | meaning |
|---|---|
| `id`, `comment` | from the upload |
| `sentiment`, `sentiment_confidence`, `polarity` | Block 3 |
| `aspects` | comma-separated aspects found |
| `need` | Block 7, when enabled |

**`aspects`** (one row per review × aspect mention)

| column | meaning |
|---|---|
| `id` | review id |
| `aspect` | normalised aspect name |
| `source` | `auto` (LLM) or `user` (zero-shot) |
| `context` | sentence(s) the aspect appears in |
| `detection_score` | zero-shot confidence (custom mode only) |
| `aspect_sentiment`, `aspect_confidence`, `aspect_polarity` | Block 6 |
| `theme` | Block 7 (equals `aspect` when not grouped) |

**`values`**: `id` plus one score column per customer value.

## 14. Module map

```
app.py                         Streamlit UI: settings → AnalysisConfig → run_analysis → tabs
absa/
  config.py                    AnalysisConfig, AspectMode, default value and theme labels
  data_io.py                   Block 1: CSV/Excel → (id, comment)
  text.py                      sentence split, label/aspect normalisation, LLM output parsing, grounding
  prompts.py                   LLM prompts and zero-shot hypothesis templates
  components/
    base.py                    interfaces + Sentiment / AspectMention value objects
    hf.py                      Hugging Face pipeline adapters
    aspect_finders.py          LLMAspectExtractor, ZeroShotAspectDetector
    registry.py                Role, CATALOG, PRESETS, load_component, ModelProvider
  pipeline/
    steps.py                   AnalysisContext + one Step class per block
    runner.py                  build_steps(config), run_analysis, AnalysisResult
  insights/
    aggregation.py             Block 8
    findings.py                Block 9: rule registry
  viz.py                       Block 10: Plotly figure builders
tests/
  fakes.py                     rule-based implementations of every interface
  test_components.py           adapters (against stand-in pipelines), finders, registry
  test_pipeline.py             step selection, all three modes, insights, charts
  test_text_and_io.py          ingestion and text utilities
```

## 15. Extending the app

**Add a model:** add one `ModelSpec` line to `CATALOG`. It appears in the sidebar automatically.

**Add a preset:** add one entry to `PRESETS` with a hub id for each `Role`.

**Use a different provider** (an API LLM, ONNX, a local server): write a class implementing the relevant interface, for example `TextGenerator.generate`, and register it in `ADAPTERS` or pass your own `loader` to `ModelProvider`.

**Add an analysis step** (for example emotion detection or language detection): subclass `Step`, write results onto `ctx.reviews` or `ctx.mentions`, and add it in `build_steps` behind a config flag.

**Add a finding:** write a function `(InsightInputs) -> str | None` and decorate it with `@rule`.

**Add a chart:** add a function to `viz.py` that takes a DataFrame and returns a figure, then call it in the relevant tab of `app.py`.

**Tune behaviour without code:** change prompts and hypothesis templates in `prompts.py`, and default labels in `config.py`.

## 16. Limitations

- **Speed on CPU.** The LLM step dominates. Expect roughly 1 to 3 seconds per review with automatic extraction on a laptop CPU. Use the Fast preset, a smaller batch of reviews, or a GPU for large files.
- **English first.** The ABSA and DeBERTa zero-shot models are English. The DistilBERT and nlptown sentiment models are multilingual, so overall sentiment works in other languages but aspect sentiment is less reliable.
- **Small-LLM recall.** Grounding prevents invented aspects but cannot recover missed ones. A larger Qwen model improves recall.
- **Thresholds matter.** The detection and value thresholds trade precision for recall. 0.6 is a reasonable start; check a few reviews in the explorer and adjust.
- **Impact is correlational.** It shows which aspects co-occur with happier or unhappier reviews, not proof that fixing them changes satisfaction.
