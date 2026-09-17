# OSHC student assistant

Prototype for COS70008. An interactive OSHC education tool for international
students, built on retrieval-augmented generation over Medibank's published
policy documents.

Nothing is trained. Documents are indexed, which takes minutes and only
repeats when Medibank republishes a policy.

## What it does

Three pages. A learning module explaining how Australian healthcare works, a
dashboard showing cover status, and an internal view for the client showing
aggregate patterns only.

An assistant sits on the first two pages. A student asks in their own words
instead of reading a 38 page PDF.

## The runtime path

```
question
  -> safety gate          clinical content stops here, routes to the nurse line
  -> product filter       applied before search, not after
  -> hybrid retrieval     BM25 for exact figures, embeddings for meaning
  -> generation           answers only from the three retrieved passages
  -> number check         blocks any figure not present in the source
  -> readability check    regenerates if above grade 9
  -> answer with source
```

The gate runs first so health information is never collected. Only the
matched topic and confidence score are logged, never the question text or a
member identity.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build the index

Download the policy documents from medibankoshc.com.au into `data/`.

```bash
python ingest/build_index.py data/Medibank_OSHC_Member_Guide.pdf comprehensive
python ingest/build_index.py data/Comprehensive_OSHC_Cover_Summary.pdf comprehensive
python ingest/build_index.py data/Cover_Summary_OSHC_Essentials.pdf essentials
```

Then embed the chunks:

```bash
python ingest/embed.py
```

The embedding model downloads on first run, roughly 130MB, and runs on ONNX
rather than PyTorch. This matters because Streamlit Community Cloud allocates
between 690MB and 2.7GB of memory, and PyTorch alone is around 2GB installed.

## Run

```bash
streamlit run app/main.py
```

Without `GROQ_API_KEY` the app returns the human approved answer for the
matched topic instead of generating. Same interface, no failure on demo day.

To enable generation:

```bash
export GROQ_API_KEY=...
```

For production this would point at an Australian region service, which keeps
processing onshore and avoids cross border disclosure under Australian
Privacy Principle 8.

## Evaluate

```bash
python eval/run_eval.py
```

Fifteen questions covering four failure types: student vocabulary that does
not match policy vocabulary, an ambiguous question that should prompt rather
than guess, three clinical questions that must reach a human, and one off
topic question that should not be answered at all.

Reports safety gate accuracy and recall@3. Change one thing, re-run the same
fifteen, keep it or discard it. Changing three things at once tells you
nothing about which one worked.

## Layout

```
ingest/build_index.py   split by heading, tag with product, section, page, date
ingest/embed.py         precompute vectors
app/pipeline.py         safety gate, hybrid retrieval, validators
app/generate.py         prompt, retry loop, approved fallback
app/main.py             the three pages
eval/run_eval.py        measurement
eval/test_set.json      fifteen labelled questions
```

## Known limits

The prototype does not connect to real member accounts. It uses synthetic
figures for balances and expiry.

Readability scoring counts sentence length and syllables. It cannot tell
whether a term was explained, so "MBS" scores well and is still the hardest
thing in the sentence. The prompt requires policy terms to be defined, and
human review remains the only real test of comprehension.

The safety gate uses a term list, chosen deliberately so a reviewer can see
exactly what triggers it. It is tuned for high recall: routing a policy
question to a nurse costs one phone call, missing a crisis message costs far
more.
