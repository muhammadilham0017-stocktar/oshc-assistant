# Deploying to Streamlit Community Cloud

## Before you push

Apply the crisis routing patch. Testing found that "i want to kill myself"
returned a generic refusal rather than reaching a human. This is the most
serious defect found in the project and it must ship before anything else.

```bash
python patch_crisis.py
python patch_secrets.py
```

Verify:

```bash
python - <<'PY'
import sys; sys.path.insert(0, ".")
from app.pipeline import crisis_check
for q in ["i want to kill myself", "i cant breathe", "saya ingin mati",
          "is the dentist covered", "my tooth hurts"]:
    print(f"  {str(crisis_check(q)):<10} {q}")
PY
```

The first three should return crisis or emergency. The last two should
return None, because they belong to the clinical gate rather than tier one.

## Build the index locally, then commit it

Streamlit Cloud runs the app only. It does not run the ingest scripts, so
the index has to be in the repository.

```bash
python ingest/build_index.py data/Medibank_OSHC_Member_Guide.pdf comprehensive
python ingest/build_index.py data/Comprehensive_OSHC_Cover_Summary.pdf comprehensive
python ingest/build_index.py data/Cover_Summary_OSHC_Essentials.pdf essentials
python ingest/embed.py
git add -f data/chunks.json data/vectors.npy
```

The PDFs stay out. The extracted text does not, because the app needs it.

## Check nothing secret is staged

```bash
git status
git check-ignore -v .env .streamlit/secrets.toml
```

Both must print a matching ignore rule. If either prints nothing, stop.

## Push

```bash
git add -A
git commit -m "Crisis routing, Streamlit deployment config"
git branch -M main
git remote add origin https://github.com/YOURNAME/oshc-assistant.git
git push -u origin main
```

## Deploy

At share.streamlit.io, connect the repository and set the main file to
`app/main.py`.

Then open App settings, Secrets, and paste:

```toml
GROQ_API_KEY = "gsk_your_key"
MODEL = "llama-3.3-70b-versatile"
```

Save. The app restarts automatically.

## Memory

Streamlit Community Cloud allocates roughly 690MB to 2.7GB. The multilingual
embedding model is about 470MB and loads into memory on first use.

If the app runs out of memory, switch to the smaller English model in both
`ingest/embed.py` and `app/pipeline.py`:

```
BAAI/bge-small-en-v1.5
```

That is about 130MB, and costs you cross-language retrieval. Re-run
`ingest/embed.py` and commit the new vectors.

## Known limits to mention in the demo

One test question in ten still fails. "How do i claim money back" returns
the exclusions section rather than the claims section.

The confidence floor thresholds on retrieval score rather than topical
relevance, and the scores cluster within about 0.003, so the threshold
operates on a narrow margin.

The corpus explains benefits, not vocabulary. Asking what a GP is returns
an accurate answer about billing rather than a definition.
