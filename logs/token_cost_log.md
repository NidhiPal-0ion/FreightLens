# Token and cost log: one full run over the entire shipment file

Command: `python -m src.pipeline` (2,940 shipments, 728 route-weeks, 18 flagged rows explained)

| Item | Value |
|---|---|
| LLM calls | 0 |
| Total input tokens | 0 |
| Total output tokens | 0 |
| Cost at a model's published rate | $0.00 |
| Retrieval | TF-IDF + cosine similarity (scikit-learn), local and offline; no embedding API |
| Explanations | Fixed templates filled from the verdict, cited note, dates and numbers |
| Runtime | about 0.1 s (see `run_metrics.json`) |

Why zero: nothing in the pipeline needs generative text. The numbers are pandas, the verdict is a rule
gate over hand-checked note metadata, and the `reason` column is templated. That also makes the output
byte-identical across runs (`repro_check.txt`). If an LLM were added to paraphrase reasons, it would run at
temperature 0 with the verdict and cited note fixed by code, and its tokens would be logged here.
