# FreightTiger shipping-cost watcher

Finds routes whose freight cost per tonne-km is rising in a way that looks out of the ordinary, then
either cites a real note that explains it or says plainly that it **could not find one**.

**Design principle:** every decision that affects the score (the numbers, the flags, the verdicts) is made
by deterministic code. Retrieval only proposes candidate notes; a rule gate has the final say. A wrong
reason is worse than "couldn't find one", so the system defaults to "unexplained".

## Run it

```bash
pip install -r requirements.txt
python -m src.pipeline        # -> outputs/results.csv (+ outputs/weekly_metrics.csv, logs/run_metrics.json)
python -m pytest -q           # 28 tests
python -m src.evaluate        # -> logs/eval_report.md, logs/sensitivity.csv
python -m src.repro_check     # 3 fresh-process runs, sha256 compared -> logs/repro_check.txt
```

Built and tested on Python 3.13 with pandas, numpy and scikit-learn (see `logs/run_metrics.json` for exact versions). A full run takes well under a second and makes **no network or LLM calls**.

## Result on the supplied data

2,940 shipments, 7 routes, 104 weeks (728 route-weeks): **18 flagged, 3 justified, 15 unexplained.**

| Pattern | Weeks | Verdict |
|---|---|---|
| Delhi-Jaipur spike | 2024-11-11, 11-18 | Unexplained: no note exists for this route or period |
| Ahmedabad-Mumbai festival week | 2025-01-20 | Justified by N002 |
| Chennai-Bangalore flood | 02-24, 03-03 | Justified by N001 (window 24 Feb to 8 Mar) |
| Chennai-Bangalore after the flood | 03-10, 03-17 | **Unexplained**: cost is still +22% / +14%, but N001 has expired and N009 says conditions were normal |
| Mumbai-Pune slow creep | 11 weeks, 2025-09-15 to 12-22 | Unexplained: cost rose about 22% over 2025 on every transporter; the only nearby notes (N006, N010) say nothing changed |

## How it works

1. **Validate** (`data_loader.py`): unique IDs, no nulls, positive quantity/distance/cost, one `route_type` per
   route, and no missing route-weeks (so "8 weeks" means 8 calendar weeks). It fails loudly otherwise.
2. **Weekly engine** (`weekly_analysis.py`): `week_of` is the Monday of the shipment's Mon-Sun week. Route is
   origin-destination. Cost per tonne-km is `sum(cost) / sum(tonnes x km)`, a ratio of sums as the brief defines it.
3. **Baselines, exactly as the brief defines them:**
   - `vs_own_history`: mean of the route's previous 8 weekly values, **strictly before** the current week.
     With fewer than 8 prior weeks it uses all that exist (weeks 2 to 8); week 1 has no history and shows `n/a`. Nothing is padded or extrapolated.
   - `vs_similar_routes`: same-week mean over the **other** routes with the same `route_type` (the route itself is excluded).
4. **Flagging** (`anomaly_detection.py`, thresholds in `config.py`): a route-week is an anomaly if cost is
   *rising* (above its own history) **and** either
   - the rise vs own history is >= 20%, or
   - the gap vs similar routes is >= 20% **and** the route has drifted up >= 10% over 26 weeks.

   A peer gap alone never flags: a route can be pricier than its peers without its cost rising. The 26-week
   drift term exists because a trailing 8-week average lags a slow creep (Mumbai-Pune is only +3% to +9% vs
   its own history even though it is up about 22% over the year).
5. **Notes** (`notes.py`, `retrieval.py`, `explain.py`):
   - *Retrieval:* TF-IDF vectors with cosine similarity rank all 10 notes for the route/week. Ties break by note ID, so ranking is deterministic.
   - *Gate:* a hand-checked metadata table gives each note a scope, an effect window and a polarity
     (`increase`, `no_material_increase`, `normalization`, `not_applicable`). A note justifies a row only if
     the route is in scope, the polarity is `increase`, the week overlaps the effect window, and (where the
     note states a size) the observed rise is consistent with it. The loader cross-checks the table against the CSV.
   - *Reasons:* fixed templates filled from the verdict, note ID, dates and numbers. Unexplained rows leave
     `matched_note_id` blank and may name the nearest in-scope notes within 28 days and why they were rejected.
6. **Output** (`reporting.py`): the header and column order come from `sample_output_format_v2.csv`. Cost
   has 2 decimals and percentages 1 decimal, like the sample. `flagged` is `Yes` for an unexplained anomaly
   and `No (justified)` for one backed by a note, the two values the sample uses.

## How I convinced myself it is right

Full numbers are in `logs/eval_report.md` (generated, not hand-typed).

| Check | Result |
|---|---|
| **Sample rows**: all graded fields of the 3 sample rows (cost, both comparisons, flagged, matched_note_id) | 15 / 15 identical |
| **Note traps**: 24 labelled cases (wrong route, wrong week, expired window, "no cost impact" notes, rise larger than the note allows, nothing in scope) | 24 / 24 correct |
| **Unit tests** with hand-checkable data (week boundary, ratio of sums, no look-ahead, <8 weeks of history, peer exclusion) | 28 tests pass |
| **Audit of justified rows**: each cited note re-checked from scratch against route, window and polarity | pass |
| **Injection**: 12 planted single-week cost rises in a copy of the data | see below |
| **Reproducibility**: 3 fresh-process runs, sha256 of outputs | identical (`logs/repro_check.txt`) |

Single-week spike detection (planted rise vs. share found): +10% 1/12, +15% 5/12, +20% 9/12, **+25% and above
12/12**. In every case there were 0 planted rises wrongly "justified" and 0 new false alarms elsewhere.

**Why 20%?** The brief gives no numeric threshold, so 20% is a configurable policy choice, not a requirement.
It is the largest round value at which all three sample rows are still flagged: at 25% the Mumbai-Pune sample
row (+23.6% vs peers) drops out. `logs/sensitivity.csv` shows exactly which route-weeks enter or leave at
8/10/15/20/25/30% (18 flags at 20%, 22 at 15%, 41 at 10%, 47 at 8%).

## Reproducibility

Everything is deterministic: no randomness, no LLM, pinned `sort` tie-breaks, fixed number formatting and
template wording. `python -m src.repro_check` runs the pipeline three times in separate processes and compares
sha256 of `results.csv` and `weekly_metrics.csv`: **identical** (flags, numbers, cited notes and wording).

## Cost of one full run (`logs/token_cost_log.md`, `logs/run_metrics.json`)

LLM calls **0**, input tokens **0**, output tokens **0**, cost at a model's published rate **$0.00**. Retrieval is
local TF-IDF (scikit-learn); runtime about 0.1 s for the whole shipment file.

## Assumptions and decisions you should know about

- **Output contains anomalous rows only**, as the sample does (the sample has 3 rows, not all route-weeks).
  The full weekly table is in `outputs/weekly_metrics.csv`.
- **Threshold:** own-history and peer thresholds are both 20%, the drift guard is 10% over 26 weeks (all configurable).
  In the first 26 weeks of data drift is unavailable, so a peer-driven flag relies on "rising" alone. This does
  not matter at 20% (no such flags) but does at low thresholds (see the 8% and 10% rows of the sensitivity table).
- **N002** is dated 2025-01-20 and calls it "a festival week", so its effect window is that Monday-to-Sunday week.
- **N003** (diesel, +5-7% from 2025-05-05) has no end date. I let it explain rises for **8 weeks** (to 2025-06-29),
  because a step change only shows as a "rise vs own history" while the trailing baseline pre-dates it, and
  only up to 1.5x its stated 7% (about 10.5%). A bigger rise stays unexplained rather than stretching the note.
  *This window and tolerance are my assumptions, not facts from the note.*
- **Retrieval is TF-IDF, not a neural embedding/vector DB.** With 10 short notes it is enough, it is free and
  exactly repeatable, and the sandbox I built this in could not download a model. `NoteRetriever.rank()` is the
  only interface, so a sentence-embedding backend could be swapped in. The gate, not retrieval, decides verdicts.
- **The brief names `sample_output_format.csv`; the file I received is `sample_output_format_v2.csv`.** I used
  its header as the contract. Also, its third data row has an *unquoted comma* inside `reason`, so a standard CSV
  parser reads 9 fields. My output quotes such fields properly; `load_sample_output()` reads the sample tolerantly.

## Limitations (honest)

- The mandated trailing-average baseline is **contaminated by a rise it has just seen**: in the 4-week +25% step
  test only 2 of 4 weeks were flagged, and a slow 1.2%/week creep was first flagged in week 17 (+20% cumulative) and in
  only 8 of 30 weeks. The 26-week drift guard helps but does not remove this.
- Short and Long route types contain only 2 routes each, so a "peer average" is a single other route.
  Mumbai-Pune normally sits about 12% above Delhi-Jaipur.
- Spikes below about +20% are mostly missed at the 20% threshold (+15%: 5/12; +10%: 1/12). That is the price of few false alarms.
- Note metadata is hand-written for 10 notes. For a large note base this step would need extraction with
  human review; the gate would stay the same.

**Improvements I considered:** a persistence rule or a "clean" baseline that excludes already-flagged weeks
(fixes contamination), a CUSUM-style drift detector for creeps, and sentence embeddings for retrieval.

## Layout

```
src/   config, data_loader, weekly_analysis, anomaly_detection, notes, retrieval, explain, reporting,
       pipeline, evaluate, repro_check
tests/ test_weekly_analysis, test_notes_gate, test_pipeline
data/  the three input CSVs        outputs/  results.csv, weekly_metrics.csv
logs/  run_metrics.json, token_cost_log.md, repro_check.txt, eval_report.md, sensitivity.csv
```
