"""Evaluation harness: how we know the system can be trusted.

    python -m src.evaluate        -> logs/eval_report.md, logs/sensitivity.csv

Four independent checks, all computed (nothing hand-typed into the report):
  1. Sample rows   - the three rows in sample_output_format_v2.csv are reproduced exactly.
  2. Note traps    - labelled route/week/note cases; every wrong-route, wrong-window,
                     "no cost impact" and oversize-rise case must come out UNEXPLAINED.
  3. Injection     - known cost rises are planted in a copy of the data; we measure whether
                     they are found (recall), whether they are wrongly "justified", and
                     whether new false alarms appear.
  4. Sensitivity   - how the flagged set changes with the threshold.
"""
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config, DEFAULT_CONFIG
from .data_loader import load_notes, load_sample_output, load_shipments, load_template_columns
from .explain import decide
from .pipeline import ROOT, NOTES_FILE, SHIPMENTS_FILE, TEMPLATE_FILE, analyse, explain_anomalies
from .retrieval import NoteRetriever

# --------------------------------------------------------------------------- 1. sample rows
SAMPLE_FIELDS = ["cost_per_tonne_km", "vs_own_history", "vs_similar_routes", "flagged", "matched_note_id"]


def sample_row_check(results: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Compare every graded field of each sample row with our output for the same route/week."""
    rows = []
    res = results.fillna("").astype(str)
    smp = sample.fillna("").astype(str)
    for _, s in smp.iterrows():
        match = res[(res["route"] == s["route"]) & (res["week_of"] == s["week_of"])]
        for field in SAMPLE_FIELDS:
            ours = match.iloc[0][field] if len(match) else "<row missing>"
            rows.append({"route": s["route"], "week_of": s["week_of"], "field": field,
                         "sample": s[field], "ours": ours, "match": ours == s[field]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. note traps
@dataclass(frozen=True)
class Trap:
    name: str
    route: str
    week: str            # Monday, YYYY-MM-DD
    own_pct: float
    expect_note: str     # "" = must be unexplained


TRAPS = [
    # --- should be justified
    Trap("N002 festival week itself", "Ahmedabad-Mumbai", "2025-01-20", 29.5, "N002"),
    Trap("N001 first flood week", "Chennai-Bangalore", "2025-02-24", 31.5, "N001"),
    Trap("N001 last (partial) flood week", "Chennai-Bangalore", "2025-03-03", 22.8, "N001"),
    Trap("N003 diesel, rise inside stated band", "Mumbai-Delhi", "2025-05-12", 6.0, "N003"),
    Trap("N003 diesel, other route, last window week", "Delhi-Chennai", "2025-06-23", 7.0, "N003"),
    Trap("N003 diesel, at the magnitude tolerance", "Kolkata-Bhubaneswar", "2025-05-19", 10.4, "N003"),
    # --- must be unexplained
    Trap("N002 right route, week after festival", "Ahmedabad-Mumbai", "2025-01-27", 25.0, ""),
    Trap("N002 right route, week before festival", "Ahmedabad-Mumbai", "2025-01-13", 25.0, ""),
    Trap("N002 wrong route, festival week", "Mumbai-Pune", "2025-01-20", 25.0, ""),
    Trap("N001 wrong route, flood week", "Delhi-Jaipur", "2025-02-24", 30.0, ""),
    Trap("N001 expired: week after flood ended", "Chennai-Bangalore", "2025-03-10", 22.1, ""),
    Trap("N009 'back to normal' week", "Chennai-Bangalore", "2025-03-17", 13.7, ""),
    Trap("N001 right route, a month later", "Chennai-Bangalore", "2025-04-14", 25.0, ""),
    Trap("N003 rise far above stated 5-7%", "Mumbai-Delhi", "2025-05-12", 11.6, ""),
    Trap("N003 huge rise inside window", "Kolkata-Bhubaneswar", "2025-06-09", 25.0, ""),
    Trap("N003 week before it started", "Mumbai-Delhi", "2025-04-28", 6.0, ""),
    Trap("N003 week after its assumed window", "Delhi-Chennai", "2025-07-07", 6.0, ""),
    Trap("N005 'costs not affected'", "Mumbai-Delhi", "2024-07-29", 25.0, ""),
    Trap("N006 'stable demand'", "Mumbai-Pune", "2025-09-15", 9.2, ""),
    Trap("N010 'no rate change'", "Mumbai-Pune", "2025-10-27", 5.5, ""),
    Trap("N004 'routes not in dataset'", "Mumbai-Delhi", "2024-03-11", 25.0, ""),
    Trap("N007 'roads improved'", "Delhi-Jaipur", "2024-05-20", 25.0, ""),
    Trap("route with no notes at all", "Delhi-Chennai", "2024-11-11", 25.0, ""),
    Trap("N008 'no disruptions' near N003 window end", "Kolkata-Bhubaneswar", "2025-07-07", 25.0, ""),
]


def run_traps(retriever: NoteRetriever, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    rows = []
    for t in TRAPS:
        d = decide(t.route, date.fromisoformat(t.week), t.own_pct, retriever, cfg)
        rows.append({"case": t.name, "route": t.route, "week": t.week, "own_pct": t.own_pct,
                     "expected_note": t.expect_note or "(unexplained)",
                     "got_note": d.note_id or "(unexplained)",
                     "pass": d.note_id == t.expect_note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3. injection
def inject(shipments: pd.DataFrame, route: str, week_starts: list, factors: list) -> pd.DataFrame:
    """Multiply freight cost of `route` in each listed week by the matching factor (copy)."""
    out = shipments.copy()
    wk = out["shipment_date"] - pd.to_timedelta(out["shipment_date"].dt.weekday, unit="D")
    for week, factor in zip(week_starts, factors):
        mask = (out["route"] == route) & (wk == pd.Timestamp(week))
        if not mask.any():
            raise ValueError(f"nothing to inject for {route} {week}")
        out.loc[mask, "freight_cost_inr"] = (out.loc[mask, "freight_cost_inr"] * factor).round().astype(int)
    return out


SPIKES = [
    ("Kolkata-Bhubaneswar", "2024-06-17"), ("Delhi-Chennai", "2024-09-09"),
    ("Mumbai-Delhi", "2024-10-14"), ("Ahmedabad-Mumbai", "2024-08-19"),
    ("Chennai-Bangalore", "2024-07-08"), ("Delhi-Jaipur", "2025-04-07"),
    ("Mumbai-Pune", "2024-05-06"), ("Kolkata-Bhubaneswar", "2025-08-04"),
    ("Delhi-Chennai", "2025-09-08"), ("Mumbai-Delhi", "2025-11-03"),
    ("Ahmedabad-Mumbai", "2025-07-14"), ("Chennai-Bangalore", "2025-10-13"),
]
SPIKE_FACTORS = [1.10, 1.15, 1.20, 1.25, 1.30, 1.40]


def _flagged_keys(results: pd.DataFrame) -> set:
    return set(zip(results["route"], results["week_of"]))


def spike_eval(shipments, retriever, baseline_keys, cfg, factors=SPIKE_FACTORS) -> pd.DataFrame:
    rows = []
    columns = list(pd.read_csv(ROOT / "data" / TEMPLATE_FILE, nrows=0).columns)
    for factor in factors:
        df = shipments
        for route, week in SPIKES:
            df = inject(df, route, [week], [factor])
        results = explain_anomalies(analyse(df, cfg), retriever, columns, cfg)
        keys = _flagged_keys(results)
        injected = set(SPIKES)
        found = injected & keys
        verdict = results.set_index(["route", "week_of"])["flagged"]
        wrongly_justified = sum(1 for k in found if verdict[k] == "No (justified)")
        new_alarms = keys - baseline_keys - injected
        rows.append({"cost_rise": f"+{round((factor - 1) * 100)}%", "injected": len(injected),
                     "flagged": len(found), "recall": f"{len(found) / len(injected):.0%}",
                     "wrongly_justified": wrongly_justified, "new_false_alarms": len(new_alarms)})
    return pd.DataFrame(rows)


def scenario_eval(shipments, retriever, cfg) -> dict:
    """Two harder, realistic patterns: a short sustained step and a slow creep."""
    columns = list(pd.read_csv(ROOT / "data" / TEMPLATE_FILE, nrows=0).columns)
    base_keys = _flagged_keys(explain_anomalies(analyse(shipments, cfg), retriever, columns, cfg))
    out = {}

    step_weeks = ["2024-09-02", "2024-09-09", "2024-09-16", "2024-09-23"]
    df = inject(shipments, "Mumbai-Delhi", step_weeks, [1.25] * 4)
    res = explain_anomalies(analyse(df, cfg), retriever, columns, cfg)
    keys = _flagged_keys(res)
    out["step"] = {"description": "Mumbai-Delhi +25% for 4 consecutive weeks from 2024-09-02",
                   "weeks": [(w, ("Mumbai-Delhi", w) in keys) for w in step_weeks],
                   "new_false_alarms": len(keys - base_keys - {("Mumbai-Delhi", w) for w in step_weeks})}

    ramp_weeks = [(pd.Timestamp("2024-09-02") + pd.Timedelta(weeks=k)).strftime("%Y-%m-%d") for k in range(30)]
    ramp_factors = [1 + 0.012 * (k + 1) for k in range(30)]
    df = inject(shipments, "Kolkata-Bhubaneswar", ramp_weeks, ramp_factors)
    res = explain_anomalies(analyse(df, cfg), retriever, columns, cfg)
    keys = _flagged_keys(res)
    hit = [i for i, w in enumerate(ramp_weeks) if ("Kolkata-Bhubaneswar", w) in keys]
    out["ramp"] = {"description": "Kolkata-Bhubaneswar cost creeping up 1.2% per week for 30 weeks from 2024-09-02",
                   "first_flag_week_index": (hit[0] + 1) if hit else None,
                   "cumulative_rise_at_first_flag": f"+{(ramp_factors[hit[0]] - 1) * 100:.0f}%" if hit else None,
                   "weeks_flagged": len(hit), "of": len(ramp_weeks),
                   "new_false_alarms": len(keys - base_keys - {("Kolkata-Bhubaneswar", w) for w in ramp_weeks})}
    return out


# --------------------------------------------------------------------------- 4. sensitivity
SENSITIVITY_THRESHOLDS = [8, 10, 15, 20, 25, 30]


def sensitivity(shipments, retriever, columns, thresholds=SENSITIVITY_THRESHOLDS) -> pd.DataFrame:
    runs = {}
    for t in thresholds:
        cfg = replace(DEFAULT_CONFIG, own_threshold_pct=float(t), peer_threshold_pct=float(t))
        runs[t] = explain_anomalies(analyse(shipments, cfg), retriever, columns, cfg)
    ref = _flagged_keys(runs[20])
    rows = []
    for t, res in runs.items():
        keys = _flagged_keys(res)
        fmt = lambda ks: "; ".join(f"{r} {w}" for r, w in sorted(ks, key=lambda k: (k[1], k[0])))
        rows.append({"threshold_pct": t, "anomalies": len(res),
                     "justified": int((res["flagged"] == "No (justified)").sum()),
                     "unexplained": int((res["flagged"] == "Yes").sum()),
                     "added_vs_20": fmt(keys - ref) if t != 20 else "",
                     "removed_vs_20": fmt(ref - keys) if t != 20 else ""})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def run_all(root: Path = ROOT, cfg: Config = DEFAULT_CONFIG) -> dict:
    shipments = load_shipments(root / "data" / SHIPMENTS_FILE)
    notes = load_notes(root / "data" / NOTES_FILE)
    columns = load_template_columns(root / "data" / TEMPLATE_FILE)
    retriever = NoteRetriever(notes)
    results = explain_anomalies(analyse(shipments, cfg), retriever, columns, cfg)
    base_keys = _flagged_keys(results)

    sample = load_sample_output(root / "data" / TEMPLATE_FILE)
    sample_cmp = sample_row_check(results, sample)
    traps = run_traps(retriever, cfg)
    spikes = spike_eval(shipments, retriever, base_keys, cfg)
    scenarios = scenario_eval(shipments, retriever, cfg)
    sens = sensitivity(shipments, retriever, columns)

    (root / "logs").mkdir(exist_ok=True)
    sens.to_csv(root / "logs" / "sensitivity.csv", index=False, lineterminator="\n")

    step, ramp = scenarios["step"], scenarios["ramp"]
    n_rw = len(analyse(shipments, cfg))
    md = [
        "# Evaluation report", "",
        "Generated by `python -m src.evaluate`. Every number below is computed, not typed.", "",
        "## 1. Sample-row reproduction",
        f"{int(sample_cmp['match'].sum())}/{len(sample_cmp)} graded fields of the 3 sample rows match exactly "
        "(cost, both comparisons, flagged, matched_note_id).", "",
        _md_table(sample_cmp[~sample_cmp["match"]]) if (~sample_cmp["match"]).any() else "_No mismatches._", "",
        "## 2. Note-trap cases",
        f"{int(traps['pass'].sum())}/{len(traps)} cases correct. A case passes only if the cited note equals the "
        "expected one (blank when it must stay unexplained).", "",
        _md_table(traps), "",
        "## 3. Injection tests (planted cost rises in a copy of the data)", "",
        "### 3a. Single-week spikes (12 planted, different routes and weeks)",
        f"Context: on the untouched data {len(results)} of {n_rw} route-weeks are flagged "
        f"({len(results) / n_rw:.1%}).", "",
        _md_table(spikes), "",
        "`wrongly_justified` = planted rises that a note was (incorrectly) allowed to explain; "
        "`new_false_alarms` = flagged weeks that are neither planted nor flagged on the untouched data.", "",
        "### 3b. Sustained step", step["description"] + ".",
        "Flagged weeks: " + ", ".join(f"{w} {'yes' if f else 'NO'}" for w, f in step["weeks"]) +
        f". New false alarms elsewhere: {step['new_false_alarms']}.", "",
        "### 3c. Slow creep", ramp["description"] + ".",
        (f"First flagged in week {ramp['first_flag_week_index']} of the ramp, at a cumulative rise of "
         f"{ramp['cumulative_rise_at_first_flag']}; {ramp['weeks_flagged']} of {ramp['of']} ramp weeks flagged; "
         f"new false alarms elsewhere: {ramp['new_false_alarms']}."
         if ramp["first_flag_week_index"] else "Never flagged."), "",
        "## 4. Threshold sensitivity",
        "Own-history and peer thresholds moved together; the drift guard stays at 10%. "
        "`added_vs_20` / `removed_vs_20` show exactly which route-weeks change relative to the chosen 20%.", "",
        _md_table(sens), "",
    ]
    (root / "logs" / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    return {"sample": sample_cmp, "traps": traps, "spikes": spikes, "scenarios": scenarios, "sensitivity": sens}


if __name__ == "__main__":
    out = run_all()
    s, t = out["sample"], out["traps"]
    print(f"sample fields matched: {int(s['match'].sum())}/{len(s)}")
    print(f"note traps correct:    {int(t['pass'].sum())}/{len(t)}")
    print(out["spikes"].to_string(index=False))
    print(out["scenarios"]["step"]["weeks"], "| step false alarms:", out["scenarios"]["step"]["new_false_alarms"])
    print({k: v for k, v in out["scenarios"]["ramp"].items()})
    print(out["sensitivity"][["threshold_pct", "anomalies", "justified", "unexplained"]].to_string(index=False))
    print("full report: logs/eval_report.md")
