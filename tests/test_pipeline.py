"""End-to-end checks on the real data and the output contract."""
import hashlib

import pandas as pd
import pytest

from src.data_loader import load_notes, load_sample_output, load_shipments, load_template_columns, validate_shipments
from src.evaluate import SAMPLE_FIELDS, sample_row_check
from src.notes import NOTE_META, evaluate_note
from src.pipeline import ROOT, run


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("run1")
    metrics = run(ROOT / "data", out / "outputs", out / "logs")
    return out, metrics


def test_input_data_is_clean():
    report = validate_shipments(load_shipments(ROOT / "data" / "shipment_records.csv"))
    assert report["rows"] == 2940 and report["routes"] == 7 and report["weeks"] == 104
    assert report["route_week_gaps"] == {}


def test_output_columns_match_template_exactly(outputs):
    out, _ = outputs
    expected = load_template_columns(ROOT / "data" / "sample_output_format_v2.csv")
    got = pd.read_csv(out / "outputs" / "results.csv")
    assert list(got.columns) == expected


def test_three_sample_rows_reproduced_exactly(outputs):
    out, _ = outputs
    results = pd.read_csv(out / "outputs" / "results.csv", dtype=str)
    sample = load_sample_output(ROOT / "data" / "sample_output_format_v2.csv")
    cmp = sample_row_check(results, sample)
    assert cmp["match"].all(), cmp[~cmp["match"]].to_string()
    assert len(cmp) == len(sample) * len(SAMPLE_FIELDS)


def test_output_is_sorted_and_flags_use_contract_values(outputs):
    out, _ = outputs
    df = pd.read_csv(out / "outputs" / "results.csv", dtype=str)
    assert list(df["week_of"]) == sorted(df["week_of"])
    assert set(df["flagged"]) <= {"Yes", "No (justified)"}
    assert (df["matched_note_id"].fillna("") != "").tolist() == (df["flagged"] == "No (justified)").tolist()


def test_every_justified_row_passes_the_gate_independently(outputs):
    """Anti-hallucination audit: re-check each cited note from scratch."""
    out, _ = outputs
    df = pd.read_csv(out / "outputs" / "results.csv", dtype=str).fillna("")
    notes = load_notes(ROOT / "data" / "context_notes.csv").set_index("note_id")
    weekly = pd.read_csv(out / "outputs" / "weekly_metrics.csv").set_index(["route", "week_of"])
    justified = df[df["flagged"] == "No (justified)"]
    assert len(justified) >= 1
    for _, r in justified.iterrows():
        nid = r["matched_note_id"]
        meta = NOTE_META[nid]
        own = weekly.loc[(r["route"], r["week_of"]), "vs_own_pct"]
        assert evaluate_note(meta, r["route"], pd.Timestamp(r["week_of"]).date(), own).ok
        assert nid in r["reason"] and str(notes.loc[nid, "date"].date()) in r["reason"]


def test_costs_in_output_match_weekly_metrics(outputs):
    out, _ = outputs
    res = pd.read_csv(out / "outputs" / "results.csv")
    weekly = pd.read_csv(out / "outputs" / "weekly_metrics.csv").set_index(["route", "week_of"])
    for _, r in res.iterrows():
        assert r["cost_per_tonne_km"] == pytest.approx(weekly.loc[(r["route"], r["week_of"]), "cost_per_tonne_km"], abs=0.005)


def test_two_runs_are_byte_identical(outputs, tmp_path):
    first, _ = outputs
    run(ROOT / "data", tmp_path / "outputs", tmp_path / "logs")
    for name in ("results.csv", "weekly_metrics.csv"):
        a = hashlib.sha256((first / "outputs" / name).read_bytes()).hexdigest()
        b = hashlib.sha256((tmp_path / "outputs" / name).read_bytes()).hexdigest()
        assert a == b, name


def test_no_llm_is_used(outputs):
    _, metrics = outputs
    assert metrics["llm_calls"] == 0 and metrics["llm_input_tokens"] == 0 and metrics["llm_output_tokens"] == 0
