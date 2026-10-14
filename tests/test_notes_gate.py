"""The evidence gate: wrong route / wrong window / 'no impact' notes must never justify a rise."""
from datetime import date

import pandas as pd
import pytest

from src.data_loader import load_notes
from src.evaluate import TRAPS, run_traps
from src.explain import decide
from src.notes import NOTE_META, evaluate_note, validate_metadata
from src.pipeline import ROOT
from src.retrieval import NoteRetriever


@pytest.fixture(scope="module")
def notes():
    return load_notes(ROOT / "data" / "context_notes.csv")


@pytest.fixture(scope="module")
def retriever(notes):
    return NoteRetriever(notes)


def test_metadata_matches_notes_csv(notes):
    validate_metadata(notes)          # raises if ids, dates or scope drift apart


def test_every_labelled_trap_case_is_correct(retriever):
    result = run_traps(retriever)
    failures = result[~result["pass"]]
    assert failures.empty, f"\n{failures.to_string()}"
    assert len(result) == len(TRAPS) >= 20


@pytest.mark.parametrize("note_id", ["N004", "N005", "N006", "N007", "N008", "N009", "N010"])
def test_non_increase_notes_never_justify_anything(note_id):
    meta = NOTE_META[note_id]
    routes = ["Mumbai-Pune", "Delhi-Jaipur", "Ahmedabad-Mumbai", "Chennai-Bangalore",
              "Kolkata-Bhubaneswar", "Mumbai-Delhi", "Delhi-Chennai"]
    for route in routes:
        for week in pd.date_range("2024-01-01", "2025-12-22", freq="7D"):
            assert not evaluate_note(meta, route, week.date(), 5.0).ok


def test_flood_note_stops_at_end_of_window():
    n1 = NOTE_META["N001"]
    assert evaluate_note(n1, "Chennai-Bangalore", date(2025, 3, 3), 20).ok        # Mar 3-9 overlaps Feb 24 - Mar 8
    assert not evaluate_note(n1, "Chennai-Bangalore", date(2025, 3, 10), 20).ok   # week after
    assert not evaluate_note(n1, "Chennai-Bangalore", date(2025, 2, 17), 20).ok   # week before
    assert not evaluate_note(n1, "Delhi-Jaipur", date(2025, 2, 24), 20).ok        # wrong route


def test_diesel_note_respects_magnitude_and_window():
    n3 = NOTE_META["N003"]
    assert evaluate_note(n3, "Mumbai-Delhi", date(2025, 5, 12), 6.0).ok
    assert not evaluate_note(n3, "Mumbai-Delhi", date(2025, 5, 12), 11.6).ok      # far above stated 5-7%
    assert not evaluate_note(n3, "Mumbai-Delhi", date(2025, 4, 28), 6.0).ok       # before it started
    assert not evaluate_note(n3, "Mumbai-Delhi", date(2025, 6, 30), 6.0).ok       # after assumed window


def test_unexplained_rows_never_cite_a_note(retriever):
    d = decide("Delhi-Chennai", date(2024, 11, 11), 25.0, retriever)
    assert d.note_id == "" and not d.justified
    assert "unexplained" in d.reason.lower()


def test_retrieval_is_deterministic(retriever, notes):
    q = "Chennai Bangalore freight cost increase week of 2025-03-03"
    assert retriever.rank(q) == NoteRetriever(notes).rank(q)
