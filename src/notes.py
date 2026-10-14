"""Note metadata and the evidence gate.

The gate is what makes a "justified" verdict trustworthy: retrieval may *suggest*
a note, but only these deterministic rules can accept it.

NOTE_META is hand-written and hand-checked against context_notes.csv (the loader
cross-checks ids, dates and scope so the table cannot silently drift from the file).
Polarity values:
    increase               the note describes a cause of HIGHER costs
    no_material_increase   the note says costs/conditions were not materially affected
    normalization          the note says conditions returned to / are normal or improved
    not_applicable         the note does not apply to any route in this dataset
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Tuple, Union

import pandas as pd

from .config import Config, DEFAULT_CONFIG

ALL = "ALL"


@dataclass(frozen=True)
class NoteMeta:
    note_id: str
    scope: Union[str, Tuple[str, ...]]      # ALL, or tuple of route names; () = applies to no route
    start: date                             # first day the described effect applies
    end: date                               # last day the described effect applies
    effect: str                             # polarity, see module docstring
    gist: str                               # short clause restating the note (used in reasons)
    max_effect_pct: Optional[float] = None  # the note's stated size of effect, if it gives one


NOTE_META = {m.note_id: m for m in [
    NoteMeta("N001", ("Chennai-Bangalore",), date(2025, 2, 24), date(2025, 3, 8), "increase",
             "heavy flooding on the Chennai-Bangalore highway forced detours and raised trip costs"),
    NoteMeta("N002", ("Ahmedabad-Mumbai",), date(2025, 1, 20), date(2025, 1, 26), "increase",
             "a regional festival week brought a temporary transporter surcharge on this corridor"),
    # ASSUMPTION: N003 gives a start date but no end. A step change in costs only shows up as a
    # "rise vs own history" while the trailing 8-week baseline still pre-dates it, so we let the
    # note explain rises for 8 weeks (2025-05-05 .. 2025-06-29). Later rises are not blamed on it.
    NoteMeta("N003", ALL, date(2025, 5, 5), date(2025, 6, 29), "increase",
             "diesel prices rose nationwide, pushing costs up roughly 5-7% on all routes",
             max_effect_pct=7.0),
    NoteMeta("N004", (), date(2024, 3, 11), date(2024, 3, 11), "not_applicable",
             "a new toll plaza was commissioned but the affected routes are not in this dataset"),
    NoteMeta("N005", ("Mumbai-Delhi",), date(2024, 7, 29), date(2024, 8, 4), "no_material_increase",
             "highway maintenance caused minor delays but costs were not significantly affected"),
    NoteMeta("N006", ALL, date(2025, 9, 22), date(2025, 9, 22), "no_material_increase",
             "freight demand stayed stable with no major disruptions"),
    NoteMeta("N007", ("Delhi-Jaipur",), date(2024, 5, 20), date(2024, 5, 20), "normalization",
             "road conditions improved after resurfacing was completed"),
    NoteMeta("N008", ("Kolkata-Bhubaneswar",), date(2025, 6, 9), date(2025, 6, 9), "no_material_increase",
             "no significant disruptions were reported and freight movement stayed normal"),
    NoteMeta("N009", ("Chennai-Bangalore",), date(2025, 3, 17), date(2025, 3, 17), "normalization",
             "the route returned to normal conditions after flood-related repairs"),
    NoteMeta("N010", ALL, date(2025, 10, 27), date(2025, 10, 27), "no_material_increase",
             "a vehicle tracking mandate's compliance costs were absorbed without a rate change"),
]}


def validate_metadata(notes_df: pd.DataFrame) -> None:
    """Fail loudly if the metadata table and the notes CSV disagree."""
    csv_ids = set(notes_df["note_id"])
    if csv_ids != set(NOTE_META):
        raise ValueError(f"note ids differ: csv-only={csv_ids - set(NOTE_META)}, meta-only={set(NOTE_META) - csv_ids}")
    for _, row in notes_df.iterrows():
        meta = NOTE_META[row["note_id"]]
        if row["date"].date() != meta.start:
            raise ValueError(f"{meta.note_id}: csv date {row['date'].date()} != metadata start {meta.start}")
        csv_all = row["applies_to"].strip().lower() == "all routes"
        if csv_all and meta.scope not in (ALL, ()):
            raise ValueError(f"{meta.note_id}: csv says All Routes but metadata scope is {meta.scope}")
        if not csv_all and meta.scope != (row["applies_to"].strip(),):
            raise ValueError(f"{meta.note_id}: csv scope '{row['applies_to']}' != metadata scope {meta.scope}")


@dataclass(frozen=True)
class GateResult:
    ok: bool
    code: str      # supports | out_of_scope | no_cost_increase | outside_window | magnitude_exceeds
    detail: str = ""


def _in_scope(meta: NoteMeta, route: str) -> bool:
    return meta.scope == ALL or route in meta.scope


def evaluate_note(meta: NoteMeta, route: str, week_start: date, vs_own_pct: float,
                  cfg: Config = DEFAULT_CONFIG) -> GateResult:
    """Can this note justify a cost rise on `route` in the week starting `week_start`?

    All four checks must pass. The order decides which failure is *reported*:
    scope, then polarity, then window, then magnitude.
    """
    week_end = week_start + timedelta(days=6)
    if not _in_scope(meta, route):
        return GateResult(False, "out_of_scope")
    if meta.effect != "increase":
        return GateResult(False, "no_cost_increase")
    if not (meta.start <= week_end and meta.end >= week_start):
        return GateResult(False, "outside_window")
    if meta.max_effect_pct is not None and vs_own_pct > meta.max_effect_pct * cfg.magnitude_tolerance:
        return GateResult(False, "magnitude_exceeds",
                          f"observed {vs_own_pct:+.1f}% vs stated ~{meta.max_effect_pct:g}%")
    return GateResult(True, "supports")


def gap_days(meta: NoteMeta, week_start: date) -> int:
    """Days between the week [Mon..Sun] and the note's effect window (0 if they overlap)."""
    week_end = week_start + timedelta(days=6)
    if meta.start > week_end:
        return (meta.start - week_end).days
    if meta.end < week_start:
        return (week_start - meta.end).days
    return 0
