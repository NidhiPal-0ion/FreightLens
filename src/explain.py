"""Turn a flagged route-week into a verdict, a cited note and a plain-English reason.

Flow: retrieval proposes ranked candidates -> the gate in notes.py accepts or rejects each
-> the first accepted note justifies the row. If none is accepted the row is UNEXPLAINED,
matched_note_id stays blank, and the reason may name the nearest notes that were rejected
(and why). Wording comes from fixed templates, never from a free-text model, so the
verdict, cited note and numbers cannot vary between runs and cannot be invented.
"""
from dataclasses import dataclass
from datetime import date

from .config import Config, DEFAULT_CONFIG
from .notes import NOTE_META, NoteMeta, GateResult, evaluate_note, gap_days, ALL
from .retrieval import NoteRetriever, build_query


@dataclass(frozen=True)
class Decision:
    justified: bool
    note_id: str       # "" when unexplained
    reason: str


def _fmt_window(meta: NoteMeta) -> str:
    if meta.start == meta.end:
        return f"dated {meta.start}"
    return f"covering {meta.start} to {meta.end}"


def _why_rejected(meta: NoteMeta, res: GateResult, week_start: date) -> str:
    if res.code == "no_cost_increase":
        if meta.effect == "not_applicable":
            return f"{meta.note_id} (dated {meta.start}) says {meta.gist}, so it does not apply here"
        return f"{meta.note_id} (dated {meta.start}) says {meta.gist}, so it does not describe a reason for a cost rise"
    if res.code == "outside_window":
        return (f"{meta.note_id} ({_fmt_window(meta)}) says {meta.gist}, "
                f"but its period does not include the week of {week_start}")
    if res.code == "magnitude_exceeds":
        return (f"{meta.note_id} (dated {meta.start}) says {meta.gist}, "
                f"but the rise seen here is much larger than it describes ({res.detail})")
    return f"{meta.note_id} does not apply"


def decide(route: str, week_start: date, vs_own_pct: float, retriever: NoteRetriever,
           cfg: Config = DEFAULT_CONFIG) -> Decision:
    ranked = retriever.rank(build_query(route, week_start))[: cfg.retrieval_top_k]

    # 1) first retrieved note that passes every gate justifies the row
    for note_id, _score in ranked:
        meta = NOTE_META[note_id]
        if evaluate_note(meta, route, week_start, vs_own_pct, cfg).ok:
            return Decision(
                True, note_id,
                f"Matches note {note_id} dated {meta.start}: {meta.gist} "
                f"(effect window {meta.start} to {meta.end}). The cost rise has a clear explanation.",
            )

    # 2) nothing passes -> unexplained; mention nearby notes that apply to this route and were rejected
    nearby = []
    for rank_pos, (note_id, _score) in enumerate(ranked):
        meta = NOTE_META[note_id]
        if not (meta.scope == ALL or route in meta.scope):
            continue
        gap = gap_days(meta, week_start)
        if gap <= cfg.proximity_days:
            nearby.append((gap, rank_pos, note_id))
    nearby.sort()
    nearby = nearby[: cfg.max_nearby_notes]

    if not nearby:
        return Decision(False, "", "No matching note found for this route or date range. "
                                   "Cost rise looks unexplained and worth a human review.")
    parts = [_why_rejected(NOTE_META[nid], evaluate_note(NOTE_META[nid], route, week_start, vs_own_pct, cfg), week_start)
             for _gap, _pos, nid in nearby]
    if len(parts) == 1:
        closest = f"Closest note checked: {parts[0]}."
    else:
        closest = f"Closest notes checked: (1) {parts[0]}; (2) {parts[1]}."
    return Decision(False, "", f"{closest} No genuine justification found; flagged for review.")
