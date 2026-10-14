"""Format results exactly like sample_output_format_v2.csv."""
import pandas as pd

from .explain import Decision


def fmt_pct(value: float) -> str:
    """Signed, one decimal, never '-0.0'."""
    if pd.isna(value):
        return "n/a"
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:+.1f}%"


def own_history_text(value: float) -> str:
    return "n/a (no prior weeks)" if pd.isna(value) else f"{fmt_pct(value)} vs this route's past average"


def peer_text(value: float) -> str:
    return "n/a (no peer routes)" if pd.isna(value) else f"{fmt_pct(value)} vs similar-length routes this week"


def result_row(row, decision: Decision) -> dict:
    return {
        "route": row.route,
        "week_of": row.week_of.strftime("%Y-%m-%d"),
        "cost_per_tonne_km": f"{row.cost_per_tonne_km:.2f}",
        "vs_own_history": own_history_text(row.vs_own_pct),
        "vs_similar_routes": peer_text(row.vs_peer_pct),
        # "Yes" = anomaly with no valid explanation; "No (justified)" = anomaly backed by a note
        "flagged": "No (justified)" if decision.justified else "Yes",
        "matched_note_id": decision.note_id,
        "reason": decision.reason,
    }


def results_frame(rows: list, columns: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["week_of", "route"], kind="stable").reset_index(drop=True)
