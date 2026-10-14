"""Load and validate the three input files."""
from pathlib import Path

import pandas as pd

SHIPMENT_COLUMNS = [
    "shipment_id", "origin", "destination", "route_type", "material",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date", "transporter",
]


def load_shipments(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["shipment_date"])
    missing = [c for c in SHIPMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"shipment file is missing columns: {missing}")
    df["route"] = df["origin"].str.strip() + "-" + df["destination"].str.strip()
    return df


def validate_shipments(df: pd.DataFrame) -> dict:
    """Raise on anything that would silently corrupt the maths; return a report."""
    if df[SHIPMENT_COLUMNS].isna().any().any():
        raise ValueError("null values found in shipment data")
    if df["shipment_id"].duplicated().any():
        raise ValueError("duplicate shipment_id values")
    for col in ("quantity_tonnes", "distance_km", "freight_cost_inr"):
        if (df[col] <= 0).any():
            raise ValueError(f"non-positive values in {col}")
    types_per_route = df.groupby("route")["route_type"].nunique()
    if (types_per_route > 1).any():
        raise ValueError(f"routes with more than one route_type: {list(types_per_route[types_per_route > 1].index)}")

    # Weekly coverage: the 8-week / 26-week windows are positional, which is only
    # equal to "calendar weeks" if no route has a missing week. Check, don't assume.
    week_of = df["shipment_date"] - pd.to_timedelta(df["shipment_date"].dt.weekday, unit="D")
    all_weeks = pd.date_range(week_of.min(), week_of.max(), freq="7D")
    gaps = {}
    for route, weeks in week_of.groupby(df["route"]):
        missing = sorted(set(all_weeks) - set(weeks))
        if missing:
            gaps[route] = [d.strftime("%Y-%m-%d") for d in missing]
    return {
        "rows": int(len(df)),
        "routes": int(df["route"].nunique()),
        "weeks": int(len(all_weeks)),
        "date_min": df["shipment_date"].min().strftime("%Y-%m-%d"),
        "date_max": df["shipment_date"].max().strftime("%Y-%m-%d"),
        "route_week_gaps": gaps,
    }


def load_notes(path) -> pd.DataFrame:
    notes = pd.read_csv(path, parse_dates=["date"])
    required = {"note_id", "date", "applies_to", "note"}
    if not required.issubset(notes.columns):
        raise ValueError(f"notes file must have columns {sorted(required)}")
    if notes["note_id"].duplicated().any():
        raise ValueError("duplicate note_id values")
    return notes


def load_sample_output(path) -> pd.DataFrame:
    """Read sample_output_format_v2.csv tolerantly.

    One sample row has an UNQUOTED comma inside `reason` ("(N006, 2025-09-22)"), so a standard
    CSV parser sees 9 fields on that line. The first seven columns never contain commas, so we
    split each line at most 7 times and keep the rest as the reason. (Our own output quotes
    such fields properly, so it parses cleanly with any CSV reader.)
    """
    lines = [ln.rstrip("\r\n") for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    columns = lines[0].split(",")
    rows = [ln.split(",", len(columns) - 1) for ln in lines[1:]]
    return pd.DataFrame(rows, columns=columns)


def load_template_columns(path) -> list:
    """The sample output's header is the contract (names AND order)."""
    return list(pd.read_csv(Path(path), nrows=0).columns)
