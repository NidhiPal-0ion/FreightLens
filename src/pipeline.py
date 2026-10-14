"""End-to-end run:  python -m src.pipeline [--data-dir data] [--out-dir outputs] [--logs-dir logs]"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .anomaly_detection import flag_anomalies
from .config import Config, DEFAULT_CONFIG
from .data_loader import load_notes, load_shipments, load_template_columns, validate_shipments
from .explain import decide
from .notes import validate_metadata
from .reporting import result_row, results_frame
from .retrieval import NoteRetriever
from .weekly_analysis import add_baselines, weekly_costs

ROOT = Path(__file__).resolve().parents[1]
SHIPMENTS_FILE = "shipment_records.csv"
NOTES_FILE = "context_notes.csv"
TEMPLATE_FILE = "sample_output_format_v2.csv"

WEEKLY_COLUMNS = ["route", "route_type", "week_of", "total_cost", "tonne_km", "n_shipments",
                  "cost_per_tonne_km", "own_history_avg", "n_prior_weeks", "peer_avg",
                  "vs_own_pct", "vs_peer_pct", "drift_pct", "is_anomaly", "trigger"]


def analyse(shipments: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Weekly metrics + numeric anomaly flags for every route-week."""
    return flag_anomalies(add_baselines(weekly_costs(shipments), cfg), cfg)


def explain_anomalies(weekly: pd.DataFrame, retriever: NoteRetriever, columns: list,
                      cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    rows = []
    for row in weekly[weekly["is_anomaly"]].itertuples():
        decision = decide(row.route, row.week_of.date(), float(row.vs_own_pct), retriever, cfg)
        rows.append(result_row(row, decision))
    return results_frame(rows, columns)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")


def run(data_dir=ROOT / "data", out_dir=ROOT / "outputs", logs_dir=ROOT / "logs",
        cfg: Config = DEFAULT_CONFIG) -> dict:
    t0 = time.perf_counter()
    data_dir, out_dir, logs_dir = Path(data_dir), Path(out_dir), Path(logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    shipments = load_shipments(data_dir / SHIPMENTS_FILE)
    validation = validate_shipments(shipments)
    notes = load_notes(data_dir / NOTES_FILE)
    validate_metadata(notes)
    columns = load_template_columns(data_dir / TEMPLATE_FILE)   # the header is the contract

    weekly = analyse(shipments, cfg)
    retriever = NoteRetriever(notes)
    results = explain_anomalies(weekly, retriever, columns, cfg)

    write_csv(results, out_dir / "results.csv")
    weekly_out = weekly[WEEKLY_COLUMNS].copy()
    weekly_out["week_of"] = weekly_out["week_of"].dt.strftime("%Y-%m-%d")
    weekly_out.to_csv(out_dir / "weekly_metrics.csv", index=False, lineterminator="\n", float_format="%.6f")

    metrics = {
        "input": validation,
        "route_weeks": int(len(weekly)),
        "anomalies": int(len(results)),
        "justified": int((results["flagged"] == "No (justified)").sum()),
        "unexplained": int((results["flagged"] == "Yes").sum()),
        # Cost log: reasons are filled from templates and retrieval is TF-IDF, so no LLM is called.
        "llm_calls": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "estimated_llm_cost_usd": 0.0,
        "retrieval_backend": "tfidf (scikit-learn), local, deterministic",
        "runtime_seconds": round(time.perf_counter() - t0, 3),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (logs_dir / "run_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="FreightTiger shipping-cost watcher")
    parser.add_argument("--data-dir", default=ROOT / "data")
    parser.add_argument("--out-dir", default=ROOT / "outputs")
    parser.add_argument("--logs-dir", default=ROOT / "logs")
    args = parser.parse_args()
    metrics = run(args.data_dir, args.out_dir, args.logs_dir)
    print(f"{metrics['anomalies']} anomalies: {metrics['justified']} justified, "
          f"{metrics['unexplained']} unexplained -> {Path(args.out_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()
