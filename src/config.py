"""All tunable policy in one place.

The brief fixes the two comparison baselines (8-week own history, same-week
peers). It does NOT prescribe an anomaly threshold, so the numbers below are
our own configurable policy, evaluated in logs/sensitivity.csv.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- comparison baselines (fixed by the brief) ---
    history_weeks: int = 8          # trailing window, strictly before current week

    # --- flag policy (OUR choice; see README + sensitivity table) ---
    own_threshold_pct: float = 20.0   # rise vs own trailing average that is anomalous on its own
    peer_threshold_pct: float = 20.0  # gap vs similar routes that is anomalous ...
    drift_weeks: int = 26             # ... if the route is also drifting up vs this many weeks ago
    drift_min_pct: float = 10.0       # minimum long-term drift for a peer-driven flag

    # --- note evidence policy ---
    proximity_days: int = 28          # "nearby" notes mentioned in unexplained reasons
    max_nearby_notes: int = 2
    magnitude_tolerance: float = 1.5  # observed rise may be up to 1.5x a note's stated max effect

    # --- retrieval ---
    retrieval_top_k: int = 10


DEFAULT_CONFIG = Config()
