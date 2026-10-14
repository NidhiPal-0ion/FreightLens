"""Numeric anomaly decision. Deliberately knows nothing about the notes."""
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


def flag_anomalies(w: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Mark route-weeks whose cost is rising AND out of the ordinary.

    rising        : cost above the route's own trailing average (vs_own_pct > 0).
    own-driven    : rising and vs_own_pct >= own_threshold_pct.
    peer-driven   : rising, vs_peer_pct >= peer_threshold_pct, and the route is also
                    drifting up long-term (drift >= drift_min_pct). If 26 weeks of
                    history do not exist yet, drift is unavailable and the peer-driven
                    flag relies on 'rising' alone (documented assumption).
    A peer gap on its own is never enough: a route can be pricier than its peers
    without its cost rising.
    """
    out = w.copy()
    rising = out["vs_own_pct"] > 0                                   # NaN -> False
    own_driven = rising & (out["vs_own_pct"] >= cfg.own_threshold_pct)
    drift_ok = out["drift_pct"].isna() | (out["drift_pct"] >= cfg.drift_min_pct)
    peer_driven = rising & (out["vs_peer_pct"] >= cfg.peer_threshold_pct) & drift_ok

    out["is_anomaly"] = own_driven | peer_driven
    out["trigger"] = np.select(
        [own_driven & peer_driven, own_driven, peer_driven],
        ["own_history+peer_gap", "own_history", "peer_gap+drift"],
        default="",
    )
    return out
