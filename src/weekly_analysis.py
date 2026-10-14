"""Weekly cost per tonne-km and the two baselines the brief defines exactly."""
import numpy as np
import pandas as pd

from .config import Config, DEFAULT_CONFIG


def add_week_of(df: pd.DataFrame) -> pd.DataFrame:
    """Weeks run Monday-Sunday; week_of is that week's Monday."""
    out = df.copy()
    out["week_of"] = out["shipment_date"] - pd.to_timedelta(out["shipment_date"].dt.weekday, unit="D")
    return out


def weekly_costs(shipments: pd.DataFrame) -> pd.DataFrame:
    """Cost per tonne-km per (route, week) = total cost / total (tonnes x km).

    A ratio of sums, exactly as in the brief, not a mean of per-shipment ratios.
    """
    df = add_week_of(shipments)
    df["tonne_km"] = df["quantity_tonnes"] * df["distance_km"]
    weekly = (
        df.groupby(["route", "route_type", "week_of"], as_index=False)
        .agg(total_cost=("freight_cost_inr", "sum"),
             tonne_km=("tonne_km", "sum"),
             n_shipments=("shipment_id", "count"))
    )
    weekly["cost_per_tonne_km"] = weekly["total_cost"] / weekly["tonne_km"]
    return weekly.sort_values(["route", "week_of"]).reset_index(drop=True)


def add_baselines(weekly: pd.DataFrame, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Attach own-history, peer and long-term-drift comparisons.

    vs own history : mean of the route's previous `history_weeks` weekly values,
                     strictly before the current week (shift(1)); with fewer prior
                     weeks it uses all that exist (min_periods=1); week 1 has none (NaN).
    vs similar     : mean of the OTHER routes with the same route_type in the same week.
    drift          : change vs the same route `drift_weeks` earlier (extra signal for
                     slow creeps that a trailing average hides).
    """
    w = weekly.sort_values(["route", "week_of"]).reset_index(drop=True).copy()
    cpt = w.groupby("route")["cost_per_tonne_km"]

    w["own_history_avg"] = cpt.transform(lambda s: s.shift(1).rolling(cfg.history_weeks, min_periods=1).mean())
    w["n_prior_weeks"] = w.groupby("route").cumcount().clip(upper=cfg.history_weeks)

    grp = w.groupby(["route_type", "week_of"])["cost_per_tonne_km"]
    w["_peer_sum"] = grp.transform("sum")
    w["_peer_n"] = grp.transform("count")
    w["peer_avg"] = np.where(w["_peer_n"] > 1,
                             (w["_peer_sum"] - w["cost_per_tonne_km"]) / (w["_peer_n"] - 1),
                             np.nan)
    w = w.drop(columns=["_peer_sum", "_peer_n"])

    w["vs_own_pct"] = (w["cost_per_tonne_km"] / w["own_history_avg"] - 1) * 100
    w["vs_peer_pct"] = (w["cost_per_tonne_km"] / w["peer_avg"] - 1) * 100
    w["cost_n_weeks_ago"] = cpt.shift(cfg.drift_weeks)
    w["drift_pct"] = (w["cost_per_tonne_km"] / w["cost_n_weeks_ago"] - 1) * 100
    return w
