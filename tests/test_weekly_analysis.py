"""Unit tests for the weekly engine, using tiny hand-checkable data."""
import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.weekly_analysis import add_baselines, add_week_of, weekly_costs


def _ship(route, date, qty, km, cost, rtype="Short", sid=[0]):
    sid[0] += 1
    origin, destination = route.split("-")
    return {"shipment_id": f"S{sid[0]}", "origin": origin, "destination": destination, "route": route,
            "route_type": rtype, "material": "x", "quantity_tonnes": qty, "distance_km": km,
            "freight_cost_inr": cost, "shipment_date": pd.Timestamp(date), "transporter": "t"}


def test_week_of_is_monday_and_sunday_belongs_to_previous_monday():
    df = pd.DataFrame([_ship("A-B", "2025-01-05", 1, 1, 1),    # Sunday
                       _ship("A-B", "2025-01-06", 1, 1, 1),    # Monday
                       _ship("A-B", "2024-12-30", 1, 1, 1)])   # Monday
    out = add_week_of(df)
    assert list(out["week_of"].dt.strftime("%Y-%m-%d")) == ["2024-12-30", "2025-01-06", "2024-12-30"]
    assert (out["week_of"].dt.weekday == 0).all()


def test_weekly_cost_is_ratio_of_sums_not_mean_of_ratios():
    # shipment 1: 100 / (10*10) = 1.0 ; shipment 2: 900 / (10*100) = 0.9
    df = pd.DataFrame([_ship("A-B", "2025-01-06", 10, 10, 100), _ship("A-B", "2025-01-07", 10, 100, 900)])
    w = weekly_costs(df)
    assert len(w) == 1
    assert w.loc[0, "cost_per_tonne_km"] == pytest.approx(1000 / (100 + 1000))   # 0.909.., not 0.95


def _weekly(values_by_route, route_types=None):
    rows = []
    for route, values in values_by_route.items():
        for i, v in enumerate(values):
            rows.append({"route": route, "route_type": (route_types or {}).get(route, "Short"),
                         "week_of": pd.Timestamp("2024-01-01") + pd.Timedelta(weeks=i),
                         "cost_per_tonne_km": v})
    return pd.DataFrame(rows)


def test_own_history_uses_only_strictly_prior_weeks_and_max_8():
    values = list(range(1, 13))                      # 1..12
    w = add_baselines(_weekly({"A-B": values}))
    w = w.set_index("week_of")["own_history_avg"]
    assert np.isnan(w.iloc[0])                                   # week 1: no history
    assert w.iloc[1] == 1.0                                      # week 2: only week 1
    assert w.iloc[3] == pytest.approx((1 + 2 + 3) / 3)           # fewer than 8 -> use what exists
    assert w.iloc[8] == pytest.approx(np.mean(range(1, 9)))      # week 9: weeks 1-8
    assert w.iloc[11] == pytest.approx(np.mean(range(4, 12)))    # week 12: weeks 4-11, current excluded


def test_no_lookahead_future_values_do_not_change_past_baselines():
    base = [2.0] * 10
    a = add_baselines(_weekly({"A-B": base + [9.0]}))
    b = add_baselines(_weekly({"A-B": base + [50.0]}))
    assert np.allclose(a["own_history_avg"].iloc[:10].fillna(-1), b["own_history_avg"].iloc[:10].fillna(-1))


def test_peer_average_excludes_the_route_itself_and_uses_same_type_only():
    w = add_baselines(_weekly({"A-B": [3.0], "C-D": [5.0], "E-F": [7.0], "G-H": [100.0]},
                              {"A-B": "Short", "C-D": "Short", "E-F": "Short", "G-H": "Long"}))
    got = w.set_index("route")["peer_avg"]
    assert got["A-B"] == pytest.approx((5.0 + 7.0) / 2)
    assert got["C-D"] == pytest.approx((3.0 + 7.0) / 2)
    assert np.isnan(got["G-H"])                      # only Long route -> no peers, no invented value


def test_percent_deviations():
    w = add_baselines(_weekly({"A-B": [2.0, 2.0, 3.0], "C-D": [2.0, 2.0, 2.0]}))
    last = w[(w["route"] == "A-B")].iloc[-1]
    assert last["vs_own_pct"] == pytest.approx((3.0 / 2.0 - 1) * 100)
    assert last["vs_peer_pct"] == pytest.approx((3.0 / 2.0 - 1) * 100)


def test_drift_compares_with_same_route_n_weeks_earlier():
    cfg = Config(drift_weeks=3)
    w = add_baselines(_weekly({"A-B": [1.0, 2.0, 3.0, 4.0]}), cfg)
    assert w["drift_pct"].iloc[3] == pytest.approx((4.0 / 1.0 - 1) * 100)
    assert w["drift_pct"].iloc[:3].isna().all()
