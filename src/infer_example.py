# -*- coding: utf-8 -*-
"""
线上推理示例 —— 单笔订单 ETA 打分
==================================

这个脚本和 delivery_eta.py 的分工：
    delivery_eta.py  离线训练：造数据、清洗、特征、训练、评估、出图
    本脚本           线上打分：加载模型包，对一笔"刚接单"的订单出 P50 / P90 与归因

线上打分与离线训练最大的差异，是**历史统计特征的来源**。
训练时用 pandas 在整张表上做 as-of 聚合；线上没有整张表，必须有一个
按天落库的特征快照（feature store），且只允许读「昨天及以前」的版本。

运行：
    python src/infer_example.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

import delivery_eta as D  # noqa: E402


class HistoryFeatureStore:
    """最简版特征快照。

    生产环境用 Redis/Redis-Bloom/离线数仓导出的天级快照替代，
    这里用内存字典演示"只读得到昨天及以前"的契约。

    核心约束：get(key, as_of_date) 在 as_of_date 当天及未来永远返回 None，
    即使快照里已经有当天的数据。这条约束是防时间穿越的最后一道闸。
    """

    def __init__(self, daily_snapshot: pd.DataFrame):
        # 期望列: stat_date, area, time_period, avg_duration, orders, riders
        self._snap = daily_snapshot.copy()
        self._snap["stat_date"] = pd.to_datetime(self._snap["stat_date"]).dt.normalize()
        self._snap = self._snap.sort_values("stat_date")
        self._key = self._snap.set_index(["area", "time_period", "stat_date"])

    def as_of(self, as_of_date: pd.Timestamp, area: str, period: str) -> dict:
        """返回 (area, period) 在 as_of_date 之前可用的一行统计。"""
        cutoff = pd.Timestamp(as_of_date).normalize()
        try:
            sub = self._key.loc[(area, period)]
        except KeyError:
            return {"area_hist_avg": None, "area_period_hist_avg": None,
                    "supply_pressure": None, "snapshot_date": None}
        sub = sub[sub.index < cutoff]           # 严格小于：当天及未来一律不可见
        if sub.empty:
            return {"area_hist_avg": None, "area_period_hist_avg": None,
                    "supply_pressure": None, "snapshot_date": None}
        last = sub.iloc[-1]
        return {
            "area_hist_avg": float(last["avg_duration"]),
            "area_period_hist_avg": float(last["avg_duration"]),
            "supply_pressure": float(last["orders"] / max(last["riders"], 1)),
            "snapshot_date": str(sub.index[-1].date()),
        }


def build_live_row(order: dict, store: HistoryFeatureStore) -> pd.DataFrame:
    """把一笔线上订单转成模型输入。字段名与离线训练保持一致。"""
    ts = pd.Timestamp(order["order_time"])
    minute_of_day = ts.hour * 60 + ts.minute
    # 线上无法保证城市片区库已覆盖所有新地址，这里按订单自带标签走，
    # 未知片区在训练时不会出现 → 走类别 "unknown"，由树模型的缺失分支兜底。
    area_pickup = order.get("area") or "unknown"
    area_drop = order.get("area_drop") or "unknown"
    period = order.get("time_period") or _period_of(ts.hour)
    snap = store.as_of(ts, area_pickup, period)

    row = {
        "distance_km": float(order["distance_km"]),
        "straight_km": float(order["straight_km"]),
        "detour_ratio": float(order["distance_km"] / max(order["straight_km"], 0.05)),
        "items": int(order["item_count"]),
        "load": int(order["rider_daily_orders"]),
        "congestion_level": int(order["congestion_level"]),
        "is_long_distance": int(order["distance_km"] >= 6),
        "has_rain": int(order["weather"] == "雨"),
        "hour": ts.hour,
        "weekday": ts.weekday(),
        "is_weekend": int(ts.weekday() >= 5),
        "minute_of_day": minute_of_day,
        "hour_sin": np.sin(2 * np.pi * minute_of_day / 1440),
        "hour_cos": np.cos(2 * np.pi * minute_of_day / 1440),
        "weekday_sin": np.sin(2 * np.pi * ts.weekday() / 7),
        "weekday_cos": np.cos(2 * np.pi * ts.weekday() / 7),
        "peak_affinity": float(D._peak_affinity(np.array([minute_of_day]))[0]),
        "is_peak": int(period in ("早高峰", "午高峰", "晚高峰")),
        "is_late_night": int(period == "凌晨"),
        "day_of_month": ts.day,
        "is_month_start_week": int(ts.day <= 7),
        "same_area": int(area_pickup == area_drop),
        "area_hist_avg": snap["area_hist_avg"],
        "area_period_hist_avg": snap["area_period_hist_avg"],
        "supply_pressure": snap["supply_pressure"],
        "time_period": period,
        "weather": order["weather"],
        "merchant_type": order["merchant_type"],
        "area_pickup": area_pickup,
        "area_drop": area_drop,
    }
    df = pd.DataFrame([row])
    for c in D.FEATURE_COLS_CAT:
        df[c] = df[c].astype("category")
    for c in D.FEATURE_COLS_NUM:
        df[c] = df[c].astype(float)
    return df


def _period_of(hour: int) -> str:
    if hour < 6:
        return "凌晨"
    if 7 <= hour < 10:
        return "早高峰"
    if 11 <= hour < 14:
        return "午高峰"
    if 17 <= hour < 20:
        return "晚高峰"
    return "平峰"


def score(order: dict, bundle: dict, store: HistoryFeatureStore) -> dict:
    """返回线上决策需要的全部字段。"""
    X = build_live_row(order, store)
    model, q90 = bundle["model"], bundle["quantile"]
    delta = bundle["p90_delta"]

    p50 = float(model.predict(X)[0])
    p90 = float(q90.predict(X)[0]) + delta
    # 分位数模型是独立训练的，可能出现 P90 < P50 的交叉；单调修正
    p90 = max(p90, p50)
    risk_th = bundle["risk_threshold"]

    contrib = model.predict(X, pred_contrib=True)[0]
    names = model.feature_name()
    parts = (pd.DataFrame({"feature": names, "contrib": contrib[:-1]})
             .assign(cn=lambda d: d["feature"].map(lambda x: D.FEATURE_CN.get(x, x)),
                     abs=lambda d: d["contrib"].abs())
             .sort_values("abs", ascending=False)
             .head(4))
    drivers = [{"name": r.cn, "minutes": round(float(r.contrib), 2)}
               for _, r in parts.iterrows()]

    return {
        "eta_p50_min": round(p50, 1),
        "eta_p90_min": round(p90, 1),
        "over_sla_prob_proxy": round(float(p90 > risk_th), 3),
        "risk_flagged": bool(p90 > risk_th),
        "sla_min": bundle["sla"],
        "drivers": drivers,
        "snapshot": store.as_of(pd.Timestamp(order["order_time"]),
                                order["area"], order.get("time_period") or
                                _period_of(pd.Timestamp(order["order_time"]).hour))["snapshot_date"],
    }


def main():
    bundle = joblib.load(BASE_DIR / "outputs" / "eta_bundle.joblib")

    # 用离线数据造一份"天级快照"，模拟数仓 T+1 的可读数据
    raw = D.simulate_orders()
    clean, _ = D.clean_orders(raw, verbose=False)
    feat = D.build_features(clean)
    snap = (feat.groupby(["order_date", "area_pickup", "time_period"])
            .agg(avg_duration=("duration_min", "mean"),
                 orders=("order_id", "count"),
                 riders=("rider_daily_orders", "mean"))
            .reset_index()
            .rename(columns={"order_date": "stat_date", "area_pickup": "area"}))
    store = HistoryFeatureStore(snap)

    demo = {
        "order_time": "2026-02-27 18:32:00",
        "area": "老城区", "area_drop": "写字楼区",
        "pickup_lat": 31.2280, "pickup_lon": 121.4550,
        "drop_lat": 31.2205, "drop_lon": 121.4900,
        "distance_km": 5.8, "straight_km": 2.3,
        "time_period": "晚高峰", "weather": "雨", "merchant_type": "正餐",
        "item_count": 4, "rider_daily_orders": 28, "congestion_level": 3,
    }
    out = score(demo, bundle, store)
    print("单笔订单线上打分")
    print("-" * 46)
    for k in ("eta_p50_min", "eta_p90_min", "risk_flagged", "sla_min",
              "snapshot", "drivers"):
        print(f"  {k:<20} {out[k]}")
    print("\n说明：p90 > 风险阈值 → 打标推送运营；drivers 是 SHAP 贡献（分钟）。")
    return out


if __name__ == "__main__":
    main()
