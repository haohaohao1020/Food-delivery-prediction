# -*- coding: utf-8 -*-
"""
外卖配送时长预测 —— 端到端可运行示例
====================================================

业务定义
--------
预测时点 T0 = 订单生成 / 骑手接单时刻。
目标 y = 骑手接单 → 送达 的总时长（分钟）。
因此所有特征必须满足：T0 时刻在线上可获取，且不依赖于本单的最终结果。

本脚本流程：
    1. 模拟一份贴近真实的城市外卖订单数据（含脏数据、缺失、异常）
    2. 数据清洗（规则化、可审计）
    3. 特征工程（时间 / 空间 / 统计衍生，统计特征严格 as-of 计算）
    4. 时序切分（按天切分 + 隔离带 embargo）
    5. 多模型对比（规则基线 / 线性回归 / LightGBM / XGBoost / CatBoost / MLP）
    6. LightGBM 训练（MAE 目标 + 单调约束 + 分位数模型）
    7. 评估（回归指标 + 业务指标：超时识别）
    8. 可视化（真实值 vs 预测值、特征重要性、误差分布、分时段误差）
    9. 导出一份 chart_data.json 供报告页使用

运行：python src/delivery_eta.py
依赖：pandas numpy scikit-learn lightgbm xgboost catboost matplotlib
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. 全局配置
# ----------------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

N_DAYS = 60                     # 模拟 60 天数据
START_DATE = "2026-01-01"
SLA_MINUTES = 45.0              # 城市级超时判定线（业务可配）
CITY_BBOX = (31.10, 31.36, 121.36, 121.60)  # lat_min, lat_max, lon_min, lon_max

# 外卖高峰期中心（一天中的分钟数），用于计算"距离高峰期还有多远"
PEAK_CENTERS = np.array([8 * 60, 12 * 60, 18 * 60])

# 可视化配色（与报告页保持一致的低饱和数据色板）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#eaf2fd", "#9ec5f4", "#5598e7", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "font.size": 10,
})


# ----------------------------------------------------------------------------
# 1. 数据模拟
# ----------------------------------------------------------------------------
AREAS = {
    # 片区: (中心纬度, 中心经度, 商家出餐难度, 配送难度)
    "中心商圈": (31.2330, 121.4750, 1.12, 1.15),
    "写字楼区": (31.2205, 121.4900, 1.05, 1.22),  # 电梯排队，末端耗时长
    "大学城": (31.2450, 121.4350, 0.95, 0.95),
    "老城区": (31.2280, 121.4550, 1.08, 1.10),    # 路窄、门牌难找
    "郊区住宅": (31.1700, 121.5200, 0.92, 1.00),
    "高新园区": (31.1950, 121.5050, 1.00, 1.05),
    "交通枢纽": (31.1900, 121.4400, 1.10, 1.18),
    "新区新城": (31.1450, 121.4850, 0.90, 1.02),
}
AREA_NAMES = list(AREAS.keys())
AREA_W = np.array([0.22, 0.16, 0.10, 0.12, 0.14, 0.10, 0.08, 0.08])
AREA_W = AREA_W / AREA_W.sum()

MERCHANT_TYPES = ["快餐", "奶茶", "正餐"]
MERCHANT_W = [0.45, 0.35, 0.20]
# 出餐基准等待时间（分钟）
MERCHANT_BASE_WAIT = {"快餐": 6.5, "奶茶": 4.0, "正餐": 11.0}

WEATHER_MULT_SPEED = {"晴": 1.00, "雨": 0.86, "大风": 0.93}
WEATHER_MULT_WAIT = {"晴": 1.00, "雨": 1.18, "大风": 1.05}
CONGESTION_MULT_SPEED = {0: 1.00, 1: 0.86, 2: 0.72, 3: 0.58}
CONGESTION_NAMES = {0: "畅通", 1: "缓行", 2: "拥堵", 3: "严重拥堵"}


def _km_to_deg(km_lat: float, km_lon: float) -> tuple[float, float]:
    """在纬度 31° 附近，把公里偏移换算成经纬度偏移。"""
    return km_lat / 111.0, km_lon / (111.0 * np.cos(np.deg2rad(31.2)))


def _haversine_km(lat1, lon1, lat2, lon2):
    """球面直线距离（公里），用于特征与数据校验。"""
    r = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _nearest_area(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    centers = np.array([AREAS[a][:2] for a in AREA_NAMES])
    d = _haversine_km(lat[:, None], lon[:, None], centers[None, :, 0], centers[None, :, 1])
    return np.array(AREA_NAMES, dtype=object)[d.argmin(axis=1)]


def simulate_orders(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """生成一份带脏数据的模拟订单表。

    设计原则：让"时长"由可观测因素 + 未观测噪声共同决定，
    这样模型既有可学习的结构，也不会被模拟成完全可解（真实业务同样如此）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_days, freq="D")

    # ---- 天气：按天做马尔可夫转移，保证"雨天成片出现" ----
    weather_by_day, cur = [], "晴"
    for _ in range(n_days):
        u = rng.random()
        if cur == "晴":
            cur = "雨" if u < 0.16 else ("大风" if u < 0.24 else "晴")
        elif cur == "雨":
            cur = "雨" if u < 0.55 else ("大风" if u < 0.62 else "晴")
        else:
            cur = "大风" if u < 0.45 else ("雨" if u < 0.55 else "晴")
        weather_by_day.append(cur)

    rows = []
    for i, d in enumerate(dates):
        weather = weather_by_day[i]
        weekday = d.weekday()
        is_weekend = int(weekday >= 5)
        # 城市级"当日扰动因子"：运力紧张/大促/临时管制等不可提前预知的整体波动。
        # 真实外卖数据里这类日级效应很强，也是"随机切分"最容易偷看的信息。
        day_factor = float(np.exp(rng.normal(0, 0.075)))
        if rng.random() < 0.06:          # 约 6% 的日子出现全城性异常
            day_factor *= float(rng.uniform(1.15, 1.30))
        # 周末订单量多 15%~20%，雨天订单量多 25%（宅家点单）
        base_vol = 1050 * (1.18 if is_weekend else 1.0) * (1.25 if weather == "雨" else 1.0)
        n_today = int(rng.normal(base_vol, base_vol * 0.08))
        n_today = max(n_today, 400)

        # 下单时刻：双峰分布（午 11-13、晚 17-19），凌晨稀疏
        hours = np.concatenate([
            rng.normal(12.0, 0.95, int(n_today * 0.40)),
            rng.normal(18.4, 1.10, int(n_today * 0.36)),
            rng.uniform(7.0, 10.5, int(n_today * 0.14)),
            rng.uniform(0.0, 6.0, int(n_today * 0.10)),
        ])
        hours = np.clip(hours, 0, 23.999)
        minute_of_day = (hours * 60).astype(int)
        order_time = d + pd.to_timedelta(minute_of_day, unit="m") + pd.to_timedelta(
            rng.integers(0, 60, size=len(minute_of_day)), unit="s"
        )
        order_time = pd.Series(order_time).sort_values().to_numpy()
        m = len(order_time)
        hour = pd.Series(pd.DatetimeIndex(order_time)).dt.hour.to_numpy()
        mod = hour * 60 + pd.Series(pd.DatetimeIndex(order_time)).dt.minute.to_numpy()

        # ---- 时段标签 ----
        period = np.select(
            [hour < 6, (hour >= 7) & (hour < 10), (hour >= 11) & (hour < 14),
             (hour >= 17) & (hour < 20)],
            ["凌晨", "早高峰", "午高峰", "晚高峰"],
            default="平峰",
        )
        is_peak = np.isin(period, ["早高峰", "午高峰", "晚高峰"]).astype(int)

        # ---- 默认拥堵等级：由时段 + 天气 + 周末 决定，再叠加随机 ----
        cong_base = (
            1.15 * is_peak
            + 0.9 * (period == "晚高峰")
            + 0.75 * (weather == "雨")
            + 0.35 * (weather == "大风")
            - 0.55 * (period == "凌晨")
            + rng.normal(0, 0.55, m)
        )
        congestion = np.clip(np.round(cong_base), 0, 3).astype(int)

        # 运力紧张：写进"骑手当日接单量"字段，并从 T0 起即可观测。
        # 注意：只把日级扰动的一部分（sqrt）映射进负载，剩下的部分不可观测，
        # 这正是真实场景——任何特征都无法完全解释"今天全城就是慢"。
        load = rng.gamma(4.0, 5.2, m) + 6.0 * is_peak + 3.0 * (weather == "雨")
        load = np.clip(np.round(load * np.sqrt(day_factor)), 0, 68).astype(int)

        # 片区级"当日出餐波动"：某商圈今天整体爆单/某商家设备故障，
        # 只有当天跑完一批单才能知道，任何 T0 时刻特征都无法提前获取。
        area_day_shock = {a: float(np.exp(rng.normal(0, 0.13))) for a in AREA_NAMES}

        # ---- 片区 / 商家 / 商品数 ----
        pickup_area = rng.choice(AREA_NAMES, size=m, p=AREA_W)
        merchant = rng.choice(MERCHANT_TYPES, size=m, p=MERCHANT_W)
        items = np.clip(np.round(rng.gamma(2.1, 1.05, m)), 1, 12).astype(int)

        # ---- 取餐点：在片区中心附近散开 ----
        pc = np.array([AREAS[a][:2] for a in pickup_area])
        dlat, dlon = _km_to_deg(rng.normal(0, 1.1, m), rng.normal(0, 1.1, m))
        pick_lat, pick_lon = pc[:, 0] + dlat, pc[:, 1] + dlon

        # ---- 距离与送餐点 ----
        dist = np.clip(rng.lognormal(mean=np.log(2.1), sigma=0.55, size=m), 0.3, 14.0)
        bearing = rng.uniform(0, 2 * np.pi, m)
        dkm_lat, dkm_lon = dist * np.sin(bearing), dist * np.cos(bearing)
        ddlat, ddlon = _km_to_deg(dkm_lat, dkm_lon)
        drop_lat, drop_lon = pick_lat + ddlat, pick_lon + ddlon
        drop_area = _nearest_area(drop_lat, drop_lon)
        straight = _haversine_km(pick_lat, pick_lon, drop_lat, drop_lon)

        # 路网距离 = 直线 × 绕行系数（老城区/枢纽更绕）
        detour = rng.normal(1.28, 0.10, m)
        detour = np.where(np.isin(pickup_area, ["老城区", "交通枢纽"]), detour + 0.08, detour)
        detour = np.clip(detour, 1.12, 1.65)
        road_km = straight * detour

        # ---- 时长生成（三段式：接驾 + 等餐 + 骑行 + 末端交付）----
        pick_difficulty = np.array([AREAS[a][2] for a in pickup_area])
        drop_difficulty = np.array([AREAS[a][3] for a in drop_area])

        # 1) 接单 → 到店
        t_accept = 1.8 + np.abs(rng.normal(0, 1.3, m)) * (1 + 0.022 * load)
        # 2) 到店 → 取餐（商家出餐）
        wait_base = np.array([MERCHANT_BASE_WAIT[x] for x in merchant])
        area_shock = np.array([area_day_shock[a] for a in pickup_area])
        t_wait = (
            wait_base
            * pick_difficulty
            * (1 + 0.10 * (items - 1))
            * (1 + 0.22 * is_peak)
            * WEATHER_MULT_WAIT[weather]
            * area_shock
            + rng.gamma(2.0, 1.5, m)
        )
        # 3) 取餐 → 送达（骑行速度受拥堵/天气/距离影响）
        speed = (
            19.0
            * np.array([CONGESTION_MULT_SPEED[c] for c in congestion])
            * WEATHER_MULT_SPEED[weather]
            * (1 + 0.22 * np.minimum(road_km, 7.0) / 7.0)   # 远单更多走主干道
            * (1 - 0.004 * np.minimum(load, 40))            # 挂单多则绕路
        )
        speed = np.clip(speed, 7.0, 32.0)
        t_ride = road_km / speed * 60.0
        # 4) 末端交付（找楼、电梯、上楼）
        t_deliver = (
            0.8
            + rng.lognormal(0.55, 0.45, m)
            + 1.4 * (drop_area == "写字楼区")
            + 1.0 * (drop_area == "老城区")
        )

        duration = t_accept + t_wait + t_ride + t_deliver + rng.normal(0, 1.6, m)
        # 未写入任何特征的日级冲击（真实的"今天全城就是慢"）
        duration = duration * day_factor * 0.5 + duration * 0.5
        duration = np.clip(duration, 3.0, 240.0)

        rows.append(pd.DataFrame({
            "order_id": [f"{d:%Y%m%d}{k:06d}" for k in range(m)],
            "order_time": order_time,
            "pickup_lat": pick_lat, "pickup_lon": pick_lon,
            "drop_lat": drop_lat, "drop_lon": drop_lon,
            "distance_km": np.round(road_km, 2),
            "time_period": period,
            "weather": weather,
            "merchant_type": merchant,
            "item_count": items,
            "rider_daily_orders": load,
            "area": pickup_area,
            "congestion_level": congestion,
            "duration_min": np.round(duration, 2),
        }))

    df = pd.concat(rows, ignore_index=True)

    # ---------------- 注入脏数据（模拟线上真实数据质量） ----------------
    n = len(df)
    df["item_count"] = df["item_count"].astype(object)  # 允许混入字符串脏值

    # (a) 距离缺失 8%
    idx = rng.choice(n, int(n * 0.08), replace=False)
    df.loc[idx, "distance_km"] = np.nan
    # (b) 天气缺失 3%
    idx = rng.choice(n, int(n * 0.03), replace=False)
    df.loc[idx, "weather"] = None
    # (c) 经纬度写反（脏数据）0.35%
    idx = rng.choice(n, int(n * 0.0035), replace=False)
    df.loc[idx, ["pickup_lat", "pickup_lon"]] = df.loc[idx, ["pickup_lon", "pickup_lat"]].to_numpy()
    # (d) 坐标漂移到城市外 0.5%
    idx = rng.choice(n, int(n * 0.005), replace=False)
    df.loc[idx, "drop_lat"] = df.loc[idx, "drop_lat"] + 1.5
    # (e) 商品数负值 / 字符串脏值 0.6%
    idx = rng.choice(n, int(n * 0.006), replace=False)
    df.loc[idx, "item_count"] = -1
    idx = rng.choice(n, int(n * 0.004), replace=False)
    df.loc[idx, "item_count"] = "3"
    # (f) 目标异常：0 分钟 / 超长单 0.25%
    idx = rng.choice(n, int(n * 0.0015), replace=False)
    df.loc[idx, "duration_min"] = 0.0
    idx = rng.choice(n, int(n * 0.0010), replace=False)
    df.loc[idx, "duration_min"] = rng.uniform(180, 600, len(idx)).round(2)
    # (g) 重复单 1.2%
    dup = df.sample(int(n * 0.012), random_state=SEED)
    df = pd.concat([df, dup], ignore_index=True)

    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. 数据清洗
# ----------------------------------------------------------------------------
def clean_orders(raw: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """规则化清洗，返回 (干净数据, 规则命中统计表)。

    原则：能修复的修复（距离缺失可重算、天气可回填），
          不能修复的删除（坐标错乱、目标异常、重复单）。
          超长尾单不直接删——单独打标后由模型学习，避免把真实业务长尾洗掉。
    """
    df = raw.copy()
    rules: list[dict] = []

    def log(name: str, hit: int, action: str, note: str = ""):
        rules.append({"rule": name, "hit": int(hit), "action": action, "note": note})
        if verbose:
            print(f"  · {name:<26} 命中 {hit:>6} 条 → {action}")

    n0 = len(df)

    # 2.1 类型规整：把混入的字符串/非法值转成数值，转不了的置 NaN
    for c in ["distance_km", "item_count", "rider_daily_orders", "congestion_level",
              "pickup_lat", "pickup_lon", "drop_lat", "drop_lon", "duration_min"]:
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        log(f"类型规整:{c}", df[c].isna().sum() - before_na, "置 NaN")

    df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
    log("时间解析失败", df["order_time"].isna().sum(), "删除")
    df = df.dropna(subset=["order_time"])

    # 2.2 重复单
    dup = df.duplicated(subset=["order_id"], keep="first")
    log("重复 order_id", dup.sum(), "保留首条")
    df = df[~dup]

    # 2.3 坐标合法性：落在城市 bbox 之外视为漂移/经纬度颠倒
    lat_min, lat_max, lon_min, lon_max = CITY_BBOX
    bad_geo = ~(
        df["pickup_lat"].between(lat_min, lat_max)
        & df["drop_lat"].between(lat_min, lat_max)
        & df["pickup_lon"].between(lon_min, lon_max)
        & df["drop_lon"].between(lon_min, lon_max)
    )
    log("坐标越界/经纬度颠倒", bad_geo.sum(), "删除", "占比低于 1% 不做插补")
    df = df[~bad_geo]

    # 2.4 商品数量异常
    bad_item = df["item_count"] < 1
    log("商品数 < 1", bad_item.sum(), "置 1")
    df.loc[bad_item, "item_count"] = 1
    df["item_count"] = df["item_count"].clip(1, 30).astype(int)

    # 2.5 骑手接单量异常
    bad_load = (df["rider_daily_orders"] < 0) | (df["rider_daily_orders"] > 80)
    log("骑手接单量越界", bad_load.sum(), "按 P99 截断")
    df["rider_daily_orders"] = df["rider_daily_orders"].clip(0, 80)

    # 2.6 距离缺失：用经纬度重算（直线 × 片区绕行系数中位数）
    hdist = _haversine_km(df["pickup_lat"], df["pickup_lon"], df["drop_lat"], df["drop_lon"])
    miss_dist = df["distance_km"].isna() | (df["distance_km"] <= 0)
    log("距离缺失/非正", miss_dist.sum(), "用经纬度重算")
    df.loc[miss_dist, "distance_km"] = (hdist[miss_dist] * 1.30).round(2)

    # 2.7 天气缺失：按"同城同日"最可能天气回填，再不行用全局众数
    day = df["order_time"].dt.normalize()
    day_mode = df.groupby(day)["weather"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    miss_w = df["weather"].isna()
    df.loc[miss_w, "weather"] = day[miss_w].map(day_mode)
    miss_w2 = df["weather"].isna()
    df.loc[miss_w2, "weather"] = df["weather"].mode().iloc[0]
    log("天气缺失", miss_w.sum(), "同日众数回填", f"仍缺 {int(miss_w2.sum())} 条用全局众数")

    # 2.8 时段缺失（正常情况下由时间推导，这里兜底）
    df["weather"] = df["weather"].astype(str).str.strip()

    # 2.9 目标异常
    bad_y = (df["duration_min"] <= 2) | (df["duration_min"] > 180)
    log("目标时长 ≤2 或 >180 分钟", bad_y.sum(), "删除", "超长真实单应走赔付流程，不进训练")
    df = df[~bad_y]

    # 2.10 物理合理性：全程平均速度超过 70km/h 必然有字段错误
    speed = df["distance_km"] / (df["duration_min"] / 60.0)
    bad_speed = speed > 70
    log("平均速度 > 70km/h", bad_speed.sum(), "删除")
    df = df[~bad_speed]

    # 2.11 距离重建后的一致性校验：直线距离不可能大于路网距离太多
    hdist = _haversine_km(df["pickup_lat"], df["pickup_lon"], df["drop_lat"], df["drop_lon"])
    bad_ratio = hdist > df["distance_km"] * 1.02
    log("直线距离 > 路网距离", bad_ratio.sum(), "按直线距离重设")
    df.loc[bad_ratio, "distance_km"] = (hdist[bad_ratio] * 1.15).round(2)

    log("合计", n0 - len(df), "删除", f"保留 {len(df)}/{n0} = {len(df)/n0:.2%}")

    df = df.reset_index(drop=True)
    if verbose:
        print(f"\n  清洗后样本量 {len(df):,}，时长均值 {df['duration_min'].mean():.2f} 分钟，"
              f"P90 {df['duration_min'].quantile(0.9):.1f}，P99 {df['duration_min'].quantile(0.99):.1f}\n")
    return df, rules


# ----------------------------------------------------------------------------
# 3. 特征工程
# ----------------------------------------------------------------------------
def _peak_affinity(minute_of_day: np.ndarray) -> np.ndarray:
    """到最近高峰中心的环形距离 → 0~1 的"高峰亲和度"。

    外卖需求不是整点突变，而是围绕 12:00/18:00 的钟形曲线，
    用连续特征比 one-hot 更平滑。
    """
    diff = np.abs(minute_of_day[:, None] - PEAK_CENTERS[None, :])
    diff = np.minimum(diff, 1440 - diff)          # 跨零点取环形最短
    return 1.0 / (1.0 + diff.min(axis=1) / 30.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """构造特征。核心约束：每列特征在 T0（接单时刻）都已知。

    注意：函数内所有"历史统计"特征都只使用**当前日期之前**的数据（as-of），
    绝不使用当天全量或未来数据。
    """
    f = df.copy()
    t = pd.DatetimeIndex(f["order_time"])

    # ---------- 3.1 基础数值特征 ----------
    f["straight_km"] = _haversine_km(f["pickup_lat"], f["pickup_lon"], f["drop_lat"], f["drop_lon"]).round(3)
    f["detour_ratio"] = (f["distance_km"] / f["straight_km"].clip(lower=0.05)).clip(0.8, 4.0)
    f["items"] = f["item_count"]
    f["load"] = f["rider_daily_orders"]
    f["congestion_level"] = f["congestion_level"].clip(0, 3).astype(int)
    f["has_rain"] = (f["weather"] == "雨").astype(int)
    # 远单: 距离超过 6km 的订单，骑手通常优先走主干道，速度曲线不同
    f["is_long_distance"] = (f["distance_km"] >= 6).astype(int)

    # ---------- 3.2 时间特征 ----------
    f["order_date"] = t.normalize()
    f["hour"] = t.hour
    f["weekday"] = t.weekday
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    f["minute_of_day"] = t.hour * 60 + t.minute
    f["hour_sin"] = np.sin(2 * np.pi * f["minute_of_day"] / 1440)
    f["hour_cos"] = np.cos(2 * np.pi * f["minute_of_day"] / 1440)
    f["weekday_sin"] = np.sin(2 * np.pi * f["weekday"] / 7)
    f["weekday_cos"] = np.cos(2 * np.pi * f["weekday"] / 7)
    f["peak_affinity"] = _peak_affinity(f["minute_of_day"].to_numpy())
    f["is_peak"] = f["time_period"].isin(["早高峰", "午高峰", "晚高峰"]).astype(int)
    f["is_late_night"] = (f["time_period"] == "凌晨").astype(int)
    # 距发薪日 / 月初的消费周期（外卖有明显月初效应）
    f["day_of_month"] = t.day
    f["is_month_start_week"] = (t.day <= 7).astype(int)

    # ---------- 3.3 空间地理特征 ----------
    drop_area = _nearest_area(f["drop_lat"].to_numpy(), f["drop_lon"].to_numpy())
    f["area_drop"] = drop_area
    f["same_area"] = (f["area"] == f["area_drop"]).astype(int)
    # 片区内/跨片区：跨片区通常要过主干道/桥，方差更大
    f["area_pickup"] = f["area"]
    # 坐标做轻微的量纲压缩，树模型可自行切分
    f["pickup_lat"], f["pickup_lon"] = f["pickup_lat"].round(5), f["pickup_lon"].round(5)
    f["drop_lat"], f["drop_lon"] = f["drop_lat"].round(5), f["drop_lon"].round(5)

    # ---------- 3.4 统计衍生特征（严格 as-of：只用昨天及以前） ----------
    # 4.4.1 片区历史平均配送时长 —— 用"累计到前一天"的均值
    daily = (
        f.groupby(["order_date", "area_pickup"])["duration_min"]
        .agg(["sum", "count"]).reset_index()
        .sort_values(["area_pickup", "order_date"])
    )
    g = daily.groupby("area_pickup", sort=False)
    daily["hist_sum"] = g["sum"].cumsum() - daily["sum"]     # 去掉当天
    daily["hist_cnt"] = g["count"].cumsum() - daily["count"]
    daily["area_hist_avg"] = np.where(
        daily["hist_cnt"] >= 30, daily["hist_sum"] / daily["hist_cnt"], np.nan
    )
    f = f.merge(daily[["order_date", "area_pickup", "area_hist_avg"]],
                on=["order_date", "area_pickup"], how="left")

    # 4.4.2 片区 × 时段 的历史均值（更细粒度，冷启动用 NaN 交给树模型）
    key = ["area_pickup", "time_period"]
    daily2 = (
        f.groupby(["order_date"] + key)["duration_min"]
        .agg(["sum", "count"]).reset_index()
        .sort_values(key + ["order_date"])
    )
    g2 = daily2.groupby(key, sort=False)
    daily2["hist_sum"] = g2["sum"].cumsum() - daily2["sum"]
    daily2["hist_cnt"] = g2["count"].cumsum() - daily2["count"]
    daily2["area_period_hist_avg"] = np.where(
        daily2["hist_cnt"] >= 15, daily2["hist_sum"] / daily2["hist_cnt"], np.nan
    )
    f = f.merge(daily2[["order_date"] + key + ["area_period_hist_avg"]],
                on=["order_date"] + key, how="left")

    # 4.4.3 前一日片区订单量 / 前一日骑手负载 → 供需紧张度
    prev_day = (
        f.groupby(["order_date", "area_pickup"])
        .agg(prev_orders=("order_id", "count"), prev_load=("rider_daily_orders", "mean"))
        .reset_index()
    )
    prev_day["order_date"] = prev_day["order_date"] + pd.Timedelta(days=1)  # 挪到"次日可用"
    f = f.merge(prev_day, on=["order_date", "area_pickup"], how="left")
    f["supply_pressure"] = f["prev_orders"] / f["prev_load"].clip(lower=1)

    return f


FEATURE_COLS_NUM = [
    "distance_km", "straight_km", "detour_ratio", "items", "load",
    "congestion_level", "is_long_distance", "has_rain",
    "hour", "weekday", "is_weekend", "minute_of_day",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "peak_affinity", "is_peak", "is_late_night", "day_of_month", "is_month_start_week",
    "same_area",
    "area_hist_avg", "area_period_hist_avg", "supply_pressure",
]
FEATURE_COLS_CAT = ["time_period", "weather", "merchant_type", "area_pickup", "area_drop"]
TARGET = "duration_min"

FEATURE_CN = {
    "distance_km": "配送距离",
    "straight_km": "直线距离",
    "detour_ratio": "绕行系数",
    "items": "商品数量",
    "load": "骑手当日接单量",
    "congestion_level": "拥堵等级",
    "is_long_distance": "是否远单",
    "has_rain": "是否雨天",
    "hour": "小时",
    "weekday": "星期",
    "is_weekend": "是否周末",
    "minute_of_day": "当日分钟数",
    "hour_sin": "时段正弦",
    "hour_cos": "时段余弦",
    "weekday_sin": "星期正弦",
    "weekday_cos": "星期余弦",
    "peak_affinity": "高峰亲和度",
    "is_peak": "是否高峰",
    "is_late_night": "是否凌晨",
    "day_of_month": "月中日",
    "is_month_start_week": "是否月初一周",
    "same_area": "是否同片区",
    "area_hist_avg": "片区历史均时长",
    "area_period_hist_avg": "片区时段历史均时长",
    "supply_pressure": "供需压力",
    "time_period": "时段",
    "weather": "天气",
    "merchant_type": "商家类型",
    "area_pickup": "取餐片区",
    "area_drop": "送达片区",
}


def make_xy(df: pd.DataFrame, cat_as_str: bool = False):
    """产出模型输入矩阵。LightGBM 用 category dtype，其他模型可转字符串。"""
    X = df[FEATURE_COLS_NUM + FEATURE_COLS_CAT].copy()
    for c in FEATURE_COLS_CAT:
        X[c] = X[c].astype("category")
        if cat_as_str:
            X[c] = X[c].astype(str)
    for c in FEATURE_COLS_NUM:
        X[c] = X[c].astype(float)
    y = df[TARGET].to_numpy()
    return X, y


# ----------------------------------------------------------------------------
# 4. 时序切分
# ----------------------------------------------------------------------------
def time_split(df: pd.DataFrame, train_days=42, valid_days=8, embargo_days=1):
    """按天切分，段与段之间留隔离带（embargo）。

    时间切分是硬约束：随机打乱会让模型看到"同一时刻的其他订单结果"，
    而这些订单在线上根本不可能知道（例如同片区同时刻的兄弟单）。
    """
    d = df["order_date"].dt.normalize()
    d0 = d.min()
    train_end = d0 + pd.Timedelta(days=train_days - 1)
    valid_start = train_end + pd.Timedelta(days=1 + embargo_days)
    valid_end = valid_start + pd.Timedelta(days=valid_days - 1)
    test_start = valid_end + pd.Timedelta(days=1 + embargo_days)

    tr = df[d <= train_end]
    va = df[(d >= valid_start) & (d <= valid_end)]
    te = df[d >= test_start]

    split_info = {
        "train": (str(tr["order_date"].min().date()), str(tr["order_date"].max().date()), len(tr)),
        "valid": (str(va["order_date"].min().date()), str(va["order_date"].max().date()), len(va)),
        "test": (str(te["order_date"].min().date()), str(te["order_date"].max().date()), len(te)),
        "embargo": f"{embargo_days} 天",
    }
    return tr, va, te, split_info


# ----------------------------------------------------------------------------
# 5. 评估
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, name="model") -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

    err = np.asarray(y_pred) - np.asarray(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE 只在实际时长 >5 分钟的样本上算，避免短单把比例放大失真
    mask = np.asarray(y_true) > 5
    mape = float(np.mean(np.abs(err[mask]) / np.asarray(y_true)[mask])) * 100
    return {
        "name": name,
        "mae": float(mae),
        "rmse": rmse,
        "medae": float(median_absolute_error(y_true, y_pred)),
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float(np.mean(err)),
        "p90_abs_err": float(np.percentile(np.abs(err), 90)),
    }


def overtime_metrics(y_true, risk_score, sla=SLA_MINUTES, threshold=None):
    """超时识别：用 P90 预测值作为风险分，> 阈值 则预警为"可能超时"。"""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    y_bin = (np.asarray(y_true) > sla).astype(int)
    if threshold is None:
        threshold = sla
    flag = (np.asarray(risk_score) > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_bin, flag, average="binary", zero_division=0)
    auc = roc_auc_score(y_bin, risk_score) if y_bin.min() < y_bin.max() else float("nan")
    return {
        "sla": sla, "threshold": float(threshold),
        "positive_rate": float(y_bin.mean()), "flag_rate": float(flag.mean()),
        "precision": float(p), "recall": float(r), "f1": float(f1), "auc": float(auc),
    }


# ----------------------------------------------------------------------------
# 6. 建模
# ----------------------------------------------------------------------------
LGB_PARAMS = dict(
    # 说明：LightGBM 的 monotone_constraints 不支持 regression_l1 / quantile 目标，
    # 这里用 Huber(δ=6 分钟)：对长尾订单鲁棒，误差 ≤δ 时梯度≈MAE，
    # 且原生支持单调约束。评估指标仍统一看 MAE。
    objective="huber",
    alpha=6.0,
    metric="mae",
    learning_rate=0.03,
    num_leaves=48,
    max_depth=-1,
    min_data_in_leaf=120,          # 样本级叶子下限，抑制对个别骑手的记忆
    feature_fraction=0.82,
    bagging_fraction=0.82,
    bagging_freq=1,
    lambda_l1=0.0,
    lambda_l2=5.0,
    max_bin=255,
    num_threads=4,
    verbosity=-1,
    seed=SEED,
)

# 单调约束：距离/拥堵/商品数/负载 增加时，时长不应下降。
# 这既是业务可解释性要求，也能在数据稀疏区（如极端天气）稳住预测。
MONOTONE = {
    "distance_km": 1, "congestion_level": 1, "items": 1, "load": 1,
    "straight_km": 1, "detour_ratio": 1, "has_rain": 1,
}


def train_lgbm(X_tr, y_tr, X_va, y_va, params=None, num_rounds=2500, monotone: bool = True):
    import lightgbm as lgb

    p = dict(LGB_PARAMS)
    if params:
        p.update(params)
    if monotone:
        p["monotone_constraints"] = [MONOTONE.get(c, 0) for c in X_tr.columns]
        p["monotone_constraints_method"] = "advanced"
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    model = lgb.train(
        p, dtrain, num_boost_round=num_rounds,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_quantile(X_tr, y_tr, X_va, y_va, alpha=0.9, num_rounds=1500):
    """分位数模型：给运营提供"最坏情况"口径。"""
    import lightgbm as lgb

    p = dict(LGB_PARAMS)
    p.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    return lgb.train(
        p, dtrain, num_boost_round=num_rounds,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )


def _sklearn_matrices(X_tr, X_va, X_te):
    """线性 / MLP 需要：数值缺失填补 + 标准化，类别 one-hot。"""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, FEATURE_COLS_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS_CAT),
    ])
    A = pre.fit_transform(X_tr)
    B = pre.transform(X_va)
    C = pre.transform(X_te)
    return A, B, C


def compare_models(X_tr, y_tr, X_va, y_va, X_te, y_te) -> list[dict]:
    """模型对比：规则基线 → 线性 → GBDT 家族 → MLP。"""
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    results = []

    # (0) 规则基线：距离 / 18km/h + 固定 12 分钟（接驾+交付）
    naive = X_te["distance_km"].to_numpy() / 18.0 * 60 + 12.0
    results.append(regression_metrics(y_te, naive, "规则基线(距离/18km/h+12)"))
    # (0b) 全局均值基线
    results.append(regression_metrics(y_te, np.full_like(y_te, y_tr.mean(), dtype=float), "全局均值基线"))

    # (1) 线性回归（对数目标，缓解右偏）
    A, B, C = _sklearn_matrices(X_tr, X_va, X_te)
    lr = LinearRegression()
    lr.fit(A, np.log1p(y_tr))
    pred = np.expm1(lr.predict(C))
    results.append(regression_metrics(y_te, pred, "线性回归"))

    # (2) LightGBM（MAE 目标，无单调约束，作为约束版的对照）
    m = train_lgbm(X_tr, y_tr, X_va, y_va,
                   params={"objective": "regression_l1"}, monotone=False)
    results.append(regression_metrics(y_te, m.predict(X_te), "LightGBM"))

    # (3) LightGBM + 单调约束（最终上线候选）
    m_c = train_lgbm(X_tr, y_tr, X_va, y_va, monotone=True)
    results.append(regression_metrics(y_te, m_c.predict(X_te), "LightGBM+单调约束"))

    # (4) XGBoost
    dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    dte = xgb.DMatrix(X_te, enable_categorical=True)
    xp = dict(objective="reg:absoluteerror", eval_metric="mae", eta=0.05, max_depth=7,
              subsample=0.85, colsample_bytree=0.85, min_child_weight=20,
              reg_lambda=5.0, tree_method="hist", nthread=4, seed=SEED)
    bst = xgb.train(xp, dtr, num_boost_round=1500, evals=[(dva, "valid")],
                    early_stopping_rounds=100, verbose_eval=False)
    results.append(regression_metrics(y_te, bst.predict(dte), "XGBoost"))

    # (5) CatBoost（原生类别特征）
    cb = CatBoostRegressor(
        loss_function="MAE", iterations=1200, learning_rate=0.05, depth=7,
        l2_leaf_reg=5.0, cat_features=FEATURE_COLS_CAT, verbose=False,
        early_stopping_rounds=80, random_seed=SEED, thread_count=4,
        allow_writing_files=False,
    )
    cb.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    results.append(regression_metrics(y_te, cb.predict(X_te), "CatBoost"))

    # (6) 小型 MLP（DNN 代理）：复用同一套缺失填补 + one-hot 矩阵
    mlp = MLPRegressor(hidden_layer_sizes=(96, 48), activation="relu",
                       alpha=1e-3, learning_rate_init=1e-3,
                       early_stopping=True, n_iter_no_change=6,
                       max_iter=60, random_state=SEED)
    mlp.fit(A, np.log1p(y_tr))
    results.append(regression_metrics(y_te, np.expm1(mlp.predict(C)), "MLP(2 层)"))
    return results


def leakage_experiment(df: pd.DataFrame, verbose=True):
    """两组对照，量化"离线好看、上线打脸"的两种典型方式。

    A. 切分方式：同一份合法特征，随机打乱 vs 时序切分。
       随机打乱会把"同一天的其他订单"同时放进训练和测试，
       模型能借同城同日的信息校准当天扰动（大促、暴雨、运力紧张），
       测试误差被系统性压低。

    B. 事后特征：把（片区, 当天）全量平均时长当特征。
       它包含本单自身与当天未来订单的结果，离线数值极漂亮，
       但线上接到单的那一刻根本算不出来 —— 属于不可部署特征。
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    p = dict(objective="regression_l1", metric="mae", learning_rate=0.05,
             num_leaves=31, min_data_in_leaf=100, verbose=-1, seed=SEED, num_threads=4)
    d = df.sort_values("order_time").reset_index(drop=True)

    base_cols = FEATURE_COLS_NUM + FEATURE_COLS_CAT

    def _fit_mae(Xa, ya, Xb, yb):
        Xa = Xa.copy(); Xb = Xb.copy()
        for c in FEATURE_COLS_CAT:
            cats = pd.Categorical(pd.concat([Xa[c], Xb[c]]).astype(str)).categories
            Xa[c] = pd.Categorical(Xa[c].astype(str), categories=cats)
            Xb[c] = pd.Categorical(Xb[c].astype(str), categories=cats)
        m = lgb.train(p, lgb.Dataset(Xa, ya), num_boost_round=400)
        pred = m.predict(Xb)
        return float(np.mean(np.abs(pred - yb))), float(np.sqrt(np.mean((pred - yb) ** 2)))

    X_legit = d[base_cols]
    y = d[TARGET].to_numpy()

    # A1 随机切分（错误做法）
    Xa, Xb, ya, yb = train_test_split(X_legit, y, test_size=0.2, random_state=SEED)
    mae_rand, rmse_rand = _fit_mae(Xa, ya, Xb, yb)

    # A2 时序切分（正确做法）
    cut = int(len(d) * 0.8)
    mae_time, rmse_time = _fit_mae(X_legit.iloc[:cut], y[:cut], X_legit.iloc[cut:], y[cut:])

    # B 事后特征 + 时序切分
    d2 = d.copy()
    d2["hindsight_day_area_avg"] = d2.groupby(["order_date", "area_pickup"])[TARGET].transform("mean")
    X_leak = d2[base_cols + ["hindsight_day_area_avg"]]
    mae_leak, rmse_leak = _fit_mae(X_leak.iloc[:cut], y[:cut], X_leak.iloc[cut:], y[cut:])

    out = {
        "random_split": {"mae": mae_rand, "rmse": rmse_rand},
        "time_split": {"mae": mae_time, "rmse": rmse_time},
        "hindsight": {"mae": mae_leak, "rmse": rmse_leak},
        "inflation": (mae_time - mae_rand) / mae_time,
    }
    if verbose:
        print(f"\n  [泄露对照 A] 合法特征，随机切分 MAE={mae_rand:.2f}，时序切分 MAE={mae_time:.2f}")
        print(f"  → 随机切分把误差低估了 {(mae_time - mae_rand) / mae_time:.0%}："
              f"同城同日样本同时进训练与测试，等于提前偷看了当日扰动。")
        print(f"  [泄露对照 B] 加入（片区,当天）全量平均时长，时序切分 MAE={mae_leak:.2f}")
        print(f"  → 离线再降 {(mae_time - mae_leak) / mae_time:.0%}，但该特征在接单时刻不可计算，属不可部署特征。")
    return out


# ----------------------------------------------------------------------------
# 7. 可视化
# ----------------------------------------------------------------------------
def _style_axes(ax, grid_axis="y"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def explain_order(model, X_row: pd.DataFrame, top_k: int = 6) -> pd.DataFrame:
    """单笔订单归因：用 LightGBM 的 pred_contrib 拆出每个特征对预测的贡献。

    输出可直接进运营工单：哪些因素把这一单推向了超时。
    contrib 单位是分钟（回归目标为原始尺度），带符号。
    """
    contrib = model.predict(X_row, pred_contrib=True)[0]
    names = model.feature_name()
    parts = pd.DataFrame({"feature": names, "contrib": contrib[:-1]})
    parts["cn"] = parts["feature"].map(lambda x: FEATURE_CN.get(x, x))
    parts["abs"] = parts["contrib"].abs()
    parts = parts.sort_values("abs", ascending=False).head(top_k)
    return parts[["cn", "feature", "contrib"]].reset_index(drop=True)


def build_risk_watchlist(model, q90, X_te, te, delta, risk_th, top_n=8) -> list[dict]:
    """高风险订单清单：给运营的排查入口。"""
    risk = q90.predict(X_te) + delta
    idx = np.argsort(-risk)[:top_n]
    out = []
    for i in idx:
        row = X_te.iloc[[i]]
        parts = explain_order(model, row, top_k=3)
        out.append({
            "order_id": str(te["order_id"].iloc[i]),
            "area": str(te["area_pickup"].iloc[i]),
            "distance_km": float(te["distance_km"].iloc[i]),
            "period": str(te["time_period"].iloc[i]),
            "weather": str(te["weather"].iloc[i]),
            "p50": round(float(model.predict(row)[0]), 1),
            "p90": round(float(risk[i]), 1),
            "above_threshold": bool(risk[i] > risk_th),
            "drivers": [{"name": r["cn"], "minutes": round(float(r["contrib"]), 2)}
                        for _, r in parts.iterrows()],
        })
    return out


def plot_pred_vs_actual(y_true, y_pred, metrics, path):
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    hb = ax.hexbin(y_true, y_pred, gridsize=46, cmap=BLUE_RAMP, mincnt=1,
                   linewidths=0.2, edgecolors=SURFACE, zorder=2)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(np.percentile(y_true, 99.5), np.percentile(y_pred, 99.5)))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("真实配送时长（分钟）")
    ax.set_ylabel("预测配送时长（分钟）")
    ax.set_title("真实值 vs 预测值", loc="left", fontsize=13, color=INK, pad=10)
    _style_axes(ax, "both")
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.045)
    cb.outline.set_visible(False)
    cb.set_label("订单密度", color=INK_2, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.text(0.04, 0.96,
            f"MAE {metrics['mae']:.2f} 分钟\nRMSE {metrics['rmse']:.2f}\nR² {metrics['r2']:.3f}\nP90 绝对误差 {metrics['p90_abs_err']:.1f}",
            transform=ax.transAxes, va="top", fontsize=9.5, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID, lw=0.8))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(model, path, top_n=15):
    imp = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
    })
    imp["share"] = imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=True).tail(top_n)
    imp["label"] = imp["feature"].map(lambda x: FEATURE_CN.get(x, x))

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.barh(imp["label"], imp["share"] * 100, height=0.62, color=BLUE, zorder=2)
    for v, lab in zip(imp["share"] * 100, imp["label"]):
        ax.text(v + imp["share"].max() * 100 * 0.012, lab, f"{v:.1f}%",
                va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("重要性占比（gain，%）")
    ax.set_title(f"特征重要性 Top {top_n}", loc="left", fontsize=13, color=INK, pad=10)
    ax.set_xlim(0, imp["share"].max() * 100 * 1.16)
    _style_axes(ax, "x")
    ax.tick_params(axis="y", labelsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


def plot_residuals(y_true, y_pred, path):
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bins = np.linspace(np.percentile(err, 0.3), np.percentile(err, 99.7), 58)
    ax.hist(err, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.9, zorder=2)
    ax.axvline(0, color=MUTED, lw=1.1, zorder=3)
    m = float(err.mean())
    ax.axvline(m, color=ORANGE, lw=1.6, zorder=4)
    ax.annotate(f"均值 {m:+.2f}", xy=(m, ax.get_ylim()[1] * 0.78),
                xytext=(-10, 0), textcoords="offset points", ha="right",
                color=INK_2, fontsize=9, va="center")
    p90 = float(np.percentile(np.abs(err), 90))
    ax.text(0.98, 0.95, f"P90 绝对误差 {p90:.1f} 分钟\n误差 ≤5 分钟占比 {(np.abs(err)<=5).mean():.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=INK_2,
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=GRID, lw=0.8))
    ax.set_xlabel("预测误差（预测 − 真实，分钟）")
    ax.set_ylabel("订单数")
    ax.set_title("误差分布", loc="left", fontsize=13, color=INK, pad=10)
    _style_axes(ax, "y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_period_error(df_te, y_pred, path):
    d = df_te[["time_period"]].copy()
    d["abs_err"] = np.abs(y_pred - df_te[TARGET].to_numpy())
    order = ["凌晨", "早高峰", "午高峰", "平峰", "晚高峰"]
    g = d.groupby("time_period")["abs_err"].agg(["mean", "count"]).reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(g.index, g["mean"], width=0.52, color=BLUE, zorder=2)
    for i, (k, v) in enumerate(g["mean"].items()):
        ax.text(i, v + g["mean"].max() * 0.02, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK_2)
    ax.set_ylabel("MAE（分钟）")
    ax.set_title("分时段平均绝对误差", loc="left", fontsize=13, color=INK, pad=10)
    ax.set_ylim(0, g["mean"].max() * 1.18)
    _style_axes(ax, "y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return g


# ----------------------------------------------------------------------------
# 8. 主流程
# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=" * 78)
    print("外卖配送时长预测 · 端到端流程")
    print("=" * 78)

    print("\n[1/8] 模拟订单数据 ...")
    raw = simulate_orders()
    print(f"  原始订单 {len(raw):,} 条，时间跨度 {raw['order_time'].min():%Y-%m-%d} ~ {raw['order_time'].max():%Y-%m-%d}")

    print("\n[2/8] 数据清洗 ...")
    clean, clean_rules = clean_orders(raw, verbose=True)

    print("[3/8] 特征工程 ...")
    feat = build_features(clean)
    print(f"  特征总数 {len(FEATURE_COLS_NUM) + len(FEATURE_COLS_CAT)}"
          f"（数值 {len(FEATURE_COLS_NUM)}，类别 {len(FEATURE_COLS_CAT)}）")

    print("\n[4/8] 时序切分 ...")
    tr, va, te, split_info = time_split(feat)
    for k in ["train", "valid", "test"]:
        s, e, n = split_info[k]
        print(f"  {k:<6} {s} ~ {e}  {n:>6,} 单")
    print(f"  隔离带 {split_info['embargo']}（避免跨日近邻样本同时进训练与验证）")

    X_tr, y_tr = make_xy(tr)
    X_va, y_va = make_xy(va)
    X_te, y_te = make_xy(te)

    print("\n[5/8] 多模型对比 ...")
    cmp_results = compare_models(X_tr, y_tr, X_va, y_va, X_te, y_te)
    print(f"  {'模型':<22}{'MAE':>8}{'RMSE':>9}{'MAPE%':>8}{'R²':>8}")
    for r in cmp_results:
        print(f"  {r['name']:<22}{r['mae']:>8.2f}{r['rmse']:>9.2f}{r['mape']:>8.2f}{r['r2']:>8.3f}")

    print("\n[6/8] 训练最终模型（LightGBM + 单调约束）...")
    model = train_lgbm(X_tr, y_tr, X_va, y_va, monotone=True)
    pred_te = model.predict(X_te)
    m = regression_metrics(y_te, pred_te, "LightGBM+单调约束")
    print(f"  最优迭代轮数 {model.best_iteration}，测试集 MAE {m['mae']:.2f} 分钟，"
          f"RMSE {m['rmse']:.2f}，R² {m['r2']:.3f}")

    q90 = train_quantile(X_tr, y_tr, X_va, y_va, alpha=0.9)
    p90_te = q90.predict(X_te)

    print("\n[7/8] 业务指标 ...")
    # ---- 分位数校准（split conformal + 按日聚合）----
    # 分位数目标函数只保证"平均意义"上的分位，遇到日级分布漂移（暴雨、大促、
    # 运力紧张）会系统性偏低。做法：
    #   1) 在验证集上算逐日 P90 残差；
    #   2) 取这些日残差的 P90 作为偏移量 δ（不是取最大，避免被单个极端日带偏）；
    #   3) 线上预测 = 分位模型输出 + δ。
    # 关键：δ 只用验证集算，测试集一律不碰。
    va_raw = q90.predict(X_va)
    day_q90 = (pd.Series(y_va - va_raw)
               .groupby(pd.DatetimeIndex(va["order_date"]).normalize())
               .quantile(0.90))
    delta = float(np.quantile(day_q90.to_numpy(), 0.90))
    p90_te_cal = p90_te + delta
    coverage_raw = float(np.mean(y_te <= p90_te))
    coverage_cal = float(np.mean(y_te <= p90_te_cal))
    print(f"  P90 校准偏移 δ = {delta:+.2f} 分钟；覆盖率 {coverage_raw:.1%} → {coverage_cal:.1%}（目标 90%）")

    # 逐日覆盖率：只看整体会掩盖"坏日子"，线上要按天监控
    te_day_cov = pd.Series(y_te <= p90_te_cal).groupby(
        pd.DatetimeIndex(te["order_date"]).normalize()).mean()

    # 风险阈值也在验证集上标定，测试集只做评估
    va_risk = q90.predict(X_va) + delta
    y_va_bin = (y_va > SLA_MINUTES).astype(int)
    best = (None, -1)
    for th in np.arange(SLA_MINUTES * 0.6, SLA_MINUTES * 1.5, 0.5):
        from sklearn.metrics import f1_score
        f1 = f1_score(y_va_bin, (va_risk > th).astype(int), zero_division=0)
        if f1 > best[1]:
            best = (float(th), float(f1))
    risk_th = best[0]
    ot = overtime_metrics(y_te, p90_te_cal, SLA_MINUTES, threshold=risk_th)
    print(f"  超时线 {SLA_MINUTES:.0f} 分钟，超时单占比 {ot['positive_rate']:.1%}")
    print(f"  风险阈值 {risk_th:.1f}（验证集 F1 最优）：Precision {ot['precision']:.3f} "
          f"Recall {ot['recall']:.3f} F1 {ot['f1']:.3f} AUC {ot['auc']:.3f}")

    print("\n[8/8] 生成可视化 ...")
    plot_pred_vs_actual(y_te, pred_te, m, OUT_DIR / "pred_vs_actual.png")
    imp = plot_importance(model, OUT_DIR / "feature_importance.png")
    plot_residuals(y_te, pred_te, OUT_DIR / "residual_hist.png")
    period_mae = plot_period_error(te, pred_te, OUT_DIR / "period_mae.png")
    print(f"  图表已保存到 {OUT_DIR}")

    leak = leakage_experiment(feat)

    # ---------- 高风险订单清单（运营排查入口） ----------
    watchlist = build_risk_watchlist(model, q90, X_te, te, delta, risk_th, top_n=6)
    print("\n  高风险订单 Top 3（运营排查样例）：")
    for w in watchlist[:3]:
        drivers = "、".join(f"{d['name']}{d['minutes']:+.1f}分" for d in w["drivers"])
        print(f"    {w['order_id']} {w['area']}/{w['period']}/{w['weather']} "
              f"P50={w['p50']} P90={w['p90']} | {drivers}")

    # ---------- 导出报告页需要的图表数据 ----------
    err = pred_te - y_te
    lo = float(max(min(y_te.min(), pred_te.min()), 0))
    hi = float(min(max(np.percentile(y_te, 99.5), np.percentile(pred_te, 99.5)), 120))
    grid = 52
    xb = np.linspace(lo, hi, grid + 1)
    hist2d, xe, ye = np.histogram2d(y_te, pred_te, bins=[xb, xb])
    cells = []
    for i in range(grid):
        for j in range(grid):
            c = int(hist2d[i, j])
            if c > 0:
                cells.append([round(float(xe[i]), 2), round(float(ye[j]), 2),
                              round(float(xe[i + 1] - xe[i]), 2),
                              round(float(ye[j + 1] - ye[j]), 2), c])

    resid_edges = np.linspace(np.percentile(err, 0.3), np.percentile(err, 99.7), 49)
    resid_counts, _ = np.histogram(err, bins=resid_edges)

    chart_data = {
        "meta": {
            "n_raw": int(len(raw)), "n_clean": int(len(clean)),
            "days": int(feat["order_date"].nunique()),
            "features": len(FEATURE_COLS_NUM) + len(FEATURE_COLS_CAT),
        },
        "split": split_info,
        "metrics": m,
        "overtime": ot,
        "p90": {"delta": delta, "coverage_raw": coverage_raw, "coverage_cal": coverage_cal},
        "baseline": next(r for r in cmp_results if r["name"].startswith("规则基线")),
        "model_compare": cmp_results,
        "clean_rules": clean_rules,
        "importance": [
            {"feature": r["feature"], "cn": FEATURE_CN.get(r["feature"], r["feature"]),
             "share": float(r["share"])}
            for _, r in imp.head(14).iterrows()
        ],
        "scatter": {"grid": grid, "lo": lo, "hi": hi, "cells": cells,
                    "max_count": int(max(c[4] for c in cells))},
        "residual": {"edges": [round(float(x), 3) for x in resid_edges],
                     "counts": [int(x) for x in resid_counts],
                     "mean": float(err.mean()), "p90": float(np.percentile(np.abs(err), 90)),
                     "within5": float((np.abs(err) <= 5).mean())},
        "period_mae": [{"period": str(k), "mae": round(float(v), 2), "n": int(period_mae.loc[k, "count"])}
                       for k, v in period_mae["mean"].items()],
        "leak": leak,
        "sla": SLA_MINUTES,
        "watchlist": watchlist,
        "daily_coverage": [{"date": str(k.date()), "coverage": round(float(v), 3)}
                           for k, v in te_day_cov.items()],
    }
    with open(OUT_DIR / "chart_data.json", "w", encoding="utf-8") as fh:
        json.dump(chart_data, fh, ensure_ascii=False, indent=1)

    # 保存模型（上线用）
    model.save_model(str(OUT_DIR / "lgbm_eta.txt"))
    import joblib
    joblib.dump({"model": model, "quantile": q90, "p90_delta": delta,
                 "features_num": FEATURE_COLS_NUM, "features_cat": FEATURE_COLS_CAT,
                 "risk_threshold": risk_th, "sla": SLA_MINUTES},
                OUT_DIR / "eta_bundle.joblib")

    print(f"\n完成，用时 {time.time()-t_start:.1f}s。产物目录：{OUT_DIR}")
    return chart_data


if __name__ == "__main__":
    main()
