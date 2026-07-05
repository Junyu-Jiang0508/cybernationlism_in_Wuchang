#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_time_analysis.py

决策2：三段式研究窗口的操作指南（自动断点 + period 标记）

目标：
1) 先从 `01_data/time_diagnose/daily_summary.csv` 中找到结构断点，
   自动生成三段边界：burst / transition / tail
2) 再把边界应用回 `01_data/comments_clean.parquet|csv`，
   为每条评论增加：
   - `period`：burst / transition / tail
   - `t_day`：距发售日天数（0 为发售日；正为发售后，负为发售前）
3) 输出：
   - `01_data/comments_clean_period.parquet|csv`（新增 period/t_day）
   - `01_data/time_diagnose/period_daily_summary.csv`
   - 终端打印：burst 期 `like_count_clean == 0` 的占比等关键指标

注意（与你的论文写作约束一致）：
- 你应当在回归主模型中只使用 `period == "burst"`。
- transition 作为“平台干预证据/背景描述”，不应与 burst/tail 混在同一回归中建模。
- tail 用作敏感性分析（敏感性重跑）。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


EVENT_START = pd.Timestamp("2025-07-23 00:00:00", tz="Asia/Shanghai")
EVENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="Asia/Shanghai")


PERIOD_BURST = "burst"
PERIOD_TRANSITION = "transition"
PERIOD_TAIL = "tail"


def _to_datetime_tz(series: pd.Series, tz: str = "Asia/Shanghai") -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    # naive -> localize；tz-aware -> convert
    tzinfo = getattr(dt.dt, "tz", None)
    if tzinfo is None:
        dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        dt = dt.dt.tz_convert(tz)
    return dt


def _parse_bool_like(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
    }
    out = ss.map(mapping)
    return out.fillna(False).astype(bool)


@dataclass(frozen=True)
class Boundaries:
    burst_end: pd.Timestamp  # inclusive (at day precision)
    transition_end: pd.Timestamp  # inclusive


def _pick_first_boundary(
    daily: pd.DataFrame,
    *,
    low_total_threshold: int,
    stable_days: int,
    rebound_margin: float,
) -> Optional[pd.Timestamp]:
    """
    burst_end：找一个“自然终点”：
    从 candidate+1 开始，接下来 stable_days 天 total_comments 不再显著反弹，
    并且稳定落在低位（<= low_total_threshold * (1 + rebound_margin)）。
    """
    d = daily.sort_values("date").copy()
    d["total_comments"] = d["total_comments"].fillna(0).astype(int)
    dates = pd.to_datetime(d["date"].astype(str))
    totals = d["total_comments"].to_numpy()

    n = len(dates)
    for i in range(n - stable_days - 1):
        # candidate 是 burst_end（inclusive），所以观察窗口从 i+1 开始
        start = i + 1
        end = start + stable_days  # exclusive
        window = totals[start:end]
        if len(window) != stable_days:
            continue
        if window.max() <= int(low_total_threshold * (1.0 + rebound_margin)) and window.mean() <= low_total_threshold:
            # burst_end = candidate day；取“首次满足条件”的自然终点
            return dates.iloc[i]
    return None


def _pick_second_boundary(
    daily: pd.DataFrame,
    *,
    second_start: pd.Timestamp,
    ratio_drop_threshold: float,
    ratio_stable_threshold: float,
    stable_days: int,
    max_search_days: int,
) -> Optional[pd.Timestamp]:
    """
    transition_end：在第二段内部先出现断崖（ratio 落到很低），
    然后在后续找到首次“稳定进入高位”（>= ratio_stable_threshold for stable_days）。

    返回 transition_end（inclusive day precision），即稳定高位窗口的最后一天。
    """
    d = daily.sort_values("date").copy()
    d["relevant_ratio"] = d["relevant_ratio"].astype(float)
    d["relevant_ratio"] = d["relevant_ratio"].replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=["relevant_ratio"])

    dates = pd.to_datetime(d["date"].astype(str))
    ratios = d["relevant_ratio"].to_numpy()

    # 找 second_start 在序列里的起点
    # 以 date precision 比较
    start_ts = pd.Timestamp(second_start.date())
    start_idx = None
    for idx, dt in enumerate(dates):
        if dt.normalize() == start_ts:
            start_idx = idx
            break
    if start_idx is None:
        # fallback：选择第一个 >= second_start 的点
        start_idx = int(np.searchsorted(dates.values.astype("datetime64[D]"), np.datetime64(start_ts.date())))
        start_idx = min(max(start_idx, 0), len(dates) - 1)

    # 1) 断崖存在性：从 start_idx 往后找到一个日比低于 ratio_drop_threshold 的点（用于验证）
    drop_found = False
    for j in range(start_idx, min(len(dates), start_idx + max_search_days)):
        if ratios[j] < ratio_drop_threshold:
            drop_found = True
            drop_idx = j
            break
    if not drop_found:
        return None

    # 2) 从 drop_idx 往后找稳定高位首次进入
    search_begin = drop_idx
    for i in range(search_begin, len(dates) - stable_days + 1):
        window = ratios[i : i + stable_days]
        if len(window) != stable_days:
            continue
        if np.nanmin(window) >= ratio_stable_threshold:
            # i..i+stable_days-1 为“稳定高位窗口”
            return dates.iloc[i + stable_days - 1]
    return None


def infer_boundaries(
    daily: pd.DataFrame,
    *,
    low_total_threshold: int,
    first_stable_days: int,
    rebound_margin: float,
    ratio_drop_threshold: float,
    ratio_stable_threshold: float,
    second_stable_days: int,
    max_search_days: int,
    manual_first_end: Optional[str],
    manual_second_end: Optional[str],
) -> Boundaries:
    if manual_first_end:
        burst_end = pd.Timestamp(manual_first_end).tz_localize(EVENT_START.tz) if pd.Timestamp(manual_first_end).tzinfo is None else pd.Timestamp(manual_first_end)
    else:
        burst_end = _pick_first_boundary(
            daily,
            low_total_threshold=low_total_threshold,
            stable_days=first_stable_days,
            rebound_margin=rebound_margin,
        )
        if burst_end is None:
            raise RuntimeError("自动推断 burst_end 失败：请检查阈值或使用 --manual-first-end 指定。")

    second_start = burst_end + pd.Timedelta(days=1)

    if manual_second_end:
        transition_end = pd.Timestamp(manual_second_end)
        if transition_end.tzinfo is None:
            transition_end = transition_end.tz_localize(EVENT_START.tz)
    else:
        transition_end = _pick_second_boundary(
            daily,
            second_start=second_start,
            ratio_drop_threshold=ratio_drop_threshold,
            ratio_stable_threshold=ratio_stable_threshold,
            stable_days=second_stable_days,
            max_search_days=max_search_days,
        )
        if transition_end is None:
            raise RuntimeError("自动推断 transition_end 失败：请检查阈值或使用 --manual-second-end 指定。")

    # 规范化为 tz-aware day precision
    burst_end = pd.Timestamp(burst_end.date()).tz_localize(EVENT_START.tz)
    transition_end = pd.Timestamp(transition_end.date()).tz_localize(EVENT_START.tz)
    return Boundaries(burst_end=burst_end, transition_end=transition_end)


def _load_daily_summary(time_diagnose_dir: Path) -> pd.DataFrame:
    p = time_diagnose_dir / "daily_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 daily_summary.csv: {p}")
    df = pd.read_csv(p, encoding="utf-8-sig", dtype={"date": str})
    if "date" not in df.columns:
        raise KeyError("daily_summary.csv 缺少列 date")
    return df


def _load_comments(data_dir: Path) -> pd.DataFrame:
    pq = data_dir / "comments_clean.parquet"
    csv = data_dir / "comments_clean.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, encoding="utf-8-sig", low_memory=False, dtype=str)
    else:
        raise FileNotFoundError(f"找不到 comments_clean.(parquet/csv): {pq} 或 {csv}")
    return df


def _ensure_columns_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    required = ["create_dt", "content_clean", "like_count_clean", "is_top_comment", "is_suspicious_user"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"comments_clean.* 缺少列 {c}（请确认你已用 clean_bilibili_comments.py 生成）")

    df = df.copy()
    df["create_dt"] = _to_datetime_tz(df["create_dt"])
    df["like_count_clean"] = pd.to_numeric(df["like_count_clean"], errors="coerce")
    df["is_top_comment"] = _parse_bool_like(df["is_top_comment"])
    df["is_suspicious_user"] = _parse_bool_like(df["is_suspicious_user"])
    return df


def assign_period_and_tday(df: pd.DataFrame, b: Boundaries) -> pd.DataFrame:
    df = df.copy()
    create_norm = df["create_dt"].dt.normalize()
    burst_mask = create_norm <= b.burst_end
    transition_mask = (create_norm > b.burst_end) & (create_norm <= b.transition_end)

    df["period"] = np.where(burst_mask, PERIOD_BURST, np.where(transition_mask, PERIOD_TRANSITION, PERIOD_TAIL))

    release_norm = EVENT_START.normalize()
    # dt.normalize 保留 tz-aware；差值得到 Timedelta，最后取天数
    df["t_day"] = (create_norm - release_norm).dt.days.astype(int)
    return df


def summarize_period(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["create_dt"].dt.date

    # period 内的“日聚合”
    g = df.groupby(["period", "date"], as_index=False)
    daily = g.agg(
        total_comments=("comment_id", "count") if "comment_id" in df.columns else ("content_clean", "count"),
        relevant_count=("content_clean", "size"),  # 占位，后面再覆盖
        suspicious_comments=("is_suspicious_user", "sum"),
        top_comments=("is_top_comment", "sum"),
        like_median=("like_count_clean", "median"),
        like_p25=("like_count_clean", lambda x: x.quantile(0.25)),
        like_p75=("like_count_clean", lambda x: x.quantile(0.75)),
    )

    # 如果 comments_clean 里没有 is_relevant，就按内容兜底用关键词（避免脚本不可复现）
    if "is_relevant" in df.columns:
        daily["relevant_count"] = df.groupby(["period", "date"])["is_relevant"].sum().to_numpy()
    else:
        # 兜底：对 content_clean 做粗匹配
        # （这里不复用你在 clean_bilibili_comments.py 的关键字列表，避免跨脚本耦合；你后续仍以 is_relevant 为准）
        kw = ["明末", "渊虚之羽", "游戏"]
        mask = df["content_clean"].astype(str).str.contains("|".join(kw), na=False)
        rel = df.loc[mask].groupby(["period", "date"]).size()
        daily = daily.set_index(["period", "date"])
        daily["relevant_count"] = rel
        daily["relevant_count"] = daily["relevant_count"].fillna(0).astype(int)
        daily = daily.reset_index()

    daily["relevant_ratio"] = np.where(daily["total_comments"] > 0, daily["relevant_count"] / daily["total_comments"], np.nan)
    daily["suspicious_ratio"] = np.where(daily["total_comments"] > 0, daily["suspicious_comments"] / daily["total_comments"], np.nan)
    daily = daily.sort_values(["period", "date"])
    daily.to_csv(out_path, index=False, encoding="utf-8-sig")
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="三段式窗口 period 标记 + t_day 生成（基于 daily_summary 结构断点）")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "01_data")
    parser.add_argument("--time-diagnose-dir", type=Path, default=Path(__file__).resolve().parents[2] / "01_data" / "time_diagnose")
    parser.add_argument("--out-prefix", type=str, default="comments_clean_period")

    # 自动/手动边界
    parser.add_argument("--manual-first-end", type=str, default=None, help="手动指定 burst_end（YYYY-MM-DD），覆盖自动推断")
    parser.add_argument("--manual-second-end", type=str, default=None, help="手动指定 transition_end（YYYY-MM-DD），覆盖自动推断")

    # 自动推断阈值（默认值与“万级到千级以下稳定”经验一致，但仍建议你在自己的数据上微调）
    parser.add_argument("--low-total-threshold", type=int, default=3000, help="burst 之后稳定低位的 total_comments 上界阈值")
    parser.add_argument("--first-stable-days", type=int, default=25, help="用于判断 burst 终点稳定性的天数（越大越不容易误判早期波动）")
    parser.add_argument("--rebound-margin", type=float, default=0.1, help="允许的反弹幅度比例（相对 low-total-threshold），越小越符合“跌破后不反弹”】【默认 0.1）")

    parser.add_argument("--ratio-drop-threshold", type=float, default=0.08, help="transition 内部用于验证断崖的相关占比阈值")
    parser.add_argument("--ratio-stable-threshold", type=float, default=0.30, help="transition 结束时相关占比稳定阈值")
    parser.add_argument("--second-stable-days", type=int, default=7, help="用于判断 transition_end 的稳定天数")
    parser.add_argument("--max-search-days", type=int, default=90, help="在第二段内搜索结构断点的最大天数")

    args = parser.parse_args()

    daily = _load_daily_summary(args.time_diagnose_dir)
    # 确保字段存在
    for c in ["date", "total_comments", "relevant_ratio"]:
        if c not in daily.columns:
            raise KeyError(f"daily_summary.csv 缺少列 {c}")

    # 推断边界
    boundaries = infer_boundaries(
        daily,
        low_total_threshold=args.low_total_threshold,
        first_stable_days=args.first_stable_days,
        rebound_margin=args.rebound_margin,
        ratio_drop_threshold=args.ratio_drop_threshold,
        ratio_stable_threshold=args.ratio_stable_threshold,
        second_stable_days=args.second_stable_days,
        max_search_days=args.max_search_days,
        manual_first_end=args.manual_first_end,
        manual_second_end=args.manual_second_end,
    )

    # 输出边界
    out_dir = args.time_diagnose_dir
    boundary_path = out_dir / "period_boundaries.json"
    boundary_payload = {
        "EVENT_START": str(EVENT_START),
        "EVENT_END": str(EVENT_END),
        "burst_end": str(boundaries.burst_end),
        "transition_end": str(boundaries.transition_end),
    }
    boundary_path.write_text(json.dumps(boundary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 三段式研究窗口边界（自动断点） ===")
    print(f"burst:      {EVENT_START.date()} ~ {boundaries.burst_end.date()}")
    print(
        "transition: "
        + str((boundaries.burst_end + pd.Timedelta(days=1)).date())
        + " ~ "
        + str(boundaries.transition_end.date())
    )
    print(
        "tail:       "
        + str((boundaries.transition_end + pd.Timedelta(days=1)).date())
        + " ~ "
        + str(EVENT_END.date())
    )
    print(f"边界文件: {boundary_path}")

    # load comments & 标记
    comments = _load_comments(args.data_dir)
    comments = _ensure_columns_for_analysis(comments)
    comments = assign_period_and_tday(comments, boundaries)

    out_pq = args.data_dir / f"{args.out_prefix}.parquet"
    out_csv = args.data_dir / f"{args.out_prefix}.csv"
    comments.to_parquet(out_pq, index=False)
    comments.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\nperiod 标记输出：")
    print(f"  {out_pq}")
    print(f"  {out_csv}")

    # 每段关键统计：burst 期 like==0 比例
    burst = comments.loc[comments["period"] == PERIOD_BURST].copy()
    burst_like = burst["like_count_clean"]
    non_na = burst_like.notna().sum()
    zero_cnt = int((burst_like.fillna(np.nan) == 0).sum())
    zero_ratio = float(zero_cnt / non_na) if non_na else float("nan")
    print("\n=== burst 期 like_count_clean==0 占比（因变量操作决策用） ===")
    print(f"burst 总评论数: {len(burst):,}")
    print(f"like_count_clean 非空: {non_na:,}")
    print(f"like_count_clean == 0 的计数: {zero_cnt:,}")
    print(f"like_count_clean == 0 占比: {zero_ratio:.2%}")

    # period 日聚合摘要（给你画图/写方法用）
    daily_out = args.time_diagnose_dir / "period_daily_summary.csv"
    daily_summary = summarize_period(comments, daily_out)
    print(f"\n期内日聚合摘要：{daily_out}（行数 {len(daily_summary):,}）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
