#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_time_analysis.py

第一步：验证分段有效性（必须先完成）

基于你已有的输出：
1) 01_data/time_diagnose/period_daily_summary.csv
2) 01_data/comments_clean_period.parquet
3) 01_data/time_diagnose/period_boundaries.json

完成三件事：
1.1 三段评论量核查统计（transition 和 tail）
    - total_comments：来自 period_daily_summary.csv
    - 涉及视频数：来自 comments_clean_period.parquet 的 video_id 去重
    - 若 total_comments < 30,000 或 平均每视频 < 5，则该段不适合做视频固定效应回归

1.2 断点日期手动验证
    - 输出 8/15-8/25 与 10/10-10/18 两个边界区间的每日评论量折线（并叠加 relevant_ratio）
    - 自动标出 burst_end / transition_end
    - 打印 boundary 附近的“跌落幅度”对比，帮助你判断是否为结构性下跌
    - 若发现断点不对：修改 period_boundaries.json 并重新生成 comments_clean_period.parquet

1.3 机器人占比分布
    - 从 comments_clean_period.parquet 计算 is_suspicious_user 的均值（按 period 分组）
    - 若 tail 期机器人占比 > 15%，需要在论文里明确说明 tail 结论稳健性依赖敏感性分析
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "01_data"
DEFAULT_TIME_DIAGNOSE_DIR = DEFAULT_DATA_DIR / "time_diagnose"
DEFAULT_PERIOD_DAILY_SUMMARY = DEFAULT_TIME_DIAGNOSE_DIR / "period_daily_summary.csv"
DEFAULT_PERIOD_BOUNDARIES = DEFAULT_TIME_DIAGNOSE_DIR / "period_boundaries.json"
DEFAULT_PERIOD_PARQUET = DEFAULT_DATA_DIR / "comments_clean_period.parquet"


def _parse_bool_like(s: pd.Series) -> pd.Series:
    """把 CSV 读回来的 'True'/'False' 等字符串转回 bool（尽量宽容）。"""
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


def _load_boundaries(path: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    burst_end = pd.Timestamp(payload["burst_end"])
    transition_end = pd.Timestamp(payload["transition_end"])
    # 统一到 Asia/Shanghai 时区（如果 payload 带 tz 则不影响）
    if burst_end.tzinfo is None:
        burst_end = burst_end.tz_localize("Asia/Shanghai")
    if transition_end.tzinfo is None:
        transition_end = transition_end.tz_localize("Asia/Shanghai")
    return burst_end, transition_end


def _make_interval_plot(
    daily: pd.DataFrame,
    *,
    start: str,
    end: str,
    burst_end: pd.Timestamp | None,
    transition_end: pd.Timestamp | None,
    out_path: Path,
    title: str,
) -> None:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"])

    s = pd.Timestamp(start).normalize()
    e = pd.Timestamp(end).normalize()
    d = d[(d["date"] >= s) & (d["date"] <= e)].sort_values("date")
    if len(d) == 0:
        raise ValueError(f"区间 {start}~{end} 没有数据行，无法绘图。")

    x = d["date"].to_numpy()
    y_total = d["total_comments"].to_numpy(dtype=float)
    y_ratio = d["relevant_ratio"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(x, y_total, linewidth=2.0, color="#1f77b4")
    ax1.set_ylabel("total_comments")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(x, y_ratio, linewidth=1.6, color="#d62728", alpha=0.85, label="relevant_ratio")
    ax2.set_ylabel("relevant_ratio")

    # 标出断点（用日期精度匹配）
    def _mark(ts: pd.Timestamp | None, label: str, color: str) -> None:
        if ts is None:
            return
        tday = pd.Timestamp(ts).normalize()
        ax1.axvline(tday.to_numpy(), linestyle="--", linewidth=1.6, color=color, alpha=0.9)
        ax1.text(tday.to_numpy(), ax1.get_ylim()[1], label, color=color, fontsize=10, va="top")

    _mark(burst_end, "burst_end", "#2ca02c")
    _mark(transition_end, "transition_end", "#ff7f0e")

    plt.title(title)
    fig.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _drop_report(
    daily: pd.DataFrame,
    *,
    boundary_day: pd.Timestamp,
    pre_days: int = 3,
    post_days: int = 3,
) -> Dict[str, float]:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    # daily 里来自 CSV 的 date 通常是 tz-naive；而边界可能是 tz-aware。
    # 统一都转成 tz-naive，避免 pandas InvalidComparison。
    try:
        if getattr(d["date"].dt, "tz", None) is not None:
            d["date"] = d["date"].dt.tz_localize(None)
    except Exception:
        pass

    b = pd.Timestamp(boundary_day)
    if b.tzinfo is not None:
        b = b.tz_localize(None)
    b = b.normalize()

    pre_start = b - pd.Timedelta(days=pre_days)
    pre_end = b
    post_start = b + pd.Timedelta(days=1)
    post_end = b + pd.Timedelta(days=post_days)

    pre = d[(d["date"] >= pre_start) & (d["date"] <= pre_end)]["total_comments"].astype(float)
    post = d[(d["date"] >= post_start) & (d["date"] <= post_end)]["total_comments"].astype(float)

    pre_mean = float(pre.mean()) if len(pre) else float("nan")
    post_mean = float(post.mean()) if len(post) else float("nan")
    ratio = post_mean / pre_mean if np.isfinite(pre_mean) and pre_mean != 0 else float("nan")
    return {"pre_mean": pre_mean, "post_mean": post_mean, "post_over_pre": ratio}


def main() -> None:
    parser = argparse.ArgumentParser(description="07：验证 period 三段式窗口是否可用（fixed effect 前置核查）")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--time-diagnose-dir", type=Path, default=DEFAULT_TIME_DIAGNOSE_DIR)
    parser.add_argument("--period-daily-summary", type=Path, default=DEFAULT_PERIOD_DAILY_SUMMARY)
    parser.add_argument("--period-boundaries", type=Path, default=DEFAULT_PERIOD_BOUNDARIES)
    parser.add_argument("--period-parquet", type=Path, default=DEFAULT_PERIOD_PARQUET)
    args = parser.parse_args()

    daily_path = args.period_daily_summary
    boundaries_path = args.period_boundaries
    pq_path = args.period_parquet

    if not daily_path.exists():
        raise FileNotFoundError(f"找不到 period_daily_summary.csv: {daily_path}")
    if not boundaries_path.exists():
        raise FileNotFoundError(f"找不到 period_boundaries.json: {boundaries_path}")
    if not pq_path.exists():
        raise FileNotFoundError(f"找不到 comments_clean_period.parquet: {pq_path}")

    daily = pd.read_csv(daily_path, encoding="utf-8-sig", low_memory=False)
    # 保护：确保需要的列都存在
    for c in ["period", "date", "total_comments", "relevant_ratio"]:
        if c not in daily.columns:
            raise KeyError(f"period_daily_summary.csv 缺少列 {c}")

    burst_end, transition_end = _load_boundaries(boundaries_path)

    comments = pd.read_parquet(pq_path)
    if "video_id" not in comments.columns or "is_suspicious_user" not in comments.columns:
        raise KeyError("comments_clean_period.parquet 缺少 video_id 或 is_suspicious_user 列")
    comments["is_suspicious_user"] = _parse_bool_like(comments["is_suspicious_user"])

    # ========== 1.1 分段可用性核查 ==========
    print("\n=== 1.1 分段可用性核查（fixed effect 前置条件） ===")
    verdict: Dict[str, bool] = {}

    # 按用户要求：重点核查 transition/tail；burst 也打印便于你整体把握
    for period in ["burst", "transition", "tail"]:
        total_comments = float(daily.loc[daily["period"] == period, "total_comments"].sum())
        video_cnt = int(comments.loc[comments["period"] == period, "video_id"].nunique())
        avg_per_video = (total_comments / video_cnt) if video_cnt else float("nan")

        # 用户给出的限制规则
        bad_total = total_comments < 30_000
        bad_avg = avg_per_video < 5 if np.isfinite(avg_per_video) else True
        ok = not (bad_total or bad_avg)
        verdict[period] = ok

        print(
            f"{period:10s} total_comments={total_comments:,.0f}  "
            f"video_cnt={video_cnt:,}  avg_per_video={avg_per_video:.2f}  => "
            + ("OK" if ok else "NOT OK")
        )
        if period in ["transition", "tail"] and (bad_total or bad_avg):
            print(
                f"  [限制说明] {period} 由于 "
                + ("total_comments < 30,000；" if bad_total else "")
                + ("avg_per_video < 5；" if bad_avg else "")
                + "需要在方法论上提前说明：不适合直接做视频固定效应回归。"
            )

    # ========== 1.2 边界区间手动验证（输出表+折线图） ==========
    print("\n=== 1.2 断点日期手动验证（每日评论量曲线） ===")
    out_dir = args.time_diagnose_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _print_interval_table(start: str, end: str, boundary_days: Dict[str, pd.Timestamp | None]) -> None:
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        d = d[(d["date"] >= s) & (d["date"] <= e)].sort_values("date")

        show_cols = ["date", "period", "total_comments", "relevant_ratio"]
        d2 = d[show_cols].copy()
        # 标注边界日（若落在区间内）
        for name, ts in boundary_days.items():
            if ts is None:
                continue
            tday = pd.Timestamp(ts).normalize()
            if tday.tzinfo is not None:
                tday = tday.tz_localize(None)
            mask = d2["date"] == tday
            if mask.any():
                # 直接在表格里给一个标记列
                d2.loc[mask, "boundary_marker"] = name
        if "boundary_marker" not in d2.columns:
            d2["boundary_marker"] = ""
        print(d2.to_string(index=False))

    # 8/15-8/25：burst_end 附近
    _print_interval_table(
        "2025-08-15",
        "2025-08-25",
        boundary_days={"burst_end": burst_end, "transition_end": transition_end},
    )
    report_burst = _drop_report(daily, boundary_day=burst_end)
    print(
        f"burst_end 跌落对比（均值 pre={report_burst['pre_mean']:.0f}, post={report_burst['post_mean']:.0f}；post/pre={report_burst['post_over_pre']:.3f}）"
    )
    plot1 = out_dir / "boundary_check_8_15_8_25.png"
    _make_interval_plot(
        daily,
        start="2025-08-15",
        end="2025-08-25",
        burst_end=burst_end,
        transition_end=None,
        out_path=plot1,
        title="Boundary Check: 2025-08-15 ~ 2025-08-25",
    )
    print(f"折线图已保存: {plot1}")

    # 10/10-10/18：transition_end 附近
    _print_interval_table(
        "2025-10-10",
        "2025-10-18",
        boundary_days={"burst_end": burst_end, "transition_end": transition_end},
    )
    report_trans = _drop_report(daily, boundary_day=transition_end)
    print(
        f"transition_end 跌落对比（均值 pre={report_trans['pre_mean']:.0f}, post={report_trans['post_mean']:.0f}；post/pre={report_trans['post_over_pre']:.3f}）"
    )
    plot2 = out_dir / "boundary_check_10_10_10_18.png"
    _make_interval_plot(
        daily,
        start="2025-10-10",
        end="2025-10-18",
        burst_end=None,
        transition_end=transition_end,
        out_path=plot2,
        title="Boundary Check: 2025-10-10 ~ 2025-10-18",
    )
    print(f"折线图已保存: {plot2}")

    # ========== 1.3 机器人占比核查 ==========
    print("\n=== 1.3 机器人占比分布（is_suspicious_user 均值） ===")
    robot_share = comments.groupby("period")["is_suspicious_user"].mean()
    for p, v in robot_share.items():
        print(f"{p:10s} suspicious_user_ratio={float(v):.2%}")
    tail_share = float(robot_share.get("tail", np.nan))
    if np.isfinite(tail_share) and tail_share > 0.15:
        print(
            f"  [需要方法论说明] tail 期机器人占比 {tail_share:.2%} > 15%。"
            "论文里需明确：tail 期结论的稳健性依赖敏感性分析（比如剔除/降权 suspicious_user）。"
        )
    else:
        print("  tail 期机器人占比未超过 15%，敏感性分析依然建议做，但方法论限制可以弱化。")

    print("\n07 分段有效性验证完成。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

