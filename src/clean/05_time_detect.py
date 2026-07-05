#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_time_detect.py

时间分布诊断（Step A）：把评论创建时间 create_dt 当作“事件发生的时间轴”。
输出：
  1) 每日评论量折线图
  2) 每日关键词相关占比折线图（相关评论数/总评论数）
  3) 每日 like_count_clean 中位数折线图（可选分位区间）
  4) 一级 vs 二级评论每日计数折线图
  5) 疑似机器人评论每日计数折线图 + 占比
  6) daily_summary.csv（聚合表）
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVENT_START = pd.Timestamp("2025-07-23", tz="Asia/Shanghai")
EVENT_END = pd.Timestamp("2025-12-31 23:59:59", tz="Asia/Shanghai")

# 如果 annotation_pool.csv 缺失，则用该关键词在 content_clean 上兜底识别相关评论
RELEVANCE_KEYWORDS = [
    "明末",
    "渊虚之羽",
    "游戏",
    "开发商",
    "汉奸",
    "辱华",
    "历史",
    "满清",
    "清朝",
    "殖民",
    "民族",
    "国产",
    "国货",
    "文化",
    "传统",
    "道歉",
    "抵制",
    "下架",
    "删改",
    "和解",
    "愤怒",
    "失望",
    "支持",
    "感动",
]


def _setup_matplotlib_chinese_font() -> None:
    """
    尽量自动选择一个包含中文的字体，避免控制台刷“Glyph ... missing from font”。
    若系统缺字形，这一步无法完全解决，但通常能显著减少告警噪声。
    """
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "PingFang SC",
        ]
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            plt.rcParams["font.sans-serif"] = [chosen]
            plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        # 不影响主逻辑
        pass


_setup_matplotlib_chinese_font()
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font.*")


def _ensure_datetime_tz(s: pd.Series, tz: str = "Asia/Shanghai") -> pd.Series:
    """把 create_dt 字符串（可能带时区）统一转成 tz-aware datetime（Asia/Shanghai）。"""
    dt = pd.to_datetime(s, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        # 原始数据是 naive：按研究时区本地化
        dt = dt.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    else:
        dt = dt.dt.tz_convert(tz)
    return dt


def _parse_bool_series(s: pd.Series) -> pd.Series:
    """把 CSV 读回来的 'True'/'False' 等字符串转回 bool（尽量宽容）。"""
    if s.dtype == bool:
        return s
    ss = s.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "t": True,
        "f": False,
    }
    out = ss.map(mapping)
    return out.fillna(False).astype(bool)


def _load_comments(data_dir: Path) -> pd.DataFrame:
    parquet_path = data_dir / "comments_clean.parquet"
    csv_path = data_dir / "comments_clean.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False, dtype=str)
    else:
        raise FileNotFoundError(f"找不到 comments_clean.(parquet/csv): {parquet_path} 或 {csv_path}")
    return df


def _load_relevant_comments(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    返回关键词相关评论（annotation_pool.csv）的 DataFrame。
    若文件缺失则返回 None，由调用方用 content_clean 兜底识别。
    """
    ann_path = data_dir / "annotation_pool.csv"
    if not ann_path.exists():
        return None
    return pd.read_csv(ann_path, encoding="utf-8-sig", low_memory=False, dtype=str)


def _maybe_plot_line(
    x: list[pd.Timestamp],
    y: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    label: str,
) -> None:
    plt.figure(figsize=(13, 4.8))
    plt.plot(x, y, linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel("日期")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _topk_dates(summary_df: pd.DataFrame, col: str, k: int = 5) -> list[tuple[str, float]]:
    top = summary_df.sort_values(col, ascending=False).head(k)
    out: list[tuple[str, float]] = []
    for _, row in top.iterrows():
        out.append((str(row["date"]), float(row[col])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="时间分布诊断（按评论 create_dt）")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[2] / "01_data")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[2] / "01_data" / "time_diagnose")
    parser.add_argument("--event-start", type=str, default=str(EVENT_START))
    parser.add_argument("--event-end", type=str, default=str(EVENT_END))
    parser.add_argument("--no-show", action="store_true", help="不显示 matplotlib 窗口（默认保存图片即可）")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 覆盖可选 event window（主要用于调试/复跑）
    event_start = pd.Timestamp(args.event_start).tz_convert("Asia/Shanghai") if pd.Timestamp(args.event_start).tzinfo else pd.Timestamp(args.event_start).tz_localize("Asia/Shanghai")
    event_end = pd.Timestamp(args.event_end).tz_convert("Asia/Shanghai") if pd.Timestamp(args.event_end).tzinfo else pd.Timestamp(args.event_end).tz_localize("Asia/Shanghai")

    comments_df = _load_comments(args.data_dir)
    if "create_dt" not in comments_df.columns:
        raise KeyError("comments_clean.* 缺少列 create_dt")

    # 统一类型
    comments_df["create_dt"] = _ensure_datetime_tz(comments_df["create_dt"])
    comments_df = comments_df.loc[
        (comments_df["create_dt"].notna())
        & (comments_df["create_dt"] >= event_start)
        & (comments_df["create_dt"] <= event_end)
    ].copy()
    if len(comments_df) == 0:
        raise ValueError("create_dt 在事件窗口内的评论数为 0，请检查数据是否已被清洗脚本过滤。")

    comments_df["like_count_clean"] = pd.to_numeric(comments_df.get("like_count_clean"), errors="coerce")
    comments_df["is_top_comment"] = _parse_bool_series(comments_df.get("is_top_comment", pd.Series(False, index=comments_df.index)))
    comments_df["is_suspicious_user"] = _parse_bool_series(comments_df.get("is_suspicious_user", pd.Series(False, index=comments_df.index)))
    comments_df["date"] = comments_df["create_dt"].dt.date

    # 关键词相关评论（用于“相关占比”）
    ann_df = _load_relevant_comments(args.data_dir)
    if ann_df is not None and "create_dt" in ann_df.columns:
        ann_df["create_dt"] = _ensure_datetime_tz(ann_df["create_dt"])
        ann_df = ann_df.loc[
            (ann_df["create_dt"].notna())
            & (ann_df["create_dt"] >= event_start)
            & (ann_df["create_dt"] <= event_end)
        ].copy()
        ann_df["date"] = ann_df["create_dt"].dt.date
        relevant_by_date = ann_df.groupby("date").size().rename("relevant_count")
    else:
        # 兜底：annotation_pool 缺失则用 content_clean 做关键词命中
        kw_pattern = "|".join(RELEVANCE_KEYWORDS)
        relevant_mask = comments_df["content_clean"].astype(str).str.contains(kw_pattern, na=False, regex=False)
        relevant_by_date = (
            comments_df.loc[relevant_mask]
            .groupby("date")
            .size()
            .rename("relevant_count")
        )

    # 全量聚合（按日）
    all_by_date = comments_df.groupby("date")
    daily_total = all_by_date.size().rename("total_comments")
    daily_like_median = all_by_date["like_count_clean"].median().rename("like_median")
    daily_like_p25 = all_by_date["like_count_clean"].quantile(0.25).rename("like_p25")
    daily_like_p75 = all_by_date["like_count_clean"].quantile(0.75).rename("like_p75")

    daily_top = comments_df.loc[comments_df["is_top_comment"]].groupby("date").size().rename("top_comments")
    daily_sub = comments_df.loc[~comments_df["is_top_comment"]].groupby("date").size().rename("sub_comments")

    daily_susp = (
        comments_df.loc[comments_df["is_suspicious_user"]]
        .groupby("date")
        .size()
        .rename("suspicious_comments")
    )

    # 构建完整日期轴，便于你判断“数据缺口 vs 真实沉默”
    date_index = pd.date_range(event_start.date(), event_end.date(), freq="D").date
    summary = pd.DataFrame({"date": date_index}).set_index("date")
    summary = summary.join(daily_total, how="left")
    summary = summary.join(relevant_by_date, how="left")
    summary = summary.join(daily_like_median, how="left")
    summary = summary.join(daily_like_p25, how="left")
    summary = summary.join(daily_like_p75, how="left")
    summary = summary.join(daily_top, how="left")
    summary = summary.join(daily_sub, how="left")
    summary = summary.join(daily_susp, how="left")

    summary["total_comments"] = summary["total_comments"].fillna(0).astype(int)
    summary["relevant_count"] = summary["relevant_count"].fillna(0).astype(int)
    summary["top_comments"] = summary["top_comments"].fillna(0).astype(int)
    summary["sub_comments"] = summary["sub_comments"].fillna(0).astype(int)
    summary["suspicious_comments"] = summary["suspicious_comments"].fillna(0).astype(int)

    summary = summary.reset_index()
    summary["relevant_ratio"] = np.where(summary["total_comments"] > 0, summary["relevant_count"] / summary["total_comments"], np.nan)
    summary["suspicious_ratio"] = np.where(summary["total_comments"] > 0, summary["suspicious_comments"] / summary["total_comments"], np.nan)

    summary_path = args.outdir / "daily_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 绘图（统一使用 date->Timestamp）
    x_ts = pd.to_datetime(summary["date"].astype(str))
    out1 = args.outdir / "01_total_comments_daily.png"
    out2 = args.outdir / "02_relevant_ratio_daily.png"
    out3 = args.outdir / "03_like_median_daily.png"
    out4 = args.outdir / "04_top_vs_sub_daily.png"
    out5 = args.outdir / "05_suspicious_daily.png"

    _maybe_plot_line(
        x=list(x_ts),
        y=summary["total_comments"].to_numpy(),
        title="每日评论量（create_dt）",
        ylabel="评论数",
        out_path=out1,
        label="total_comments",
    )

    _maybe_plot_line(
        x=list(x_ts),
        y=summary["relevant_ratio"].to_numpy(),
        title="每日关键词相关占比（相关评论数/总评论数）",
        ylabel="相关占比",
        out_path=out2,
        label="relevant_ratio",
    )

    # like_count：画 median + p25/p75 区间
    plt.figure(figsize=(13, 4.8))
    plt.plot(x_ts, summary["like_median"], linewidth=1.8, label="like_median")
    plt.fill_between(x_ts, summary["like_p25"], summary["like_p75"], alpha=0.22, label="p25~p75")
    plt.title("每日 like_count_clean 中位数（以及 IQR 区间）")
    plt.xlabel("日期")
    plt.ylabel("like_count_clean")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out3, dpi=160)
    plt.close()

    # 一级 vs 二级
    plt.figure(figsize=(13, 4.8))
    plt.plot(x_ts, summary["top_comments"], linewidth=1.6, label="top_comments(一级)")
    plt.plot(x_ts, summary["sub_comments"], linewidth=1.6, label="sub_comments(二级)")
    plt.title("每日一级 vs 二级评论量（create_dt）")
    plt.xlabel("日期")
    plt.ylabel("评论数")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out4, dpi=160)
    plt.close()

    # 疑似机器人（含占比）
    plt.figure(figsize=(13, 4.8))
    ax1 = plt.gca()
    ax1.plot(x_ts, summary["suspicious_comments"], linewidth=1.8, color="#d62728", label="suspicious_comments")
    ax1.set_ylabel("疑似机器人评论数")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(x_ts, summary["suspicious_ratio"], linewidth=1.5, color="#1f77b4", label="suspicious_ratio")
    ax2.set_ylabel("疑似占比")
    plt.title("疑似机器人评论的时间分布（数量 + 占比）")
    # 合并图例
    lines, labels = [], []
    for ax in [ax1, ax2]:
        handles, ax_labels = ax.get_legend_handles_labels()
        lines.extend(handles)
        labels.extend(ax_labels)
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(out5, dpi=160)
    plt.close()

    # 文字摘要（峰值/缺口）
    print("\n=== 时间分布诊断摘要 ===")
    print(f"事件窗口: {event_start} ~ {event_end}")
    print(f"聚合输出: {summary_path}")
    print(f"图片输出目录: {args.outdir}")

    zero_days = summary.loc[summary["total_comments"] == 0, "date"].astype(str).tolist()
    if zero_days:
        print(f"潜在数据缺口（总评论为 0 的天数）: {len(zero_days)} 天，示例: {', '.join(zero_days[:10])}")

    peaks_total = _topk_dates(summary, "total_comments", k=5)
    print(f"\n[1] 每日评论量峰值 Top-5: {peaks_total}")

    peaks_relevant = _topk_dates(summary, "relevant_ratio", k=5)
    print(f"[2] 相关占比峰值 Top-5: {peaks_relevant}")

    peaks_like = _topk_dates(summary, "like_median", k=5)
    print(f"[3] like_count_clean 中位数峰值 Top-5: {peaks_like}")

    peaks_top = _topk_dates(summary, "top_comments", k=5)
    peaks_sub = _topk_dates(summary, "sub_comments", k=5)
    print(f"[4] 一级评论峰值 Top-5: {peaks_top}")
    print(f"[4] 二级评论峰值 Top-5: {peaks_sub}")

    peaks_susp = _topk_dates(summary, "suspicious_comments", k=5)
    print(f"[5] 疑似机器人评论峰值 Top-5: {peaks_susp}")

    if not args.no_show:
        # 默认不弹窗，避免你跑脚本时阻塞；你可以手动取消 --no-show 来调试
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
