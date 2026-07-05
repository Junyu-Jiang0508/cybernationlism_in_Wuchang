#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam 商店评论爬虫：通过官方公开 API 抓取指定游戏的玩家评论。
文档参考: https://store.steampowered.com/appreviews/<AppID>?json=1
"""

import pathlib
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ============ 配置（按需修改）============
APP_ID = 2277560  # WUCHANG: Fallen Feathers（明末：渊虚之羽）store.steampowered.com/app/2277560
MAX_REVIEWS = 50000 # 最多抓取条数；设为 None 表示一直翻到没有新数据
NUM_PER_PAGE = 100
LANGUAGE = "schinese"
FILTER = "recent"  # recent | all | updated 等
OUTPUT_CSV = str(
    pathlib.Path(__file__).resolve().parents[2]
    / "00_output" / "01_Raw_data" / "03_Steam" / "steam_reviews.csv"
)
REQUEST_INTERVAL_SEC = 0.35  # 请求间隔，避免过快
# 商店页「简体中文」评论总数，用于进度「已抓 / 总数」；若填 None 则只显示已抓条数（不显示百分比）
EXPECTED_TOTAL_REVIEWS = 33426
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

API_URL = "https://store.steampowered.com/appreviews/{appid}"


def _playtime_display(minutes: Optional[int]) -> str:
    """将分钟转为可读游玩时长（以评论时为准）。"""
    if minutes is None:
        return ""
    if minutes < 60:
        return f"{minutes} 分钟"
    h = minutes / 60.0
    return f"{h:.1f} 小时"


def _ts_to_datetime(ts: Optional[int]) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def fetch_page(
    app_id: int,
    cursor: str,
    session: requests.Session,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "json": "1",
        "language": LANGUAGE,
        "filter": FILTER,
        "num_per_page": NUM_PER_PAGE,
    }
    if cursor:
        params["cursor"] = cursor

    url = API_URL.format(appid=app_id)
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_reviews(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in payload.get("reviews") or []:
        author = r.get("author") or {}
        play_mins = author.get("playtime_at_review")
        rows.append(
            {
                "用户名": author.get("personaname", ""),
                "游玩时长_分钟": play_mins if play_mins is not None else "",
                "游玩时长_可读": _playtime_display(play_mins),
                "是否推荐": bool(r.get("voted_up")),
                "评论文本": (r.get("review") or "").replace("\r\n", "\n").strip(),
                "点赞数": int(r.get("votes_up") or 0),
                "发布时间": _ts_to_datetime(r.get("timestamp_created")),
                "recommendationid": r.get("recommendationid", ""),
            }
        )
    return rows


def _print_progress(
    current: int,
    expected_total: Optional[int],
    max_reviews: Optional[int],
    end: str = "\r",
) -> None:
    """单行刷新进度（换游戏时改 EXPECTED_TOTAL_REVIEWS 与商店页一致）。"""
    parts = [f"进度: {current:,}"]
    if expected_total is not None and expected_total > 0:
        pct = min(100.0, 100.0 * current / expected_total)
        parts.append(f"/ {expected_total:,} ({pct:.1f}%)")
    if max_reviews is not None:
        parts.append(f"[上限 {max_reviews:,}]")
    bar_w = 28
    if expected_total and expected_total > 0:
        filled = int(bar_w * min(1.0, current / expected_total))
        bar = "█" * filled + "░" * (bar_w - filled)
        parts.append(f"|{bar}|")
    msg = " ".join(parts)
    sys.stdout.write(msg + " " * max(0, 80 - len(msg)) + end)
    sys.stdout.flush()


def crawl_steam_reviews(
    app_id: int,
    max_reviews: Optional[int] = None,
    expected_total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    seen_ids: set = set()
    cursor = ""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    first_page = True

    while True:
        if max_reviews is not None and len(all_rows) >= max_reviews:
            break

        data = fetch_page(app_id, cursor, session)
        if not data.get("success"):
            print("\nAPI 返回 success!=1，停止。")
            break

        if first_page:
            first_page = False
            qs = data.get("query_summary") or {}
            tr = qs.get("total_reviews")
            if tr is not None:
                print(
                    f"API 汇总 total_reviews={tr:,}（全语言等，与简体中文条数可能不一致）"
                )

        batch = parse_reviews(data)
        if not batch:
            print("\n本页无评论，结束翻页。")
            break

        new_count = 0
        for row in batch:
            rid = row.get("recommendationid")
            if rid and rid in seen_ids:
                continue
            if rid:
                seen_ids.add(rid)
            all_rows.append(row)
            new_count += 1
            if max_reviews is not None and len(all_rows) >= max_reviews:
                break

        _print_progress(len(all_rows), expected_total, max_reviews)

        if new_count == 0:
            print("\n未出现新评论（可能已到末尾或重复），结束。")
            break

        next_cursor = data.get("cursor") or ""
        if not next_cursor or next_cursor == cursor:
            print("\n无下一页 cursor，已全部抓取。")
            break
        cursor = next_cursor

        time.sleep(REQUEST_INTERVAL_SEC)

    if max_reviews is not None and len(all_rows) > max_reviews:
        all_rows = all_rows[:max_reviews]

    sys.stdout.write("\n")
    sys.stdout.flush()
    return all_rows


def main() -> None:
    print(f"开始抓取 AppID={APP_ID}，语言={LANGUAGE}，排序={FILTER}，每页={NUM_PER_PAGE}")
    if MAX_REVIEWS is None:
        print("上限：直到没有新数据")
    else:
        print(f"上限：最多 {MAX_REVIEWS} 条")

    rows = crawl_steam_reviews(APP_ID, MAX_REVIEWS, EXPECTED_TOTAL_REVIEWS)
    if not rows:
        print("未获取到任何评论。")
        return

    df = pd.DataFrame(rows)
    # 导出时可去掉内部 id，若需要可追溯可保留
    out_cols = [
        "用户名",
        "游玩时长_分钟",
        "游玩时长_可读",
        "是否推荐",
        "评论文本",
        "点赞数",
        "发布时间",
    ]
    df[out_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存 {len(rows):,} 条评论 -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
