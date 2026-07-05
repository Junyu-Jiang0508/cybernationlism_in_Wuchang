# -*- coding: utf-8 -*-
"""
B 站评论清洗全流程：快照 → 去重 → 时间过滤 → 内容过滤 → 文本标准化
                  → 机器人标记 → 关键词筛选 → 三路输出 → 四关键数字。

输入默认: 01_data/search_comments_all.csv（约 174 万行，分块处理）
输出目录: 01_data/
  search_comments_cleaned.csv  — 全量清洗数据（去重+过滤+标准化）
  annotation_pool.csv          — 关键词相关评论（用于标注）
  comments_clean.parquet/csv   — 回归分析用（pyarrow 存在则 parquet，否则 csv）
  sbert_sample.csv             — SBERT 聚类抽样
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ── 可选依赖（缺少时给出安装提示，不中断） ───────────────────────────────────
try:
    import opencc as _opencc
    _converter = _opencc.OpenCC("t2s")
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False
    print(
        "[WARNING] opencc 未安装，跳过繁简转换。"
        "如需安装: pip install opencc-python-reimplemented",
        file=sys.stderr,
    )

try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    print(
        "[WARNING] pyarrow 未安装，parquet 输出将改为 csv。"
        "如需安装: pip install pyarrow",
        file=sys.stderr,
    )

# ── 路径 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT  = SCRIPT_DIR / "01_data" / "search_comments_all.csv"
DEFAULT_OUTDIR = SCRIPT_DIR / "01_data"

# ── 保留列 ────────────────────────────────────────────────────────────────────
KEEP_COLS: List[str] = [
    "comment_id", "parent_comment_id", "create_time",
    "video_id", "content", "user_id",
    "sub_comment_count", "like_count",
]

# ── 时间窗口（北京时间）—— 《明末：渊虚之羽》发售前后 ─────────────────────────
EVENT_START = pd.Timestamp("2025-07-23", tz="Asia/Shanghai")
EVENT_END   = pd.Timestamp("2025-12-31 23:59:59", tz="Asia/Shanghai")

# ── 关键词（宽口径，命中任意一个即为"相关"） ───────────────────────────────────
RELEVANCE_KEYWORDS: List[str] = [
    "明末", "渊虚之羽", "游戏", "开发商",
    "汉奸", "辱华", "历史", "满清", "清朝", "殖民",
    "民族", "国产", "国货", "文化", "传统",
    "道歉", "抵制", "下架", "删改", "和解",
    "愤怒", "失望", "支持", "感动",
]

CHUNKSIZE       = 80_000
ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "gb18030", "gbk"]
ENCODING_ERRORS  = "replace"
SBERT_PER_VIDEO  = 20   # 每个视频最多抽取多少条进 SBERT 样本
BOT_QUANTILE     = 0.999

# like_count 的合理上界（B站点赞数一般不会到这个量级）
# 用于标记/截断“数据类型崩坏/溢出”的条目，避免均值/回归被污染。
LIKE_COUNT_CLIP_UPPER = 1_000_000
SUB_COMMENT_CLIP_UPPER = 1_000_000


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _probe_encoding(path: Path) -> str:
    """读前 5000 行，选替换字符最少的编码。"""
    best_enc, best_rep = "utf-8-sig", None
    for enc in ENCODINGS_TO_TRY:
        try:
            probe = pd.read_csv(
                path, encoding=enc, encoding_errors=ENCODING_ERRORS,
                usecols=KEEP_COLS, dtype=str, nrows=5000, low_memory=False,
            )
            if not set(KEEP_COLS).issubset(probe.columns):
                continue
            rep_cnt = int(probe["content"].astype(str).str.count("\ufffd").sum())
            if best_rep is None or rep_cnt < best_rep:
                best_rep, best_enc = rep_cnt, enc
        except Exception:
            continue
    return best_enc


# 内容有效性 ── 正则预编译
_RE_EMOJI_ONLY = re.compile(r"^(\[[\w_]+\]\s*)+$")
_RE_NON_TEXT   = re.compile(r"^[\d\s\W]+$")
_RE_CHINESE    = re.compile(r"[\u4e00-\u9fff]")


def is_valid_content(text) -> bool:
    """返回 True 表示保留。"""
    if pd.isna(text):
        return False
    text = str(text).strip()
    if not text:
        return False
    if _RE_EMOJI_ONLY.fullmatch(text):     # 纯表情，如 [doge][doge]
        return False
    if _RE_NON_TEXT.fullmatch(text):       # 纯数字/标点
        return False
    if len(_RE_CHINESE.findall(text)) < 2:  # 少于 2 个汉字
        return False
    return True


_TS_MIN_S = 1_000_000_000.0   # ~year 2001，合理下界
_TS_MAX_S = 2_500_000_000.0   # ~year 2049，安全上界


def _ts_to_seconds(ts: "pd.Series") -> "pd.Series":
    """
    将时间戳归一化到秒级，兼容秒/毫秒/微秒/纳秒混杂情况。
    1. 只要仍 > _TS_MAX_S，就除以 1000，最多 3 次。
    2. 去除 inf / -inf。
    3. 范围外（< _TS_MIN_S 或 > _TS_MAX_S）的值置为 NaN，
       避免 pd.to_datetime 内部乘以 1e9 时溢出。
    """
    result = pd.to_numeric(ts, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    for _ in range(3):
        too_large = result.notna() & (result > _TS_MAX_S)
        if not too_large.any():
            break
        result = result.where(~too_large, result / 1000.0)
    result = result.where(
        result.isna() | ((result >= _TS_MIN_S) & (result <= _TS_MAX_S))
    )
    return result


def _ts_to_dt(ts_seconds: "pd.Series", tz: str = "Asia/Shanghai") -> "pd.Series":
    """将秒级时间戳转为 tz-aware datetime，逐元素处理以彻底避免 C 层溢出。"""
    def _convert(v):
        try:
            if pd.isna(v):
                return pd.NaT
            return pd.Timestamp(int(v), unit="s", tz="UTC").tz_convert(tz)
        except (OverflowError, ValueError, OSError):
            return pd.NaT
    return ts_seconds.apply(_convert)


# 文本标准化 ── 正则预编译
_RE_URL       = re.compile(r"https?://\S+")
_RE_AT        = re.compile(r"@[\w\-]+")
_RE_REPEAT    = re.compile(r"(.)\1{3,}")
_RE_EMOJI_VAR = re.compile(r"\[(\w+)_[\w]+\]")


def normalize_text(text) -> str:
    """繁简转换 + URL/@ 去除 + 重复字符折叠 + 表情变体标准化。"""
    text = str(text)
    if HAS_OPENCC:
        text = _converter.convert(text)
    text = _RE_URL.sub("", text)
    text = _RE_AT.sub("", text)
    text = _RE_REPEAT.sub(r"\1\1\1", text)           # 哈哈哈哈哈 → 哈哈哈
    text = _RE_EMOJI_VAR.sub(r"[\1]", text)          # [doge_金箍] → [doge]
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="B 站评论清洗全流程")
    parser.add_argument("--input",  type=Path, default=DEFAULT_INPUT,
                        help="原始评论 CSV 路径")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="输出目录（默认 01_data/）")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE,
                        help="分块行数（内存紧张可调小）")
    parser.add_argument("--no-time-filter", action="store_true",
                        help="跳过时间窗口过滤（用于调试或全量保留）")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"找不到输入文件: {args.input}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    cleaned_csv = args.outdir / "search_comments_cleaned.csv"

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1：分块清洗 + 快照统计同步进行
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 62)
    print("PASS 1 — 分块读取、清洗、写出 cleaned CSV")
    print("=" * 62)

    chosen_enc = _probe_encoding(args.input)
    print(f"  探测编码: {chosen_enc}")
    if not HAS_OPENCC:
        print("  [跳过] 繁简转换（opencc 未安装）")

    reader = pd.read_csv(
        args.input,
        encoding=chosen_enc,
        encoding_errors=ENCODING_ERRORS,
        usecols=KEEP_COLS,
        dtype=str,
        chunksize=args.chunksize,
        low_memory=False,
    )

    # 快照累计变量
    total_raw = 0
    null_counts: Counter = Counter()
    content_len_sample: list = []
    like_sample: list = []
    top_count = 0
    sub_count = 0
    video_ids: set = set()
    ts_min: Optional[float] = None
    ts_max: Optional[float] = None

    # 清洗状态
    seen_ids: set = set()
    total_cleaned = 0
    first_write = True
    cnt_dup = cnt_time_out = cnt_nat = cnt_content = 0
    cnt_empty_after_clean = 0

    for chunk_no, chunk in enumerate(reader):
        total_raw += len(chunk)

        # ── 快照：缺失率（原始）──────────────────────────────────────────────
        for col in KEEP_COLS:
            if col in chunk.columns:
                null_counts[col] += int(chunk[col].isna().sum())

        # ── 快照：视频 ID ─────────────────────────────────────────────────────
        video_ids.update(chunk["video_id"].dropna().astype(str).tolist())

        # ══ 清洗 Step 1：comment_id 全局去重 ═════════════════════════════════
        chunk["comment_id"] = chunk["comment_id"].astype(str).str.strip()
        before = len(chunk)
        chunk = chunk.drop_duplicates(subset=["comment_id"], keep="first")
        chunk = chunk[~chunk["comment_id"].isin(seen_ids)]
        seen_ids.update(chunk["comment_id"].tolist())
        cnt_dup += before - len(chunk)

        # ── 快照：一级 vs 二级评论（去重后、时间过滤前）────────────────────
        pid = chunk["parent_comment_id"].fillna("").astype(str)
        cid = chunk["comment_id"].fillna("").astype(str)
        is_top_mask = pid.isin(["", "nan", "0"]) | (pid == cid)
        top_count += int(is_top_mask.sum())
        sub_count += int((~is_top_mask).sum())

        # ══ 清洗 Step 2：时间窗口过滤 ════════════════════════════════════════
        if not args.no_time_filter:
            ts = _ts_to_seconds(pd.to_numeric(chunk["create_time"], errors="coerce"))
            dt = _ts_to_dt(ts)
            before = len(chunk)

            mask_nat = dt.isna()
            mask_window = dt.notna() & (dt >= EVENT_START) & (dt <= EVENT_END)
            mask_out = dt.notna() & ~mask_window

            cnt_nat += int(mask_nat.sum())
            cnt_time_out += int(mask_out.sum())

            chunk = chunk.loc[mask_window].copy()

        # ── 快照：create_time 范围（时间过滤后；若跳过则为去重后）──────
        ts_s = _ts_to_seconds(
            pd.to_numeric(chunk["create_time"], errors="coerce").dropna()
        )
        if len(ts_s):
            ts_min = ts_s.min() if ts_min is None else min(ts_min, ts_s.min())
            ts_max = ts_s.max() if ts_max is None else max(ts_max, ts_s.max())

        if chunk.empty:
            continue

        # ══ 清洗 Step 3：内容有效性过滤 ══════════════════════════════════════
        before = len(chunk)
        chunk = chunk.loc[chunk["content"].apply(is_valid_content)].copy()
        cnt_content += before - len(chunk)

        if chunk.empty:
            continue

        # ══ 清洗 Step 4：文本标准化 ══════════════════════════════════════════
        chunk["content_clean"] = chunk["content"].apply(normalize_text)
        # 标准化后可能出现空字符串（例如只剩 URL/@/表情被去掉）
        before = len(chunk)
        chunk = chunk.loc[chunk["content_clean"].astype(str).str.len() > 0].copy()
        cnt_empty_after_clean += before - len(chunk)

        # ── 快照：content_len / like_count 抽样统计 ──────────────────────────
        if len(content_len_sample) < 100_000:
            content_len_sample.extend(
                chunk["content_clean"].astype(str).str.len().tolist()
            )
        lk = pd.to_numeric(chunk["like_count"], errors="coerce").dropna()
        lk = lk.clip(lower=0, upper=LIKE_COUNT_CLIP_UPPER)  # 采样即截断，避免快照被污染
        if len(like_sample) < 100_000:
            like_sample.extend(lk.tolist())
        # （机器人检测阈值在 PASS 2 基于最终 df 重新计算）

        # ── 写出 ──────────────────────────────────────────────────────────────
        chunk.to_csv(
            cleaned_csv,
            mode="w" if first_write else "a",
            index=False,
            encoding="utf-8-sig",
            header=first_write,
        )
        total_cleaned += len(chunk)
        first_write = False

        if (chunk_no + 1) % 5 == 0:
            print(
                f"  进度: 已读 {total_raw:,} 原始行 → 保留 {total_cleaned:,} 行...",
                end="\r",
            )

    print()

    # ── 打印数据快照 ──────────────────────────────────────────────────────────
    sep = "─" * 62
    print(f"\n{'=' * 62}")
    print("=== 数据快照（基于原始文件）===")
    print(f"{'=' * 62}")
    print(f"  总行数        : {total_raw:,}")
    print(f"  涉及视频数    : {len(video_ids):,}")

    print(f"\n{sep}")
    print("  字段缺失率（原始）")
    for col in KEEP_COLS:
        rate = null_counts[col] / total_raw if total_raw else 0
        print(f"    {col:<24}: {rate:.2%}  ({null_counts[col]:,} 行)")

    print(f"\n{sep}")
    print("  content_clean 文本长度分布（清洗后）")
    if content_len_sample:
        s = pd.Series(content_len_sample, dtype=float)
        print(
            s.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            .rename(lambda x: f"    {x}")
            .to_string()
        )

    print(f"\n{sep}")
    print("  like_count 分布（清洗后抽样）")
    if like_sample:
        s = pd.Series(like_sample, dtype=float)
        print(
            s.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])
            .rename(lambda x: f"    {x}")
            .to_string()
        )

    total_comments = top_count + sub_count
    print(f"\n{sep}")
    print("  一级 vs 二级评论（去重后、时间过滤前）")
    print(f"    一级评论: {top_count:,}  ({top_count/total_comments:.1%})")
    print(f"    二级评论: {sub_count:,}  ({sub_count/total_comments:.1%})")

    if ts_min is not None:
        print(f"\n{sep}")
        print(
            "  create_time 范围（北京时间"
            + ("，已跳过时间窗过滤" if args.no_time_filter else "，时间窗过滤后")
            + "）"
        )
        def _single_ts(v: float) -> str:
            try:
                return str(pd.Timestamp(int(v), unit="s", tz="UTC").tz_convert("Asia/Shanghai"))
            except Exception:
                return str(v)
        print(f"    最早: {_single_ts(ts_min)}")
        print(f"    最晚: {_single_ts(ts_max)}")

    print(f"\n{sep}")
    print("  清洗过滤摘要")
    print(f"    原始行数         : {total_raw:,}")
    print(f"    去重过滤         : {cnt_dup:,}")
    if args.no_time_filter:
        print("    时间窗口过滤     : 已跳过 --no-time-filter")
    else:
        print(f"    时间戳解析失败(NaT): {cnt_nat:,}")
        print(f"    时间窗外(有效)   : {cnt_time_out:,}")
    print(f"    内容有效性过滤   : {cnt_content:,}")
    print(f"    标准化后空文本过滤: {cnt_empty_after_clean:,}")
    print(f"    清洗后保留       : {total_cleaned:,}")

    if total_cleaned == 0:
        print("\n[ERROR] 清洗后无数据，请检查时间窗口或过滤条件（可加 --no-time-filter 调试）。")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2：读回 cleaned CSV → 机器人标记 + 关键词筛选 + 三路输出
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 62}")
    print("PASS 2 — 机器人标记 + 关键词筛选 + 三路输出")
    print(f"{'=' * 62}")

    df = pd.read_csv(cleaned_csv, encoding="utf-8-sig", dtype=str, low_memory=False)

    # 时间列
    df["create_time_num"] = _ts_to_seconds(
        pd.to_numeric(df["create_time"], errors="coerce")
    )
    df["create_dt"] = _ts_to_dt(df["create_time_num"])

    # 兜底：保证清洗后结果仍全部落在研究时间窗内
    mask_time = (
        df["create_dt"].notna()
        & (df["create_dt"] >= EVENT_START)
        & (df["create_dt"] <= EVENT_END)
    )
    if not mask_time.all():
        before = len(df)
        df = df.loc[mask_time].copy()
        print(f"  [兜底过滤] 时间窗外数据已剔除: {before - len(df):,} 行")

    # 强制检查：清洗后不应再有时间窗外的 create_dt
    if df["create_dt"].notna().any():
        max_dt = df["create_dt"].max()
        min_dt = df["create_dt"].min()
        if (max_dt > EVENT_END) or (min_dt < EVENT_START):
            print(f"  [警告] 清洗后仍存在时间窗外数据: min={min_dt}, max={max_dt}")

    # 兜底：确保 standardized 文本不为空
    df = df.loc[df["content_clean"].astype(str).str.len() > 0].copy()

    # 数值列（防止数据类型崩坏/溢出污染）
    df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce")
    df["sub_comment_count"] = pd.to_numeric(df["sub_comment_count"], errors="coerce")

    df["like_count_overflow"] = (
        df["like_count"].notna()
        & ((df["like_count"] < 0) | (df["like_count"] > LIKE_COUNT_CLIP_UPPER))
    )
    df["sub_comment_overflow"] = (
        df["sub_comment_count"].notna()
        & ((df["sub_comment_count"] < 0) | (df["sub_comment_count"] > SUB_COMMENT_CLIP_UPPER))
    )
    df["like_count_clean"] = df["like_count"].clip(lower=0, upper=LIKE_COUNT_CLIP_UPPER)
    df["sub_comment_count_clean"] = df["sub_comment_count"].clip(
        lower=0, upper=SUB_COMMENT_CLIP_UPPER
    )

    # 快速诊断：看看溢出主要集中在什么视频/账号
    if bool(df["like_count_overflow"].any()):
        ov = df.loc[df["like_count_overflow"], ["video_id", "user_id"]]
        ov_top_vid = ov["video_id"].value_counts().head(10)
        ov_top_user = ov["user_id"].value_counts().head(10)
        print(f"  [诊断] like_count_overflow 命中行: {len(ov):,}")
        print(f"  [诊断] top overflow videos: {', '.join(f'{k}({v})' for k, v in ov_top_vid.items())}")
        print(f"  [诊断] top overflow users : {', '.join(f'{k}({v})' for k, v in ov_top_user.items())}")

    # 一级/二级标记
    pid = df["parent_comment_id"].fillna("").astype(str)
    cid = df["comment_id"].astype(str)
    df["is_top_comment"] = pid.isin(["", "nan", "0"]) | (pid == cid)

    # 内容重复（用于区分“爬虫重复抓取” vs “同内容被不同用户复述/转发”）
    df["is_duplicate_content"] = df["content_clean"].astype(str).duplicated(keep=False)

    # ── Step 5：机器人 / 刷屏账号标记 ────────────────────────────────────────
    user_counter_final = Counter(df["user_id"].dropna().astype(str).tolist())
    user_ser = pd.Series(user_counter_final, dtype=float)
    if len(user_ser):
        bot_threshold = float(user_ser.quantile(BOT_QUANTILE))
        suspicious_users = set(user_ser[user_ser > bot_threshold].index)
    else:
        bot_threshold = float("nan")
        suspicious_users = set()
    df["is_suspicious_user"] = df["user_id"].isin(suspicious_users)

    # ── 关键词过滤 ────────────────────────────────────────────────────────────
    kw_pattern = "|".join(re.escape(k) for k in RELEVANCE_KEYWORDS)
    df["is_relevant"] = (
        df["content_clean"].astype(str).str.contains(kw_pattern, na=False)
    )
    relevant_df = df[df["is_relevant"]].copy()

    # ── 输出 1：annotation_pool.csv ──────────────────────────────────────────
    ann_cols = [
        "comment_id",
        "video_id",
        "content",
        "content_clean",
        "create_dt",
        "like_count_clean",
        "like_count_overflow",
        "user_id",
        "is_suspicious_user",
    ]
    ann_path = args.outdir / "annotation_pool.csv"
    relevant_df[[c for c in ann_cols if c in relevant_df.columns]].to_csv(
        ann_path, index=False, encoding="utf-8-sig"
    )
    print(f"  [1] annotation_pool.csv      : {len(relevant_df):,} 行 → {ann_path.name}")

    # ── 输出 2：comments_clean.parquet（或 .csv）─────────────────────────────
    reg_cols = [
        "comment_id",
        "video_id",
        "content_clean",
        "create_dt",
        "like_count_clean",
        "like_count_overflow",
        "sub_comment_count_clean",
        "sub_comment_overflow",
        "user_id",
        "is_top_comment",
        "is_suspicious_user",
        "is_duplicate_content",
    ]
    reg_df = df[[c for c in reg_cols if c in df.columns]].copy()
    if HAS_PYARROW:
        out2 = args.outdir / "comments_clean.parquet"
        reg_df.to_parquet(out2, index=False)
        print(f"  [2] comments_clean.parquet   : {len(reg_df):,} 行 → {out2.name}")
    else:
        out2 = args.outdir / "comments_clean.csv"
        reg_df.to_csv(out2, index=False, encoding="utf-8-sig")
        print(f"  [2] comments_clean.csv       : {len(reg_df):,} 行 → {out2.name}")

    # ── 输出 3：sbert_sample.csv ──────────────────────────────────────────────
    # 优先从关键词相关评论中抽样，保证 SBERT 聚类/主题发现聚焦你的研究语料。
    sbert_source = relevant_df if len(relevant_df) > 0 else df
    sbert_parts = []
    for _vid, grp in sbert_source.groupby("video_id"):
        n = min(len(grp), SBERT_PER_VIDEO)
        sbert_parts.append(grp.sample(n, random_state=42))
    sbert_df = pd.concat(sbert_parts, ignore_index=True) if sbert_parts else df.iloc[:0]
    out3 = args.outdir / "sbert_sample.csv"
    sbert_df[["comment_id", "video_id", "content_clean"]].to_csv(
        out3, index=False, encoding="utf-8-sig"
    )
    print(
        f"  [3] sbert_sample.csv         : {len(sbert_df):,} 行 → {out3.name}"
        f"（抽样来源: {'relevant' if len(relevant_df) > 0 else 'all'}）"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 四个关键数字
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 62}")
    print("=== 四个关键数字（决策依据）===")
    print(f"{'=' * 62}")

    n = len(df)

    print(f"\n[1] 关键词相关评论数: {len(relevant_df):,} / {n:,}  "
          f"({len(relevant_df)/n:.1%})")
    print("    → 决定标注策略：相关量大可抽样+众包；量小可全量人工")

    lk_clean = df["like_count_clean"].dropna()
    lk_p99 = lk_clean.quantile(0.99) if len(lk_clean) else float("nan")
    overflow_rate = float(df["like_count_overflow"].mean()) if "like_count_overflow" in df.columns else 0.0
    lk_max = lk_clean.max() if len(lk_clean) else float("nan")
    print(
        f"\n[2] like_count_clean 99th pct: {lk_p99:.0f} "
        f"(mean={lk_clean.mean():.1f}  median={lk_clean.median():.1f}  max={lk_max:.0f})"
    )
    print(f"    like_count_overflow: {overflow_rate:.2%}（> {LIKE_COUNT_CLIP_UPPER:,} 或 <0）")
    print("    → 因变量建议用 like_count_clean + log1p 或截尾")

    top_n = int(df["is_top_comment"].sum())
    print(f"\n[3] 一级评论占比: {top_n:,} / {n:,}  ({top_n/n:.1%})")
    print("    → 决定是否分开建模（一级评论 vs 回复链）")

    bot_n = int(df["is_suspicious_user"].sum())
    print(f"\n[4] 疑似机器人评论占比: {bot_n:,} / {n:,}  ({bot_n/n:.2%})")
    print(
        f"    阈值来源: 用户评论数 > {BOT_QUANTILE*100:.1f}th pct（即 > {bot_threshold:.0f}）"
        f"，涉及账号 {len(suspicious_users):,} 个"
    )
    print("    → 建议：仅做敏感性分析（不建议直接删除）")

    print("\n完成。")


if __name__ == "__main__":
    main()
