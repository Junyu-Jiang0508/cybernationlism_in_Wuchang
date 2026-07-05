# -*- coding: utf-8 -*-
"""
数据准备 1.1：探索 + 样本A(3万分层) + 样本B(2000，关键词+随机)

用法:
  python 08_data_sampling_pipeline.py --explore-only          # 仅探索
  python 08_data_sampling_pipeline.py --full                  # 探索 + 写样本与 log
  python 08_data_sampling_pipeline.py --full --seed 42

输入默认: 01_data/search_comments_cleaned.csv
输出目录: data/
  - raw_cleaned.csv       （首次 --full 时从输入复制，便于归档）
  - sample_A_30k.csv
  - sample_B_2000.csv     （含 keyword_hit, keyword_tier 等列）
  - sampling_log.txt
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = SCRIPT_DIR / "01_data" / "search_comments_cleaned.csv"
DATA_DIR = SCRIPT_DIR / "data"
TZ = "Asia/Shanghai"

# ── 时间分层（北京时间）────────────────────────────────────────
EARLY_START = pd.Timestamp("2025-07-24", tz=TZ)
EARLY_END = pd.Timestamp("2025-08-15 23:59:59", tz=TZ)
MID_START = pd.Timestamp("2025-08-16", tz=TZ)
MID_END = pd.Timestamp("2025-10-31 23:59:59", tz=TZ)
LATE_START = pd.Timestamp("2025-11-01", tz=TZ)
LATE_END = pd.Timestamp("2025-12-31 23:59:59", tz=TZ)

PEAK_START = pd.Timestamp("2025-07-24", tz=TZ)
PEAK_END = pd.Timestamp("2025-08-31 23:59:59", tz=TZ)

# 各段目标条数（样本A 总计 3 万）
SEGMENT_TARGETS = {
    "early": 15_000,  # 50%
    "mid": 9_000,  # 30%
    "late": 6_000,  # 20%
}
LIKE_HIGH_THRESHOLD = 15  # 高互动：>15
TIER_FRAC_HIGH = 0.30
TIER_FRAC_MID = 0.30  # 低互动层取剩余份额（约 0.40）

# 视频截断：若 Top10 视频评论占比 > 50%，则每视频在 A 中上限
VIDEO_CAP_IF_CONCENTRATED = 300
TOP10_CONC_THRESHOLD = 0.50

# 样本 B
B_TOTAL = 2000
B_FROM_KEYWORD = 1200
B_FROM_A_RANDOM = 800

# ── 关键词（重构分组）────────────────────────────────────────
KW_GROUP1_TRAUMA = [
    "扬州十日",
    "嘉定三屠",
    "留发不留头",
    "留头不留发",
    "剃发令",
    "薙发",
    "屠城",
    "血债",
    "五胡乱华",
    "衣冠南渡",
    "崖山之后",
]
KW_GROUP2_ATTACK = [
    "汉奸",
    "卖国贼",
    "走狗",
    "跪族",
    "跪清",
    "认贼作父",
    "数典忘祖",
    "精汉",
    "辱汉",
    "斯德哥尔摩",
    "奴才思维",
    "建奴粉",
    "鞑粉",
]
KW_GROUP3_ORTHODOXY = [
    "驱除鞑虏",
    "反清复明",
    "华夷之辨",
    "华夏正统",
    "衣冠文明",
    "恢复中华",
    "复我汉家",
    "建奴",
    "鞑子",
    "胡人",
]
KW_GROUP4_GAME_PHRASES = [
    "删了清兵",
    "没有清军",
    "为什么删除清",
    "跪清游戏",
    "媚清",
    "洗白满清",
    "美化满清",
    "丑化汉人",
    "历史虚无主义",
    "篡改历史",
    "政治正确",
    "自我审查",
]
# 第四组需与游戏/议题锚词共现
KW_GROUP4_ANCHORS = [
    "明末",
    "渊虚",
    "武昌",
    "起义",
    "游戏",
    "开发商",
    "清兵",
    "清军",
    "删改",
    "历史",
    "道歉",
    "国产",
]
KW_GROUP5_MOBILIZE = [
    "汉家儿郎",
    "炎黄子孙",
    "汉魂",
    "民族气节",
    "不忘国耻",
    "铭记历史",
    "正统汉服",
    "华夏儿女",
]

NEGATION_EXCLUDE = [
    "反对民族主义",
    "理性看待",
    "不要极端",
    "别上纲上线",
    "不要极端化",
    "拒绝极端",
]

# 宽口径：仅用于扩大「疑似民族主义」抽样池（keyword_hit 仍为上面的精炼规则）
# 刻意不含单独「汉族」「满清」等高频中性词，减少科普评论误入
KW_BROAD_FOR_POOL = [
    "汉奸",
    "满遗",
    "精汉",
    "辱汉",
    "抗清",
    "驱除鞑虏",
    "汉家",
    "汉文化",
    "民族气节",
    "清军",
    "剃发易服",
    "历史虚无",
    "删除历史",
    "媚外",
    "跪舔",
    "洗白",
    "辱华",
    "不尊重历史",
    "删了清兵",
    "没有清军",
    "为什么没有清",
    "走狗",
    "跪族",
    "恨国党",
    "卖国",
    "鞑虏",
    "反清复明",
    "华夷",
    "建奴",
    "鞑子",
    "清兵",
    "虚无主义",
    "篡改历史",
    "自我审查",
    "政治正确",
    "明粉",
    "清粉",
    "包衣",
    "遗老",
    "洋大人",
    "软骨病",
    "崇洋",
    "删改",
    "下架",
    "抵制",
    "辱明",
    "黑明",
    "美化清朝",
    "丑化明朝",
    "大汉族",
    "皇汉",
    "少贵",
    "一等洋",
    "辫子戏",
    "金钱鼠尾",
    "通古斯",
    "元清非中国",
    "崖山",
    "汉服",
    "华夏",
    "祖宗",
    "国耻",
]


def _log_line(buf: StringIO, msg: str) -> None:
    buf.write(msg + "\n")


def is_top_level(pid, cid) -> bool:
    p = "" if pd.isna(pid) else str(pid).strip().lower()
    c = "" if pd.isna(cid) else str(cid).strip()
    return p in ("", "nan", "0", "none") or p == c


def load_cleaned(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if "content_clean" in df.columns:
                return df
        except Exception:
            continue
    raise SystemExit(f"无法读取: {path}")


def parse_create_dt(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(TZ)


def run_exploration(df: pd.DataFrame, log: StringIO) -> dict:
    n = len(df)
    pid, cid = df["parent_comment_id"], df["comment_id"]
    top_mask = pd.Series([is_top_level(a, b) for a, b in zip(pid, cid)], index=df.index)
    n_top = int(top_mask.sum())

    lk = pd.to_numeric(df["like_count"], errors="coerce").fillna(0).astype(int)
    zero_rate = float((lk == 0).mean())

    vc = df["video_id"].astype(str).value_counts()
    n_vid = len(vc)
    top10_share = float(vc.head(10).sum() / n) if n else 0.0

    dt = parse_create_dt(df["create_time"])
    daily = dt.dt.floor("D").value_counts().sort_index()
    peak_mask = (dt >= PEAK_START) & (dt <= PEAK_END)
    peak_n = int(peak_mask.sum())

    _log_line(log, "\n=== 数据探索 ===")
    _log_line(log, f"总评论数: {n:,}")
    _log_line(log, f"一级评论数（parent 空/0/等于自身）: {n_top:,} ({n_top/n:.1%})")
    _log_line(log, f"like_count=0 占比: {zero_rate:.2%}")
    _log_line(log, f"视频数: {n_vid:,} | Top10 视频评论占比: {top10_share:.1%}")
    _log_line(log, f"7/24–8/31（高峰期）评论数: {peak_n:,} ({peak_n/n:.1%})")
    if len(daily):
        top_days = daily.nlargest(8)
        _log_line(log, "评论量最高的 8 天:")
        for d, c in top_days.items():
            _log_line(log, f"  {d.date()}: {c:,}")

    return {
        "n": n,
        "n_top": n_top,
        "top10_share": top10_share,
        "zero_rate": zero_rate,
    }


def build_eligible_a(df: pd.DataFrame, log: Optional[StringIO], label: str = "") -> pd.DataFrame:
    """一级 + 清洗后长度 >= 5"""
    pid, cid = df["parent_comment_id"], df["comment_id"]
    top_mask = pd.Series([is_top_level(a, b) for a, b in zip(pid, cid)], index=df.index)
    cc = df["content_clean"].astype(str)
    len_mask = cc.str.len() >= 5
    sub = df.loc[top_mask & len_mask].copy()
    sub["_dt"] = parse_create_dt(sub["create_time"])
    sub["_like"] = pd.to_numeric(sub["like_count"], errors="coerce").fillna(0).astype(int)
    if log is not None:
        _log_line(log, f"\n[过滤{label}] 一级且 content_clean 长度>=5: {len(sub):,}")
    return sub


def assign_segment(dt: pd.Series) -> pd.Series:
    seg = pd.Series("out", index=dt.index, dtype=object)
    seg.loc[(dt >= EARLY_START) & (dt <= EARLY_END)] = "early"
    seg.loc[(dt >= MID_START) & (dt <= MID_END)] = "mid"
    seg.loc[(dt >= LATE_START) & (dt <= LATE_END)] = "late"
    return seg


def stratified_sample_segment(
    pool: pd.DataFrame,
    segment: str,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    p = pool[pool["_segment"] == segment]
    if len(p) == 0 or target_n <= 0:
        return p.iloc[:0]

    high = p[p["_like"] > LIKE_HIGH_THRESHOLD]
    mid = p[(p["_like"] >= 2) & (p["_like"] <= LIKE_HIGH_THRESHOLD)]
    low = p[p["_like"] <= 1]

    nh = int(round(target_n * TIER_FRAC_HIGH))
    nm = int(round(target_n * TIER_FRAC_MID))
    nl = target_n - nh - nm

    parts: List[pd.DataFrame] = []

    # 高互动：按点赞降序取 top nh
    high_take = min(nh, len(high))
    if high_take > 0:
        parts.append(high.nlargest(high_take, "_like"))

    # 中、低：随机
    mid_take = min(nm, len(mid))
    if mid_take > 0:
        parts.append(mid.sample(n=mid_take, random_state=rng))

    low_take = min(nl, len(low))
    if low_take > 0:
        parts.append(low.sample(n=low_take, random_state=rng))

    got = sum(len(x) for x in parts)
    short = target_n - got
    if short > 0:
        used_idx = set()
        for part in parts:
            used_idx.update(part.index.tolist())
        remain = p.loc[~p.index.isin(used_idx)]
        if len(remain) > 0:
            parts.append(remain.sample(n=min(short, len(remain)), random_state=rng))

    out = pd.concat(parts, ignore_index=False) if parts else p.iloc[:0]
    out = out[~out.index.duplicated(keep="first")]
    return out


def apply_video_cap(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if cap <= 0 or len(df) == 0:
        return df
    kept: List[pd.DataFrame] = []
    for vid, grp in df.groupby("video_id"):
        if len(grp) <= cap:
            kept.append(grp)
        else:
            kept.append(grp.sample(n=cap, random_state=rng))
    return pd.concat(kept, ignore_index=True)


def supplement_to_n(
    current: pd.DataFrame,
    eligible: pd.DataFrame,
    target: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    have = set(current["comment_id"].astype(str))
    pool = eligible.loc[~eligible["comment_id"].astype(str).isin(have)]
    need = target - len(current)
    if need <= 0 or len(pool) == 0:
        return current
    add = pool.sample(n=min(need, len(pool)), random_state=rng)
    return pd.concat([current, add], ignore_index=True)


def _compile_or(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(re.escape(p) for p in patterns))


def keyword_score_row(text: str) -> Tuple[int, str]:
    """
    返回 (是否命中 0/1, 分层标签)
    G4 需短语命中且与锚词共现；G5 单独；否定句排除。
    """
    t = text if isinstance(text, str) else str(text)
    if not t:
        return 0, ""
    for neg in NEGATION_EXCLUDE:
        if neg in t:
            return 0, "excluded_negation"

    g1 = _compile_or(KW_GROUP1_TRAUMA)
    g2 = _compile_or(KW_GROUP2_ATTACK)
    g3 = _compile_or(KW_GROUP3_ORTHODOXY)
    g4p = _compile_or(KW_GROUP4_GAME_PHRASES)
    g4a = _compile_or(KW_GROUP4_ANCHORS)
    g5 = _compile_or(KW_GROUP5_MOBILIZE)

    hit_tiers: List[str] = []
    if g1.search(t):
        hit_tiers.append("G1_trauma")
    if g2.search(t):
        hit_tiers.append("G2_attack")
    if g3.search(t):
        hit_tiers.append("G3_orthodoxy")
    if g4p.search(t) and g4a.search(t):
        hit_tiers.append("G4_game")
    if g5.search(t):
        hit_tiers.append("G5_mobilize")

    if not hit_tiers:
        return 0, ""
    # 精汉 单独可标
    tier = "+".join(sorted(set(hit_tiers)))
    if "精汉" in t:
        tier += "|jinghan_token"
    return 1, tier


def broad_pool_match(text: str) -> bool:
    t = text if isinstance(text, str) else str(text)
    if not t:
        return False
    for neg in NEGATION_EXCLUDE:
        if neg in t:
            return False
    broad_re = _compile_or(KW_BROAD_FOR_POOL)
    return bool(broad_re.search(t))


def annotate_keyword_hits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hits = []
    tiers = []
    broad = []
    for txt in out["content_clean"].astype(str):
        h, tier = keyword_score_row(txt)
        hits.append(h)
        tiers.append(tier)
        broad.append(1 if (h or broad_pool_match(txt)) else 0)
    out["keyword_hit"] = hits  # 精炼规则，用于与 API 对比
    out["keyword_tier"] = tiers
    out["nationalist_pool"] = broad  # 宽口径：疑似池成员
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="数据探索 + 样本A/B 抽样")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--outdir", type=Path, default=DATA_DIR)
    ap.add_argument("--explore-only", action="store_true")
    ap.add_argument("--full", action="store_true", help="探索并写出 sample_A / sample_B / log")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-copy-raw", action="store_true", help="不把输入复制为 data/raw_cleaned.csv")
    args = ap.parse_args()

    if not args.explore_only and not args.full:
        print("请指定 --explore-only 或 --full", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    log_buf = StringIO()

    if not args.input.exists():
        print(f"找不到输入文件: {args.input}", file=sys.stderr)
        sys.exit(1)

    _log_line(log_buf, f"输入: {args.input}")
    _log_line(log_buf, f"读取时间: {datetime.now().isoformat()}")
    df = load_cleaned(args.input)
    stats = run_exploration(df, log_buf)

    if args.explore_only:
        out = log_buf.getvalue()
        print(out)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_path = args.outdir / "raw_cleaned.csv"
    if not args.no_copy_raw and not raw_path.exists():
        shutil.copy2(args.input, raw_path)
        _log_line(log_buf, f"已复制 raw_cleaned.csv → {raw_path}")
    elif raw_path.exists():
        _log_line(log_buf, "raw_cleaned.csv 已存在，跳过复制")

    eligible = build_eligible_a(df, log_buf, label="样本A/B基础")
    eligible_kw = annotate_keyword_hits(eligible)
    kw_pool_strict = eligible_kw[eligible_kw["keyword_hit"] == 1]
    kw_pool = eligible_kw[eligible_kw["nationalist_pool"] == 1].copy()
    _log_line(
        log_buf,
        f"精炼关键词命中（一级+len>=5）: {len(kw_pool_strict):,} | "
        f"宽口径疑似池（用于抽 B）: {len(kw_pool):,}",
    )

    eligible["_segment"] = assign_segment(eligible["_dt"])
    in_window = eligible[eligible["_segment"] != "out"].copy()
    _log_line(
        log_buf,
        f"时间三段内 eligible: {len(in_window):,} (排除窗口外 {len(eligible) - len(in_window):,})",
    )

    for seg, tn in SEGMENT_TARGETS.items():
        sub = in_window[in_window["_segment"] == seg]
        _log_line(log_buf, f"  段 [{seg}] 池子大小: {len(sub):,} → 目标抽取: {tn:,}")

    parts_a: List[pd.DataFrame] = []
    for seg, target_n in SEGMENT_TARGETS.items():
        parts_a.append(
            stratified_sample_segment(in_window, seg, target_n, rng)
        )
    sample_a = pd.concat(parts_a, ignore_index=True)
    sample_a = sample_a.drop_duplicates(subset=["comment_id"], keep="first")
    _log_line(log_buf, f"\n合并去重后样本A: {len(sample_a):,}")

    use_cap = stats["top10_share"] > TOP10_CONC_THRESHOLD
    if use_cap:
        before = len(sample_a)
        sample_a = apply_video_cap(sample_a, VIDEO_CAP_IF_CONCENTRATED, rng)
        _log_line(
            log_buf,
            f"Top10 占比>{TOP10_CONC_THRESHOLD:.0%}，已按每视频上限 {VIDEO_CAP_IF_CONCENTRATED} 截断: {before:,} → {len(sample_a):,}",
        )
    else:
        _log_line(
            log_buf,
            f"Top10 占比 {stats['top10_share']:.1%} ≤ {TOP10_CONC_THRESHOLD:.0%}，未做视频上限截断",
        )

    TARGET_A = 30_000
    if len(sample_a) > TARGET_A:
        sample_a = sample_a.sample(n=TARGET_A, random_state=rng)
        _log_line(log_buf, f"随机裁至 {TARGET_A:,} 条")
    elif len(sample_a) < TARGET_A:
        sample_a = supplement_to_n(sample_a, in_window, TARGET_A, rng)
        _log_line(log_buf, f"补足后样本A: {len(sample_a):,}")

    drop_cols = ["_dt", "_like", "_segment"]
    sample_a_out = sample_a.drop(columns=[c for c in drop_cols if c in sample_a.columns])
    path_a = args.outdir / "sample_A_30k.csv"
    sample_a_out.to_csv(path_a, index=False, encoding="utf-8-sig")
    _log_line(log_buf, f"已写 {path_a}")

    # ── 样本 B：1200 自关键词池，800 自 A（去重）；不足则先补关键词池再补 A
    n_kw = min(B_FROM_KEYWORD, len(kw_pool))
    strict_sub = kw_pool[kw_pool["keyword_hit"] == 1]
    broad_sub = kw_pool[kw_pool["keyword_hit"] == 0]
    parts_kw: List[pd.DataFrame] = []
    n_strict = min(len(strict_sub), n_kw)
    if n_strict > 0:
        parts_kw.append(strict_sub.sample(n=n_strict, random_state=rng))
    rem = n_kw - n_strict
    if rem > 0 and len(broad_sub) > 0:
        parts_kw.append(broad_sub.sample(n=min(rem, len(broad_sub)), random_state=rng))
    part_kw = pd.concat(parts_kw, ignore_index=True) if parts_kw else kw_pool.iloc[:0]
    part_kw["keyword_pool_tier"] = np.where(
        part_kw["keyword_hit"] == 1, "strict", "broad_only"
    )
    kw_chosen = set(part_kw["comment_id"].astype(str))

    pool_a_only = sample_a_out.copy()
    pool_rand = pool_a_only[~pool_a_only["comment_id"].astype(str).isin(kw_chosen)]
    n_rand = min(B_FROM_A_RANDOM, len(pool_rand))
    part_rand = pool_rand.sample(n=n_rand, random_state=rng) if n_rand else pool_rand.iloc[:0]

    part_kw_b = part_kw.copy()
    part_rand_b = annotate_keyword_hits(part_rand.copy()) if len(part_rand) else part_rand
    if "keyword_pool_tier" not in part_rand_b.columns:
        part_rand_b["keyword_pool_tier"] = ""
    part_kw_b["sample_b_source"] = "keyword_pool"
    part_rand_b["sample_b_source"] = "random_from_A"

    sample_b = pd.concat([part_kw_b, part_rand_b], ignore_index=True)
    sample_b = sample_b.drop_duplicates(subset=["comment_id"], keep="first")
    shortfall = B_TOTAL - len(sample_b)
    if shortfall > 0:
        used = set(sample_b["comment_id"].astype(str))
        more_kw = kw_pool[~kw_pool["comment_id"].astype(str).isin(used)]
        more_a = pool_a_only[~pool_a_only["comment_id"].astype(str).isin(used)]
        fill_kw_n = min(shortfall, len(more_kw))
        extra_rows: List[pd.DataFrame] = []
        if fill_kw_n > 0:
            ek = more_kw.sample(n=fill_kw_n, random_state=rng).copy()
            ek["keyword_pool_tier"] = np.where(ek["keyword_hit"] == 1, "strict", "broad_only")
            ek["sample_b_source"] = "keyword_pool_supp"
            extra_rows.append(ek)
        rem = shortfall - fill_kw_n
        if rem > 0 and len(more_a) > 0:
            ea = more_a.sample(n=min(rem, len(more_a)), random_state=rng)
            ea = annotate_keyword_hits(ea.copy())
            ea["sample_b_source"] = "random_from_A_supp"
            extra_rows.append(ea)
        if extra_rows:
            sample_b = pd.concat([sample_b] + extra_rows, ignore_index=True).drop_duplicates(
                subset=["comment_id"], keep="first"
            )

    if len(sample_b) > B_TOTAL:
        sample_b = sample_b.sample(n=B_TOTAL, random_state=rng)

    path_b = args.outdir / "sample_B_2000.csv"
    sample_b.to_csv(path_b, index=False, encoding="utf-8-sig")
    _log_line(log_buf, f"已写 {path_b} (n={len(sample_b):,})")

    # B 的 like / 时间 分布摘要
    lk_b = pd.to_numeric(sample_b["like_count"], errors="coerce").fillna(0)
    _log_line(log_buf, "\n样本 B 诊断:")
    _log_line(
        log_buf,
        f"  keyword_hit(精炼)=1: {(sample_b['keyword_hit']==1).sum():,} | "
        f"keyword_pool_tier=broad_only: {(sample_b.get('keyword_pool_tier','')=='broad_only').sum():,}",
    )
    _log_line(log_buf, f"  like_count 中位数: {lk_b.median():.0f}")
    dt_b = parse_create_dt(sample_b["create_time"])
    _log_line(log_buf, f"  时间范围: {dt_b.min()} ~ {dt_b.max()}")

    log_path = args.outdir / "sampling_log.txt"
    log_path.write_text(log_buf.getvalue(), encoding="utf-8")
    print(log_buf.getvalue())
    print(f"\n日志已写入 {log_path}")


if __name__ == "__main__":
    main()
