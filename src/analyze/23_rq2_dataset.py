# -*- coding: utf-8 -*-
"""
23_rq2_dataset.py — Phase 2：RQ2 参与度建模的数据组装（回应 R3-3 的前置）

合并三层数据 → 一张建模表：
  评论层  data/sample_A_30k.csv（30k，全部一级评论）
  标签层  results/phase1/sampleA_probs_calibrated.npz（校准后 flat5 概率；
          argmax 为主标签，5 列概率保留给测量误差多重插补）
  视频层  01_data/search_videos_all.csv + search_creators_all.csv
          （播放量、UP 主粉丝数、视频发布时间；对 Sample A 覆盖率 100%）

清洗决策（记录在输出 JSON）：
  - like_count > 1e6 视为解析错误剔除（B 站单评论点赞不可能达 1e8+；
    诊断见 16_ 脚本注释），不做 winsorize——NB 模型直接建模计数；
  - 粉丝数缺失（<1% 视频）的行剔除；
  - days_since_video < 0（时钟噪声）截断为 0。

输出: results/phase2/rq2_dataset.csv.gz（R 建模用）
      results/phase2/rq2_dataset_meta.json（清洗记录）
"""
import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPT_DIR / "data"
P1_DIR = SCRIPT_DIR / "results" / "phase1"
P2_DIR = SCRIPT_DIR / "results" / "phase2"

LIKE_PARSE_ERROR_THRESHOLD = 1e6


def main():
    P2_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}

    sa = pd.read_csv(DATA_DIR / "sample_A_30k.csv", encoding="utf-8-sig")
    meta["n_raw"] = len(sa)

    # ── 标签层：校准后概率 ──
    npz = np.load(P1_DIR / "sampleA_probs_calibrated.npz", allow_pickle=True)
    probs = pd.DataFrame(npz["flat5_probs_calibrated"],
                         columns=[f"p_type{j}" for j in range(5)])
    probs["comment_id"] = npz["comment_id"]
    sa = sa.merge(probs, on="comment_id", how="inner", validate="1:1")
    sa["discourse_type"] = sa[[f"p_type{j}" for j in range(5)]].to_numpy().argmax(axis=1)
    meta["n_after_label_join"] = len(sa)

    # ── 结局变量清洗 ──
    sa["likes"] = pd.to_numeric(sa["like_count"], errors="coerce")
    bad = sa["likes"].isna() | (sa["likes"] > LIKE_PARSE_ERROR_THRESHOLD)
    meta["n_like_parse_errors_dropped"] = int(bad.sum())
    sa = sa[~bad].copy()
    sa["likes"] = sa["likes"].astype(int)

    # ── 评论层协变量 ──
    sa["text_len"] = sa["content_clean"].fillna("").astype(str).str.len()
    ct = pd.to_datetime(pd.to_numeric(sa["create_time"], errors="coerce"), unit="s")
    sa["post_hour"] = ct.dt.hour
    sa["is_weekend"] = ct.dt.dayofweek.isin([5, 6]).astype(int)

    # ── 视频层协变量 ──
    v = pd.read_csv(SCRIPT_DIR / "01_data" / "search_videos_all.csv",
                    usecols=["video_id", "user_id", "create_time", "video_play_count"])
    v = v.dropna(subset=["video_id"]).copy()
    v["video_id"] = v["video_id"].astype("int64")
    v = v.drop_duplicates("video_id")
    v = v.rename(columns={"create_time": "video_create_time", "user_id": "up_user_id"})

    c = pd.read_csv(SCRIPT_DIR / "01_data" / "search_creators_all.csv",
                    usecols=["user_id", "total_fans"])
    c = c.drop_duplicates("user_id").rename(columns={"user_id": "up_user_id"})
    v = v.merge(c, on="up_user_id", how="left")

    sa = sa.merge(v, on="video_id", how="left", validate="m:1")
    meta["n_missing_video_meta"] = int(sa["video_create_time"].isna().sum())
    meta["n_missing_fans"] = int(sa["total_fans"].isna().sum())
    sa = sa.dropna(subset=["video_create_time", "total_fans", "post_hour"]).copy()

    sa["days_since_video"] = (
        (pd.to_numeric(sa["create_time"]) - sa["video_create_time"]) / 86400
    ).clip(lower=0)
    sa["log_play"] = np.log1p(sa["video_play_count"])
    sa["log_fans"] = np.log1p(sa["total_fans"])
    sa["log_textlen"] = np.log1p(sa["text_len"])

    meta["n_final"] = len(sa)
    meta["n_videos"] = int(sa["video_id"].nunique())
    meta["comments_per_video_median"] = float(sa.groupby("video_id").size().median())
    meta["likes_zero_share"] = round(float((sa["likes"] == 0).mean()), 4)
    meta["likes_p50_p99"] = [float(sa["likes"].quantile(q)) for q in (0.5, 0.99)]
    meta["discourse_type_counts"] = sa["discourse_type"].value_counts().sort_index().to_dict()

    cols = (["comment_id", "video_id", "likes", "discourse_type",
             "log_textlen", "post_hour", "is_weekend", "days_since_video",
             "log_play", "log_fans"] + [f"p_type{j}" for j in range(5)])
    out = P2_DIR / "rq2_dataset.csv.gz"
    sa[cols].to_csv(out, index=False, compression="gzip")
    (P2_DIR / "rq2_dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[OK] 建模表 → {out}  ({len(sa)} 行 × {len(cols)} 列)")


if __name__ == "__main__":
    main()
