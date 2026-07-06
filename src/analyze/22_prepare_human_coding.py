# -*- coding: utf-8 -*-
"""
22_prepare_human_coding.py — Phase 1：人工编码材料准备（计划 §三.3 / §三.4）

输入: results/phase1/oof_predictions.parquet, results/phase1/sampleA_probs.npz,
      data/sample_A_30k.csv, data/annotations_merged.csv
输出: results/phase1/human_coding/
      error_analysis_coder{1,2}.csv   Type1↔2 双向错分各≤50 条，双人独立归因
      error_analysis_key.csv          含模型概率与折号的对照表（编码后合并用）
      boundary_batch_coder{1,2}.csv   定向双编码批次（1/2 边界 100 + 2/3 边界 50）
      boundary_batch_key.csv          选样依据（模型概率），编码期间勿给编码者
      CODING_INSTRUCTIONS.md          两项任务的编码说明

设计：
  - 错误归因需要看到 true/pred 标签（任务即解释分歧），三分归因：
    A=标注噪声（金标签本身可疑）B=真实语义模糊（构成性歧义）C=模型容量不足；
  - 定向标注批次对编码者盲(不含模型信息)，避免锚定；选样 = Sample A 中
    flat5 折平均概率的 1/2 与 2/3 最小间隔样本，排除已标注过的 comment_id。
"""
import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPT_DIR / "data"
P1_DIR = SCRIPT_DIR / "results" / "phase1"
OUT_DIR = P1_DIR / "human_coding"

RNG = np.random.default_rng(42)
sys.path.insert(0, str(Path(__file__).resolve().parent))


def softmax_np(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def make_error_analysis(df):
    p_flat = softmax_np(df[[f"flat5_logit{j}" for j in range(5)]].to_numpy())
    pred = p_flat.argmax(axis=1)
    y = df["discourse_type"].to_numpy()

    picks = []
    for true_c, pred_c in [(1, 2), (2, 1)]:
        m = np.flatnonzero((y == true_c) & (pred == pred_c))
        if len(m) > 50:
            m = RNG.choice(m, 50, replace=False)
        picks.append(pd.DataFrame({
            "case_id": [f"E{true_c}{pred_c}_{i:03d}" for i in range(len(m))],
            "content": df.iloc[m]["content_clean"].values,
            "gold_label": true_c,
            "model_pred": pred_c,
            "model_confidence": p_flat[m].max(axis=1).round(3),
            "comment_id": df.iloc[m]["comment_id"].values,
            "fold": df.iloc[m]["fold"].values,
        }))
        print(f"  true={true_c} → pred={pred_c}: 抽取 {len(m)} 条")

    full = pd.concat(picks, ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱两方向的顺序

    key_cols = ["case_id", "comment_id", "fold", "content", "gold_label",
                "model_pred", "model_confidence"]
    full[key_cols].to_csv(OUT_DIR / "error_analysis_key.csv", index=False, encoding="utf-8-sig")

    coder = full[["case_id", "content", "gold_label", "model_pred"]].copy()
    coder["attribution"] = ""   # A=标注噪声 B=语义模糊 C=模型不足
    coder["notes"] = ""
    for i in (1, 2):
        coder.to_csv(OUT_DIR / f"error_analysis_coder{i}.csv", index=False, encoding="utf-8-sig")
    return len(full)


def make_boundary_batch():
    sa_probs = np.load(P1_DIR / "sampleA_probs.npz", allow_pickle=True)
    p = sa_probs["flat5_probs_mean"]
    ids = sa_probs["comment_id"]

    sample_a = pd.read_csv(DATA_DIR / "sample_A_30k.csv", encoding="utf-8-sig")
    sample_a = sample_a.set_index("comment_id").loc[ids].reset_index()

    annotated = set(pd.read_csv(DATA_DIR / "annotations_merged.csv",
                                encoding="utf-8-sig")["comment_id"])
    fresh = ~sample_a["comment_id"].isin(annotated).to_numpy()

    batches = []
    for (c1, c2), n_want, tag in [((1, 2), 100, "b12"), ((2, 3), 50, "b23")]:
        mass = p[:, c1] + p[:, c2]
        margin = np.abs(p[:, c1] - p[:, c2])
        cand = np.flatnonzero(fresh & (mass >= 0.6))
        cand = cand[np.argsort(margin[cand])][:n_want]
        batches.append(pd.DataFrame({
            "batch_id": [f"{tag}_{i:03d}" for i in range(len(cand))],
            "comment_id": sample_a.iloc[cand]["comment_id"].values,
            "content": sample_a.iloc[cand]["content_clean"].fillna("").values,
            "stratum": f"type{c1}/type{c2} boundary",
            f"prob_type{c1}": p[cand, c1].round(3),
            f"prob_type{c2}": p[cand, c2].round(3),
        }))
        print(f"  type{c1}/{c2} 边界: 候选池达标后取 {len(cand)} 条")

    full = pd.concat(batches, ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)
    full.to_csv(OUT_DIR / "boundary_batch_key.csv", index=False, encoding="utf-8-sig")

    coder = full[["batch_id", "content"]].copy()
    coder["discourse_type"] = ""      # 0–4，同原编码本；可标 99=不可用
    coder["othering_intensity"] = ""  # 0–3
    coder["affect_intensity"] = ""    # 0–3
    coder["notes"] = ""
    for i in (1, 2):
        coder.to_csv(OUT_DIR / f"boundary_batch_coder{i}.csv", index=False, encoding="utf-8-sig")
    return len(full)


INSTRUCTIONS = """# Phase 1 人工编码说明（2026-07）

## 任务一：边界错误归因（error_analysis_coder1/2.csv）

对象：模型在 5 折 out-of-fold 预测中 Type1（情绪化非政治）↔ Type2（政治化批评）
双向错分的样本。两位编码者**独立**填写，不讨论、不查看对方文件。

`attribution` 列三选一（必要时可复选，用 "+" 连接，如 "A+B"）：
- **A 标注噪声**：金标签本身可疑——按编码本此条应标成模型预测的类（或第三类）；
- **B 真实语义模糊**：构成性歧义——评论本身处于情绪/政治的叠加态，
  两个标签都有文本依据（这是理论上有意义的类别，不要回避）；
- **C 模型容量不足**：金标签清楚无误，模型错得没有道理。

`notes`：一句话依据（引用评论中的关键短语）。

完成后交回，用 error_analysis_key.csv 合并计算归因一致率。

## 任务二：定向边界标注（boundary_batch_coder1/2.csv）

对象：Sample A 中模型最难区分的 type1/2 边界 100 条 + type2/3 边界 50 条
（文件内已打乱，编码者不知道每条来自哪个边界，也看不到模型概率——请勿参考 key 文件）。

按原编码本独立标注 `discourse_type`（0–4；不可用标 99）、
`othering_intensity`（0–3）、`affect_intensity`（0–3）。

目的：(1) 边界层单独报告 κ（回应 LLM 标注标准讨论）；
(2) 为 Stage B / 层级分类器补充最难样本的金标签。

预计工作量：每人约 250 条 × ~20 秒 ≈ 1.5 小时。
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(P1_DIR / "oof_predictions.parquet")
    print("错误归因样本：")
    n_err = make_error_analysis(df)
    print("定向边界批次：")
    n_bnd = make_boundary_batch()
    (OUT_DIR / "CODING_INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    print(f"\n[OK] {OUT_DIR}: 错误归因 {n_err} 条 × 2 coder；边界批次 {n_bnd} 条 × 2 coder")
    print("[OK] 编码说明 → CODING_INSTRUCTIONS.md")


if __name__ == "__main__":
    main()
