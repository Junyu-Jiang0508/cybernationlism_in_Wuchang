# -*- coding: utf-8 -*-
"""
15_rq1_analysis.py
RQ1: 民族主义话语是否表现出语义同质化与情感饱和？

分析内容：
  1. 聚类 x 话语类型交叉分析（NMI / ARI / Homogeneity）
  2. 各话语类型内部余弦相似度（同类 vs 跨类）
  3. 高频 n-gram 模板检测（nationalist vs non-nationalist）

运行：
  python 15_rq1_analysis.py

输出：
  results/rq1_cluster_cross.csv
  results/rq1_cosine_similarity.csv
  results/rq1_ngram_top50.csv
  results/rq1_summary.txt
"""
from __future__ import annotations
import os, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
EMBED_DIR = BASE / "embeddings"
RESULTS.mkdir(exist_ok=True)

TYPE_NAMES = {
    0: "Game Review",
    1: "Emotional",
    2: "Politicized",
    3: "Nationalist",
    4: "Neutral/Analytic",
}

# ── 加载数据 ───────────────────────────────────────────────
print("=== 加载数据 ===")
df = pd.read_csv(RESULTS / "finetune/sample_A_predicted.csv", encoding="utf-8-sig")
print(f"  预测数据: {len(df)} 条")

embeddings_minilm = np.load(EMBED_DIR / "embeddings_minilm.npy")
comment_ids_emb   = np.load(EMBED_DIR / "comment_ids.npy", allow_pickle=True)
labels_cluster    = np.load(RESULTS / "labels_minilm_best.npy")
print(f"  MiniLM embeddings: {embeddings_minilm.shape}")
print(f"  聚类标签: {len(labels_cluster)} 条, {len(np.unique(labels_cluster))} 个聚类")

# 对齐 embedding 与预测（按 comment_id 匹配）
df["comment_id"] = df["comment_id"].astype(str)
id_to_pred = dict(zip(df["comment_id"].astype(str), df["pred_discourse_type"]))
comment_ids_emb_str = [str(x) for x in comment_ids_emb]

aligned_preds = np.array([id_to_pred.get(cid, -1) for cid in comment_ids_emb_str])
valid_mask = aligned_preds >= 0
print(f"  有效对齐: {valid_mask.sum()} / {len(valid_mask)} 条")

emb_valid     = embeddings_minilm[valid_mask]
cluster_valid = labels_cluster[valid_mask]
pred_valid    = aligned_preds[valid_mask]


# ── 分析 1：聚类 × 话语类型交叉 ───────────────────────────
print("\n=== 分析1：聚类 x 话语类型交叉指标 ===")

nmi  = normalized_mutual_info_score(cluster_valid, pred_valid)
ari  = adjusted_rand_score(cluster_valid, pred_valid)
homo = homogeneity_score(cluster_valid, pred_valid)
comp = completeness_score(cluster_valid, pred_valid)
vme  = v_measure_score(cluster_valid, pred_valid)

print(f"  NMI  = {nmi:.4f}  (聚类与话语类型的互信息，0=独立，1=完美)")
print(f"  ARI  = {ari:.4f}  (调整兰德指数，0=随机，1=完美)")
print(f"  Homogeneity = {homo:.4f}  (每个聚类是否只含一种话语类型)")
print(f"  Completeness= {comp:.4f}  (每种话语类型是否集中在少数聚类)")
print(f"  V-measure   = {vme:.4f}  (H和C的调和平均)")

# 各聚类内话语类型分布
cluster_rows = []
for c in sorted(np.unique(cluster_valid)):
    mask_c = cluster_valid == c
    sub_preds = pred_valid[mask_c]
    n = mask_c.sum()
    dominant = int(pd.Series(sub_preds).mode()[0])
    purity = (sub_preds == dominant).mean()
    dist = {f"type{t}": int((sub_preds == t).sum()) for t in range(5)}
    cluster_rows.append({"cluster": int(c), "n": int(n),
                          "dominant_type": dominant, "purity": round(purity, 4), **dist})

df_cluster = pd.DataFrame(cluster_rows).sort_values("purity", ascending=False)
df_cluster.to_csv(RESULTS / "rq1_cluster_cross.csv", index=False, encoding="utf-8-sig")
print(f"\n  Top 5 最纯聚类:")
for _, row in df_cluster.head(5).iterrows():
    print(f"    Cluster {int(row['cluster'])}: n={int(row['n'])}, "
          f"dominant=type{int(row['dominant_type'])}({TYPE_NAMES[int(row['dominant_type'])]}), "
          f"purity={row['purity']:.3f}")


# ── 分析 2：各话语类型内部余弦相似度 ──────────────────────
print("\n=== 分析2：余弦相似度（类内 vs 跨类）===")
np.random.seed(42)
N_SAMPLE = 800  # 每次随机抽 N 条计算

sim_rows = []
for dt in range(5):
    mask = pred_valid == dt
    n = mask.sum()
    if n < 20:
        continue
    emb_sub = emb_valid[mask]
    idx = np.random.choice(len(emb_sub), min(N_SAMPLE, len(emb_sub)), replace=False)
    sim_mat = cosine_similarity(emb_sub[idx])
    upper = sim_mat[np.triu_indices(len(idx), k=1)]
    avg_sim = float(upper.mean())
    std_sim  = float(upper.std())
    sim_rows.append({
        "discourse_type": dt,
        "name": TYPE_NAMES[dt],
        "n": int(n),
        "intra_cosine_mean": round(avg_sim, 4),
        "intra_cosine_std":  round(std_sim, 4),
    })
    print(f"  type{dt} ({TYPE_NAMES[dt]}): n={n}, intra-sim={avg_sim:.4f} ± {std_sim:.4f}")

# 跨类平均（全体随机 1600 对）
idx_all = np.random.choice(len(emb_valid), min(1600, len(emb_valid)), replace=False)
sim_all = cosine_similarity(emb_valid[idx_all])
cross_sim = float(sim_all[np.triu_indices(len(idx_all), k=1)].mean())
print(f"\n  全体（跨类）平均余弦相似度: {cross_sim:.4f}")

df_sim = pd.DataFrame(sim_rows)
df_sim["cross_class_mean"] = round(cross_sim, 4)
df_sim["intra_minus_cross"] = (df_sim["intra_cosine_mean"] - cross_sim).round(4)
df_sim.to_csv(RESULTS / "rq1_cosine_similarity.csv", index=False, encoding="utf-8-sig")
print("\n  结论（intra > cross 说明该类语义更聚集）:")
for _, row in df_sim.sort_values("intra_minus_cross", ascending=False).iterrows():
    sign = "聚集" if row["intra_minus_cross"] > 0 else "分散"
    print(f"    type{int(row['discourse_type'])} {row['name']}: "
          f"delta={row['intra_minus_cross']:+.4f} ({sign})")


# ── 分析 3：N-gram 模板检测 ────────────────────────────────
print("\n=== 分析3：N-gram 模板检测 ===")

df_pred = df[df["pred_discourse_type"].isin(range(5))].copy()
nationalist   = df_pred[df_pred["pred_discourse_type"].isin([2, 3])]["content_clean"].dropna().astype(str)
non_national  = df_pred[df_pred["pred_discourse_type"].isin([0, 1, 4])]["content_clean"].dropna().astype(str)

def extract_ngrams(texts: pd.Series, n: int = 3) -> Counter:
    c: Counter = Counter()
    for t in texts:
        chars = list(t.strip())
        for i in range(len(chars) - n + 1):
            c["".join(chars[i:i+n])] += 1
    return c

tri_nat  = extract_ngrams(nationalist, 3)
tri_non  = extract_ngrams(non_national, 3)

# 民族主义特征词（在 nationalist 中频率显著高于 non-nationalist）
nat_total = nationalist.str.len().sum()
non_total = non_national.str.len().sum()

rows_ng = []
for gram, cnt in tri_nat.most_common(200):
    freq_nat = cnt / max(nat_total, 1)
    freq_non = tri_non.get(gram, 0) / max(non_total, 1)
    ratio = freq_nat / max(freq_non, 1e-9)
    rows_ng.append({"ngram": gram, "cnt_nationalist": cnt,
                    "freq_nat": round(freq_nat * 1e5, 2),
                    "freq_non": round(freq_non * 1e5, 2),
                    "ratio": round(ratio, 2)})

df_ng = pd.DataFrame(rows_ng).sort_values("ratio", ascending=False)
df_ng.to_csv(RESULTS / "rq1_ngram_top50.csv", index=False, encoding="utf-8-sig")

print(f"  民族主义评论数: {len(nationalist)}")
print(f"  非民族主义评论数: {len(non_national)}")
print(f"\n  Top 20 民族主义特征三字词（ratio = 在nationalist中频率/non-nationalist）:")
for _, row in df_ng.head(20).iterrows():
    print(f"    '{row['ngram']}': 出现{int(row['cnt_nationalist'])}次, ratio={row['ratio']:.1f}x")

# 重叠率分析
nat_set = set(g for g, _ in tri_nat.most_common(500))
non_set = set(g for g, _ in tri_non.most_common(500))
overlap = len(nat_set & non_set) / len(nat_set | non_set)
print(f"\n  Top-500 三字词 Jaccard 重叠率: {overlap:.3f}")
print(f"  (越低说明两类评论词汇越不同，支持'民族主义话语模板化'假设)")


# ── 汇总输出 ──────────────────────────────────────────────
summary = f"""
=== RQ1 分析汇总 ===

[聚类 x 话语类型 交叉指标]
NMI         = {nmi:.4f}
ARI         = {ari:.4f}
Homogeneity = {homo:.4f}
Completeness= {comp:.4f}
V-measure   = {vme:.4f}

[各类型内部余弦相似度]
{df_sim[['discourse_type','name','n','intra_cosine_mean','intra_minus_cross']].to_string(index=False)}
全体跨类平均: {cross_sim:.4f}

[N-gram 模板检测]
民族主义评论数: {len(nationalist)}
非民族主义评论数: {len(non_national)}
Top-500 三字词 Jaccard 重叠率: {overlap:.3f}
Top10 特征三字词:
{df_ng.head(10)[['ngram','cnt_nationalist','ratio']].to_string(index=False)}
"""
print(summary)
(RESULTS / "rq1_summary.txt").write_text(summary, encoding="utf-8")
print("[OK] RQ1 分析完成，结果已保存到 results/")
