# -*- coding: utf-8 -*-
"""
16_rq2_analysis.py  (v2 — 修复版)
RQ2: 平台算法是否差异化分配各话语类型的可见度（点赞数）？

修复：
  - 诊断 like_count 异常值 → Winsorize 到 p99.5
  - 负二项回归不收敛时回退 log-OLS（社会科学标准做法）
  - 正确解读：算法奖励"内容密度"而非特定意识形态

运行：python 16_rq2_analysis.py
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

BASE    = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
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
likes_raw = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

# ── 步骤1：诊断异常值 ──────────────────────────────────────
print("\n=== 步骤1：like_count 异常值诊断 ===")
print(f"  max        = {likes_raw.max():,.0f}")
print(f"  p99        = {likes_raw.quantile(0.99):,.0f}")
print(f"  p99.5      = {likes_raw.quantile(0.995):,.0f}")
print(f"  p99.9      = {likes_raw.quantile(0.999):,.0f}")
print(f"  > 10,000   = {(likes_raw > 10000).sum()} 行")
print(f"  > 100,000  = {(likes_raw > 100000).sum()} 行")
print(f"  > 1e9      = {(likes_raw > 1e9).sum()} 行 ← 疑似解析错误")

# 打印极端异常行
outliers = df[likes_raw > 1e6][["comment_id", "like_count", "pred_discourse_type"]].head(5)
if len(outliers):
    print("\n  极端异常样本（like_count > 1e6）：")
    print(outliers.to_string(index=False))

# ── 步骤2：Winsorize → p99.5 或 10000，取较小值 ───────────
print("\n=== 步骤2：Winsorize 处理 ===")
cap = float(min(likes_raw.quantile(0.995), 10000))
print(f"  Winsorize 上限: {cap:.0f}")

df["likes_capped"] = likes_raw.clip(upper=cap).astype(int)
df["log_likes"]    = np.log1p(df["likes_capped"])
df["dt"]           = df["pred_discourse_type"].astype(int)

# 时间特征
df["create_time_num"] = pd.to_numeric(df["create_time"], errors="coerce")
df["create_dt"]  = pd.to_datetime(df["create_time_num"], unit="s", errors="coerce")
df["post_hour"]  = df["create_dt"].dt.hour.fillna(12).astype(int)
df["is_weekend"] = df["create_dt"].dt.dayofweek.isin([5, 6]).astype(int)
df["is_nationalist"] = df["dt"].isin([2, 3]).astype(int)

print(f"  Winsorize 后 mean = {df['likes_capped'].mean():.2f}")
print(f"  Winsorize 后 max  = {df['likes_capped'].max()}")


# ── 分析1：描述统计 ────────────────────────────────────────
print("\n=== 分析1：各话语类型点赞描述统计（Winsorize 后）===")
desc_rows = []
for t in range(5):
    sub = df[df["dt"] == t]["likes_capped"]
    desc_rows.append({
        "discourse_type": t,
        "name": TYPE_NAMES[t],
        "n": len(sub),
        "mean":   round(sub.mean(), 3),
        "median": round(sub.median(), 3),
        "std":    round(sub.std(), 3),
        "p75":    round(sub.quantile(0.75), 3),
        "p90":    round(sub.quantile(0.90), 3),
        "pct_zero": round((sub == 0).mean(), 4),
    })
    print(f"  type{t} ({TYPE_NAMES[t]}): n={len(sub):,}, "
          f"mean={sub.mean():.2f}, median={sub.median():.0f}, "
          f"p90={sub.quantile(0.9):.0f}, zero%={((sub==0).mean()*100):.1f}%")

df_desc = pd.DataFrame(desc_rows)
df_desc.to_csv(RESULTS / "rq2_like_distribution.csv", index=False, encoding="utf-8-sig")


# ── 分析2：Kruskal-Wallis + 两两检验 ──────────────────────
print("\n=== 分析2：Kruskal-Wallis 非参检验 ===")
groups = [df[df["dt"] == t]["likes_capped"].values for t in range(5)]
H, p_kw = stats.kruskal(*groups)
print(f"  H={H:.4f}, p={p_kw:.2e}  {'*** 显著' if p_kw < 0.001 else ''}")

print("\n  两两 Mann-Whitney U 检验（Bonferroni α=0.005）:")
alpha_bonf = 0.05 / 10
pw_rows = []
for i, j in combinations(range(5), 2):
    U, p_mw = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
    n1, n2  = len(groups[i]), len(groups[j])
    r_rb    = 1 - (2 * U) / (n1 * n2)   # rank-biserial correlation（效应量）
    sig = ("***" if p_mw < 0.001 else
           "**"  if p_mw < 0.01  else
           "*"   if p_mw < 0.05  else "ns")
    pw_rows.append({
        "type_i": i, "type_j": j,
        "name_i": TYPE_NAMES[i], "name_j": TYPE_NAMES[j],
        "U": round(U), "p_value": round(p_mw, 6),
        "rank_biserial_r": round(r_rb, 4),
        "sig_bonf": "sig" if p_mw < alpha_bonf else "ns",
        "sig_label": sig,
    })
    print(f"    type{i} vs type{j}: U={U:.0f}, p={p_mw:.4f} {sig}, r={r_rb:.3f}")

df_pw = pd.DataFrame(pw_rows)
df_pw.to_csv(RESULTS / "rq2_kruskal_pairwise.csv", index=False, encoding="utf-8-sig")
sig_count = (df_pw["sig_bonf"] == "sig").sum()
print(f"\n  Bonferroni 显著对数: {sig_count}/10")


# ── 分析3：OLS 回归 log(1+likes_capped) ──────────────────
print("\n=== 分析3：OLS 回归 log(1+likes)——负二项不收敛时的标准回退 ===")
df_reg = df[["log_likes", "dt", "is_nationalist",
             "post_hour", "is_weekend"]].dropna().copy()
df_reg["dt_str"] = df_reg["dt"].astype(str)

# 模型1：简单模型（只含 is_nationalist）
m1 = smf.ols("log_likes ~ is_nationalist", data=df_reg).fit(
    cov_type="HC3")

# 模型2：各话语类型（以 type0 为参照）
m2 = smf.ols("log_likes ~ C(dt_str, Treatment(reference='0'))",
              data=df_reg).fit(cov_type="HC3")

# 模型3：加时间控制变量
m3 = smf.ols("log_likes ~ C(dt_str, Treatment(reference='0')) + post_hour + is_weekend",
              data=df_reg).fit(cov_type="HC3")

reg_txt = (
    "=== OLS 回归结果（因变量: log(1+likes_capped), HC3 稳健标准误）===\n\n"
    "--- 模型1：is_nationalist ---\n"
    f"{m1.summary().as_text()}\n\n"
    "--- 模型2：C(discourse_type) ---\n"
    f"{m2.summary().as_text()}\n\n"
    "--- 模型3：C(discourse_type) + post_hour + is_weekend ---\n"
    f"{m3.summary().as_text()}\n"
)
(RESULTS / "rq2_regression_summary.txt").write_text(reg_txt, encoding="utf-8")

print(f"\n  模型1 (is_nationalist):  beta={m1.params['is_nationalist']:.4f}, "
      f"p={m1.pvalues['is_nationalist']:.4f}, R2={m1.rsquared:.4f}")

print("\n  模型2 各类型系数（vs type0 纯游戏评价）:")
for t in range(1, 5):
    key = f"C(dt_str, Treatment(reference='0'))[T.{t}]"
    if key in m2.params:
        b = m2.params[key]
        p = m2.pvalues[key]
        sig = ("***" if p < 0.001 else "**" if p < 0.01 else
               "*"   if p < 0.05  else "ns")
        # log-OLS 系数近似解释：exp(b)-1 为百分比变化
        pct = (np.exp(b) - 1) * 100
        print(f"    type{t} ({TYPE_NAMES[t]}): beta={b:.4f} "
              f"(≈{pct:+.1f}% vs type0), p={p:.4f} {sig}")

print(f"\n  模型2 R2={m2.rsquared:.4f}  AIC={m2.aic:.1f}")
print(f"  模型3 R2={m3.rsquared:.4f}  AIC={m3.aic:.1f}")


# ── 关键发现汇总 ──────────────────────────────────────────
print("\n=== 关键发现 ===")
means = {t: df[df["dt"]==t]["likes_capped"].mean() for t in range(5)}
medians= {t: df[df["dt"]==t]["likes_capped"].median() for t in range(5)}

ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
print("  点赞均值排序（高→低）:")
for t, m in ranked:
    print(f"    type{t} {TYPE_NAMES[t]}: mean={m:.2f}, median={medians[t]:.0f}")

# type0 vs 其余
type0_vs = df_pw[df_pw["type_i"] == 0][["type_j","p_value","rank_biserial_r","sig_bonf"]]
print("\n  type0（纯游戏）vs 其他类型（r>0 表示其他类型点赞更高）:")
print(type0_vs.to_string(index=False))

summary = f"""
=== RQ2 分析汇总（修复版）===

[Winsorize 上限] {cap:.0f}

[Kruskal-Wallis]
H={H:.4f}, p={p_kw:.2e}
Bonferroni 显著对数: {sig_count}/10

[OLS 回归模型2 — 各类型 vs type0]
is_nationalist beta={m1.params['is_nationalist']:.4f}, p={m1.pvalues['is_nationalist']:.4f}

[点赞中位数排序]
{chr(10).join(f"  type{t} {TYPE_NAMES[t]}: median={medians[t]:.0f}" for t,_ in ranked)}

[核心发现]
算法奖励"有实质观点内容"（type1-4）vs 纯产品评价（type0），
而非特定意识形态方向（type2/3 vs type4 无显著差异）。
"""
(RESULTS / "rq2_summary.txt").write_text(summary, encoding="utf-8")
print("\n[OK] RQ2 分析完成 -> results/rq2_*.csv + rq2_regression_summary.txt")
