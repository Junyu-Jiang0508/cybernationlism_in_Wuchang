# -*- coding: utf-8 -*-
"""
18_visualizations.py
生成论文所需全部图表（8张）

运行：python 18_visualizations.py

输出（results/figures/）：
  fig1_hyperparam_heatmap.png   超参热力图（LR × Epoch，3模型）
  fig2_model_boxplot.png        3模型 5-fold F1 对比箱线图
  fig3_confusion_matrix.png     最佳模型混淆矩阵
  fig4_maxlen_ablation.png      MAX_LEN 消融（128 vs 256 per-class F1）
  fig5_cluster_discourse.png    聚类 × 话语类型堆叠柱状图
  fig6_likes_boxplot.png        各话语类型 log-likes 箱线图
  fig7_tsne.png                 t-SNE 嵌入散点图（按话语类型着色）
  fig8_timeline.png             各类型占比随时间变化（月度折线图）
"""
from __future__ import annotations
import json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

BASE    = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
FIG_DIR = RESULTS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 全局字体（支持中文）
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
})

TYPE_NAMES  = ["Game\nReview", "Emotional", "Politicized", "Nationalist", "Neutral/\nAnalytic"]
TYPE_COLORS = ["#4CAF50", "#FF9800", "#E53935", "#9C27B0", "#607D8B"]
EMBED_DIR   = BASE / "embeddings"

print("=== 开始生成图表 ===\n")


# ─────────────────────────────────────────────────────────
# Fig 1：超参热力图（LR × Epoch × 3模型）
# ─────────────────────────────────────────────────────────
def fig1_hyperparam_heatmap():
    path = RESULTS / "finetune/sweep_round1.json"
    if not path.exists():
        print("  [SKIP] sweep_round1.json 不存在")
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    lrs  = sorted({r["lr"]    for r in results})
    eps  = sorted({r["epochs"] for r in results})
    models = ["bert", "roberta", "macbert"]
    model_labels = ["BERT-base-Chinese", "RoBERTa-wwm-ext", "MacBERT-base"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, model, label in zip(axes, models, model_labels):
        mat = np.full((len(lrs), len(eps)), np.nan)
        for r in results:
            if r["model"] == model:
                i = lrs.index(r["lr"])
                j = eps.index(r["epochs"])
                mat[i, j] = r["best_val_f1_macro"]
        sns.heatmap(mat, annot=True, fmt=".3f", ax=ax,
                    xticklabels=eps,
                    yticklabels=[f"{lr:.0e}" for lr in lrs],
                    cmap="YlOrRd", vmin=0.50, vmax=0.65,
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Learning Rate")
    fig.suptitle("Macro F1 by Learning Rate × Epochs (Fold 0)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_hyperparam_heatmap.png")
    plt.close()
    print("  [OK] fig1_hyperparam_heatmap.png")


# ─────────────────────────────────────────────────────────
# Fig 2：3模型 5-fold F1 对比箱线图
# ─────────────────────────────────────────────────────────
def fig2_model_boxplot():
    data = {}
    for model in ["bert", "roberta", "macbert"]:
        p = RESULTS / f"finetune/sweep_round2_{model}_maxlen128.json"
        if not p.exists():
            p = RESULTS / f"finetune/sweep_round2_{model}.json"
        if p.exists():
            res = json.loads(p.read_text(encoding="utf-8"))
            data[model] = [r["best_val_f1_macro"] for r in res]

    if not data:
        print("  [SKIP] 无 sweep_round2 结果文件")
        return

    # 也加入最佳配置 maxlen=256
    p256 = RESULTS / "finetune/sweep_round2_roberta_maxlen256.json"
    if p256.exists():
        res256 = json.loads(p256.read_text(encoding="utf-8"))
        data["roberta\n(256)"] = [r["best_val_f1_macro"] for r in res256]

    labels = list(data.keys())
    vals   = list(data.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(vals, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#5C6BC0", "#EF5350", "#66BB6A", "#AB47BC"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 标注各点
    for i, v in enumerate(vals, 1):
        ax.scatter([i]*len(v), v, color="black", s=20, zorder=5, alpha=0.6)
        ax.text(i, np.mean(v) + 0.003, f"{np.mean(v):.3f}", ha="center",
                fontsize=8, color="black")

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Macro F1 (5-fold CV)")
    ax.set_title("Model Comparison: 5-Fold Cross-Validation Macro F1", fontsize=11)
    ax.set_ylim(0.48, 0.72)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_model_boxplot.png")
    plt.close()
    print("  [OK] fig2_model_boxplot.png")


# ─────────────────────────────────────────────────────────
# Fig 3：混淆矩阵
# ─────────────────────────────────────────────────────────
def fig3_confusion_matrix():
    # 从 smoke_result 或 final 的 training_result.json 读取
    candidates = [
        RESULTS / "finetune/best_model/training_result.json",
        RESULTS / "finetune/smoke_result.json",
    ]
    res = None
    for p in candidates:
        if p.exists():
            res = json.loads(p.read_text(encoding="utf-8"))
            break
    if res is None:
        print("  [SKIP] 找不到训练结果 JSON")
        return

    cm = np.array(res["confusion_matrix"])
    labels = ["Type0\nGame", "Type1\nEmotional",
              "Type2\nPoliticized", "Type3\nNationalist", "Type4\nNeutral"]

    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", ax=ax,
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", vmin=0, vmax=1,
                linewidths=0.5)
    # 在格子上叠加原始计数
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j+0.5, i+0.72, f"({cm[i,j]})",
                    ha="center", va="center", fontsize=7, color="gray")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    f1 = res.get("best_val_f1_macro", 0)
    ax.set_title(f"Confusion Matrix — Best Model (Macro F1={f1:.4f})", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_confusion_matrix.png")
    plt.close()
    print("  [OK] fig3_confusion_matrix.png")


# ─────────────────────────────────────────────────────────
# Fig 4：MAX_LEN 消融（per-class F1）
# ─────────────────────────────────────────────────────────
def fig4_maxlen_ablation():
    # 从 JSON 结果提取各类型平均 per-class F1
    maxlen_data = {}
    for ml in [128, 256]:
        p = RESULTS / f"finetune/sweep_round2_roberta_maxlen{ml}.json"
        if not p.exists():
            # 128 可能用旧文件名
            if ml == 128:
                p = RESULTS / "finetune/sweep_round2_roberta.json"
            if not p.exists():
                continue
        res = json.loads(p.read_text(encoding="utf-8"))
        # 取5折平均 per-class F1
        per_class = {}
        key_names = ["type0_game", "type1_emotional", "type2_political",
                     "type3_nationalist", "type4_neutral"]
        for k in key_names:
            vals = [r["per_class_f1"].get(k, 0) for r in res if "per_class_f1" in r]
            per_class[k] = np.mean(vals) if vals else 0
        maxlen_data[ml] = per_class

    if len(maxlen_data) < 2:
        print("  [SKIP] 缺少 MAX_LEN 比较数据")
        return

    x = np.arange(5)
    w = 0.35
    labels_abbr = ["Type0\nGame", "Type1\nEmotional",
                   "Type2\nPoliticized", "Type3\nNationalist", "Type4\nNeutral"]
    key_names = ["type0_game", "type1_emotional", "type2_political",
                 "type3_nationalist", "type4_neutral"]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_ml = ["#90CAF9", "#EF9A9A"]
    for idx, (ml, color) in enumerate(zip([128, 256], colors_ml)):
        if ml not in maxlen_data:
            continue
        vals = [maxlen_data[ml][k] for k in key_names]
        bars = ax.bar(x + idx*w - w/2, vals, w, label=f"MAX_LEN={ml}",
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_abbr, fontsize=9)
    ax.set_ylabel("Mean Per-class F1 (5-fold avg)")
    ax.set_ylim(0, 0.85)
    ax.set_title("MAX_LEN Ablation: Per-class F1 (RoBERTa, lr=3e-5, ep=3)", fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_maxlen_ablation.png")
    plt.close()
    print("  [OK] fig4_maxlen_ablation.png")


# ─────────────────────────────────────────────────────────
# Fig 5：聚类 × 话语类型堆叠柱状图
# ─────────────────────────────────────────────────────────
def fig5_cluster_discourse():
    p = RESULTS / "rq1_cluster_cross.csv"
    if not p.exists():
        print("  [SKIP] rq1_cluster_cross.csv 不存在，先运行 15_rq1_analysis.py")
        return
    df_c = pd.read_csv(p)
    type_cols = [f"type{t}" for t in range(5)]
    df_c["total"] = df_c[type_cols].sum(axis=1)
    for col in type_cols:
        df_c[col+"_pct"] = df_c[col] / df_c["total"]

    df_c = df_c.sort_values("cluster")
    x = np.arange(len(df_c))
    bottom = np.zeros(len(df_c))

    fig, ax = plt.subplots(figsize=(11, 5))
    for t, color in enumerate(TYPE_COLORS):
        col = f"type{t}_pct"
        if col in df_c.columns:
            ax.bar(x, df_c[col].values, bottom=bottom,
                   color=color, label=TYPE_NAMES[t], edgecolor="white", linewidth=0.3)
            bottom += df_c[col].values

    # 标注每个聚类的样本数
    for i, (_, row) in enumerate(df_c.iterrows()):
        ax.text(i, 1.01, f"n={int(row['total'])}", ha="center",
                fontsize=6.5, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{int(r['cluster'])}" for _, r in df_c.iterrows()], fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.15)
    ax.set_title("Discourse Type Distribution Across Clusters (MiniLM Best K)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=5,
              bbox_to_anchor=(1, 1.12))
    ax.axhline(0.118, color="gray", linestyle="--", linewidth=0.8,
               label="Nationalist baseline (11.8%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_cluster_discourse.png")
    plt.close()
    print("  [OK] fig5_cluster_discourse.png")


# ─────────────────────────────────────────────────────────
# Fig 6：各话语类型 log-likes 箱线图
# ─────────────────────────────────────────────────────────
def fig6_likes_boxplot():
    p = RESULTS / "finetune/sample_A_predicted.csv"
    if not p.exists():
        print("  [SKIP] sample_A_predicted.csv 不存在")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["likes_raw"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)
    cap = float(min(df["likes_raw"].quantile(0.995), 10000))
    df["likes_capped"] = df["likes_raw"].clip(upper=cap)
    df["log_likes"]    = np.log1p(df["likes_capped"])
    df["dt"] = df["pred_discourse_type"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    data_by_type = [df[df["dt"] == t]["log_likes"].values for t in range(5)]
    bp = ax.boxplot(data_by_type, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=1, alpha=0.3),
                    showfliers=True)
    for patch, color in zip(bp["boxes"], TYPE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # 标注中位数
    for i, (vals, t) in enumerate(zip(data_by_type, range(5)), 1):
        med = np.median(vals)
        ax.text(i, med + 0.05, f"{med:.2f}", ha="center", fontsize=8)

    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(TYPE_NAMES, fontsize=9)
    ax.set_ylabel("log(1 + like_count_winsorized)")
    ax.set_title("Like Count Distribution by Discourse Type\n(Winsorized to p99.5)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_likes_boxplot.png")
    plt.close()
    print("  [OK] fig6_likes_boxplot.png")


# ─────────────────────────────────────────────────────────
# Fig 7：t-SNE 散点图
# ─────────────────────────────────────────────────────────
def fig7_tsne():
    pred_path = RESULTS / "finetune/sample_A_predicted.csv"
    emb_path  = EMBED_DIR / "embeddings_minilm.npy"
    ids_path  = EMBED_DIR / "comment_ids.npy"
    if not all(p.exists() for p in [pred_path, emb_path, ids_path]):
        print("  [SKIP] t-SNE 所需文件不全")
        return

    df_pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    emb     = np.load(emb_path)
    cids    = np.load(ids_path, allow_pickle=True)

    id_to_dt = dict(zip(df_pred["comment_id"].astype(str),
                        df_pred["pred_discourse_type"]))
    dts = np.array([id_to_dt.get(str(c), -1) for c in cids])
    valid = dts >= 0
    emb_v, dts_v = emb[valid], dts[valid]

    # 随机抽 5000 条跑 t-SNE
    np.random.seed(42)
    N = min(5000, len(emb_v))
    idx = np.random.choice(len(emb_v), N, replace=False)
    emb_sub, dt_sub = emb_v[idx], dts_v[idx]

    print(f"  运行 t-SNE (n={N})，约需 1-2 分钟...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    emb_2d = tsne.fit_transform(emb_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for t, (color, name) in enumerate(zip(TYPE_COLORS, TYPE_NAMES)):
        mask = dt_sub == t
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=color, s=4, alpha=0.45, label=name.replace("\n", " "))
    ax.legend(markerscale=4, fontsize=9, loc="upper right")
    ax.set_title("t-SNE of MiniLM Embeddings\n(colored by predicted discourse type, n=5,000)",
                 fontsize=11)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.grid(linestyle="--", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_tsne.png")
    plt.close()
    print("  [OK] fig7_tsne.png")


# ─────────────────────────────────────────────────────────
# Fig 8：时间线（各类型月度占比折线图）
# ─────────────────────────────────────────────────────────
def fig8_timeline():
    p = RESULTS / "finetune/sample_A_predicted.csv"
    if not p.exists():
        print("  [SKIP] sample_A_predicted.csv 不存在")
        return
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["ts"] = pd.to_numeric(df["create_time"], errors="coerce")
    df["dt_obj"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    df = df.dropna(subset=["dt_obj"])
    df["month"] = df["dt_obj"].dt.to_period("M")
    df["dt"] = df["pred_discourse_type"].astype(int)

    monthly = (df.groupby(["month", "dt"]).size()
               .unstack(fill_value=0))
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0)

    if len(monthly_pct) < 2:
        print("  [SKIP] 时间跨度不足，无法绘制时间线")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for t, (color, name) in enumerate(zip(TYPE_COLORS, TYPE_NAMES)):
        if t in monthly_pct.columns:
            ax.plot(monthly_pct.index.astype(str),
                    monthly_pct[t].values,
                    color=color, linewidth=2, marker="o",
                    markersize=4, label=name.replace("\n", " "))

    ax.set_xlabel("Month")
    ax.set_ylabel("Proportion of Comments")
    ax.set_title("Discourse Type Proportion Over Time", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.xticks(rotation=30, ha="right")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_timeline.png")
    plt.close()
    print("  [OK] fig8_timeline.png")


# ── 执行所有图表 ──────────────────────────────────────────
if __name__ == "__main__":
    fig1_hyperparam_heatmap()
    fig2_model_boxplot()
    fig3_confusion_matrix()
    fig4_maxlen_ablation()
    fig5_cluster_discourse()
    fig6_likes_boxplot()
    fig7_tsne()        # 最慢，约1-2分钟
    fig8_timeline()
    print(f"\n[OK] 全部图表已保存到 {FIG_DIR}")
