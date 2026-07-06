# -*- coding: utf-8 -*-
"""
20_hierarchical_vs_flat.py — Phase 1：两步层级分类 vs 五路 softmax 对比（回应 R3-1）

输入: results/phase1/oof_predictions.parquet（19_ 脚本 merge 产出）
输出: results/phase1/hierarchical_vs_flat.json
      results/phase1/figures/fig_f1_perclass.png（逐类 F1 对比）
      results/phase1/figures/fig_confusion.png（混淆矩阵并排）

对比口径（全部基于同折 out-of-fold 预测，无泄漏）：
  flat   = argmax(flat5 logits)
  hier   = 硬路由：stageA 判政治与否 → 政治走 stageB(2/3)，非政治走 stageC(0/1/4)
  hier_soft = 概率合成后 argmax（校准用的口径，作参考）
  alt    = 对照切法：altA 判实质性 → 实质走 altB(2/3/4)，输出 4 桶 {01, 2, 3, 4}

核心检验：Type1↔2 边界错误是否被隔离在 Stage A 内部、
Stage B 在真实政治子集内的 2/3 区分度是否高于 flat 的受限 argmax。
"""
import warnings

warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

SCRIPT_DIR = Path(__file__).resolve().parents[2]
P1_DIR = SCRIPT_DIR / "results" / "phase1"
FIG_DIR = P1_DIR / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_style import CAT, CMAP_SEQ, INK, INK_2, MUTED, apply_style  # noqa: E402

RNG = np.random.default_rng(42)
# 图内文字用英文：渲染环境无 CJK 字体，且海报/论文均为英文
CLASS_NAMES = ["0 game", "1 emotional", "2 political", "3 nationalist", "4 neutral"]
CLASS_NAMES_EN = ["type0_game", "type1_emotional", "type2_political",
                  "type3_nationalist", "type4_neutral"]


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def logits_of(df, task, n):
    return df[[f"{task}_logit{j}" for j in range(n)]].to_numpy()


def hier_hard_route(pa, pb, pc):
    """硬路由层级预测：Stage A argmax 决定走 B 还是 C。"""
    pred = np.empty(len(pa), dtype=int)
    pol = pa.argmax(axis=1) == 1
    pred[pol] = np.where(pb[pol].argmax(axis=1) == 0, 2, 3)
    nonpol_map = np.array([0, 1, 4])
    pred[~pol] = nonpol_map[pc[~pol].argmax(axis=1)]
    return pred


def hier_soft_probs(pa, pb, pc):
    """概率合成：P(2)=P(pol)P(B=2), P(0/1/4)=P(nonpol)P(C=·)。"""
    out = np.zeros((len(pa), 5))
    out[:, 2] = pa[:, 1] * pb[:, 0]
    out[:, 3] = pa[:, 1] * pb[:, 1]
    out[:, 0] = pa[:, 0] * pc[:, 0]
    out[:, 1] = pa[:, 0] * pc[:, 1]
    out[:, 4] = pa[:, 0] * pc[:, 2]
    return out


def per_class_f1(y, pred):
    return f1_score(y, pred, average=None, labels=list(range(5))).round(4).tolist()


def summarize(y, pred):
    return {
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "per_class_f1": dict(zip(CLASS_NAMES_EN, per_class_f1(y, pred))),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(5))).tolist(),
    }


def paired_bootstrap_delta(y, pred_a, pred_b, n_boot=2000):
    """macro-F1(B) − macro-F1(A) 的配对 bootstrap 95% CI。"""
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        deltas[i] = (f1_score(y[idx], pred_b[idx], average="macro")
                     - f1_score(y[idx], pred_a[idx], average="macro"))
    return {"delta_mean": round(float(deltas.mean()), 4),
            "ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                     round(float(np.percentile(deltas, 97.5)), 4)]}


def mcnemar(y, pred_a, pred_b):
    """精确 McNemar（二项）：flat 对 vs hier 对的不一致格。"""
    from scipy.stats import binomtest
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int((~a_ok & b_ok).sum())   # 只有 hier 对
    n10 = int((a_ok & ~b_ok).sum())   # 只有 flat 对
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return {"hier_only_correct": n01, "flat_only_correct": n10, "p_value": round(float(p), 5)}


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(P1_DIR / "oof_predictions.parquet")
    y = df["discourse_type"].to_numpy()
    is_random = (df["sample_b_source"] == "random_from_A").to_numpy()

    p_flat = softmax(logits_of(df, "flat5", 5))
    pa = softmax(logits_of(df, "stageA", 2))
    pb = softmax(logits_of(df, "stageB", 2))
    pc = softmax(logits_of(df, "stageC", 3))
    p_alt_a = softmax(logits_of(df, "altA", 2))
    p_alt_b = softmax(logits_of(df, "altB", 3))

    pred_flat = p_flat.argmax(axis=1)
    pred_hier = hier_hard_route(pa, pb, pc)
    pred_soft = hier_soft_probs(pa, pb, pc).argmax(axis=1)

    res = {"n": len(df), "n_random_from_A": int(is_random.sum())}

    # ── 总体对比 ──
    res["flat"] = summarize(y, pred_flat)
    res["hier_hard"] = summarize(y, pred_hier)
    res["hier_soft"] = summarize(y, pred_soft)
    res["flat_random_subset"] = summarize(y[is_random], pred_flat[is_random])
    res["hier_hard_random_subset"] = summarize(y[is_random], pred_hier[is_random])
    res["bootstrap_delta_macroF1_hier_minus_flat"] = paired_bootstrap_delta(y, pred_flat, pred_hier)
    res["mcnemar"] = mcnemar(y, pred_flat, pred_hier)

    # ── Stage A 单独：政治 vs 非政治 ──
    y_bin = np.isin(y, [2, 3]).astype(int)
    a_pred = pa.argmax(axis=1)
    flat_bin = np.isin(pred_flat, [2, 3]).astype(int)
    res["stageA_binary"] = {
        "stageA_f1_macro": round(f1_score(y_bin, a_pred, average="macro"), 4),
        "flat_derived_f1_macro": round(f1_score(y_bin, flat_bin, average="macro"), 4),
        "stageA_confusion": confusion_matrix(y_bin, a_pred).tolist(),
    }

    # ── Stage B 单独：真实政治子集内 2 vs 3 ──
    pol_mask = np.isin(y, [2, 3])
    y_pol = (y[pol_mask] == 3).astype(int)
    b_pred = pb[pol_mask].argmax(axis=1)
    flat_restricted = (p_flat[pol_mask][:, 3] > p_flat[pol_mask][:, 2]).astype(int)
    res["stageB_within_political"] = {
        "n": int(pol_mask.sum()),
        "stageB_f1_macro": round(f1_score(y_pol, b_pred, average="macro"), 4),
        "stageB_accuracy": round(accuracy_score(y_pol, b_pred), 4),
        "flat_restricted_f1_macro": round(f1_score(y_pol, flat_restricted, average="macro"), 4),
        "flat_restricted_accuracy": round(accuracy_score(y_pol, flat_restricted), 4),
    }

    # ── Type1↔2 边界错误 ──
    cm_flat = confusion_matrix(y, pred_flat, labels=list(range(5)))
    cm_hier = confusion_matrix(y, pred_hier, labels=list(range(5)))
    res["type12_boundary_errors"] = {
        "flat_1to2": int(cm_flat[1, 2]), "flat_2to1": int(cm_flat[2, 1]),
        "hier_1to2": int(cm_hier[1, 2]), "hier_2to1": int(cm_hier[2, 1]),
        "flat_total": int(cm_flat[1, 2] + cm_flat[2, 1]),
        "hier_total": int(cm_hier[1, 2] + cm_hier[2, 1]),
    }

    # ── 对照切法：{0,1} / 2 / 3 / 4 四桶 ──
    def to_bucket(labels5):
        return np.where(np.isin(labels5, [0, 1]), 0, labels5)  # 0=非实质桶
    alt_pred = np.empty(len(df), dtype=int)
    subst = p_alt_a.argmax(axis=1) == 1
    alt_map = np.array([2, 3, 4])
    alt_pred[subst] = alt_map[p_alt_b[subst].argmax(axis=1)]
    alt_pred[~subst] = 0
    y_bucket, flat_bucket = to_bucket(y), to_bucket(pred_flat)
    res["alt_hierarchy_4bucket"] = {
        "alt_macro_f1": round(f1_score(y_bucket, alt_pred, average="macro"), 4),
        "flat_macro_f1": round(f1_score(y_bucket, flat_bucket, average="macro"), 4),
        "hier_macro_f1": round(f1_score(y_bucket, to_bucket(pred_hier), average="macro"), 4),
    }

    out = P1_DIR / "hierarchical_vs_flat.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 指标 → {out}")

    # ── 图 1：逐类 F1 对比（分组条形，直接标数） ──
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(6)
    f_flat = per_class_f1(y, pred_flat) + [res["flat"]["macro_f1"]]
    f_hier = per_class_f1(y, pred_hier) + [res["hier_hard"]["macro_f1"]]
    w = 0.36
    b1 = ax.bar(xs - w / 2, f_flat, w, color=CAT["blue"], label="Flat 5-way", zorder=3)
    b2 = ax.bar(xs + w / 2, f_hier, w, color=CAT["aqua"], label="Hierarchical", zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in CLASS_NAMES] + ["Macro\nF1"], fontsize=9)
    ax.set_ylabel("Out-of-fold F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Hierarchical vs flat: per-class F1 (5-fold out-of-fold, n=1931)", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "fig_f1_perclass.png")
    plt.close(fig)

    # ── 图 2：混淆矩阵并排（行归一化，顺序蓝 ramp） ──
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax_i, (cm, name) in zip(axes, [(cm_flat, "Flat 5-way"), (cm_hier, "Hierarchical")]):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ax_i.imshow(cm_norm, cmap=CMAP_SEQ, vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                color = "#ffffff" if cm_norm[i, j] > 0.5 else INK
                ax_i.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9, color=color)
        ax_i.set_xticks(range(5), [f"pred {k}" for k in range(5)], fontsize=8, color=MUTED)
        ax_i.set_yticks(range(5), [f"true {k}" for k in range(5)], fontsize=8, color=MUTED)
        ax_i.set_title(name, fontsize=11)
        ax_i.grid(visible=False)
        for s in ax_i.spines.values():
            s.set_visible(False)
    fig.suptitle("Out-of-fold confusion matrices (counts; color = row-normalized)", fontsize=12, y=1.0)
    fig.savefig(FIG_DIR / "fig_confusion.png")
    plt.close(fig)
    print(f"[OK] 图 → {FIG_DIR}/fig_f1_perclass.png, fig_confusion.png")

    # 摘要打印
    print(f"\nflat macro-F1 = {res['flat']['macro_f1']}, hier macro-F1 = {res['hier_hard']['macro_f1']}")
    print(f"Δ(hier−flat) = {res['bootstrap_delta_macroF1_hier_minus_flat']}")
    print(f"Stage B (2 vs 3) within true political: hier={res['stageB_within_political']['stageB_f1_macro']}"
          f" vs flat_restricted={res['stageB_within_political']['flat_restricted_f1_macro']}")
    print(f"Type1↔2 errors: flat={res['type12_boundary_errors']['flat_total']}"
          f" hier={res['type12_boundary_errors']['hier_total']}")


if __name__ == "__main__":
    main()
