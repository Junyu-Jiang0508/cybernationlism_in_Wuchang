# -*- coding: utf-8 -*-
"""
26_rq2_figures.py — Phase 2：RQ2 结果图（读取 24_/25_ R 脚本的 CSV 输出）

输出: results/phase2/figures/fig_rq2_marginal_means.png
          nbinom2 GLMM 响应尺度边际均值（每话语类型预测点赞数 + 95% CI）
      results/phase2/figures/fig_rq2_robustness.png
          dt 系数（log-IRR）跨设定 forest：主模型/ZINB/nbinom1/≥5视频/MI 合并
          （IHS-lmer 量纲不同，单独小面板）
"""
import warnings

warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
P2_DIR = SCRIPT_DIR / "results" / "phase2"
FIG_DIR = P2_DIR / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_style import CAT, INK_2, MUTED, apply_style  # noqa: E402

TYPE_LABELS = ["0 game", "1 emotional", "2 political", "3 nationalist", "4 neutral"]


def fig_marginal_means():
    emm = pd.read_csv(P2_DIR / "rq2_emmeans.csv")
    emm["dt"] = emm["dt"].astype(int)
    emm = emm.sort_values("dt")
    lo_col = [c for c in emm.columns
              if "lower" in c.lower() or "lcl" in c.lower()][0]
    hi_col = [c for c in emm.columns
              if "upper" in c.lower() or "ucl" in c.lower()][0]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = np.arange(5)
    est = emm["response"].to_numpy()
    lo, hi = emm[lo_col].to_numpy(), emm[hi_col].to_numpy()
    ax.errorbar(xs, est, yerr=[est - lo, hi - est], fmt="o", ms=8,
                color=CAT["blue"], capsize=4, lw=1.8, zorder=3)
    for x, e in zip(xs, est):
        ax.annotate(f"{e:.1f}", (x, e), xytext=(8, 4), textcoords="offset points",
                    fontsize=9, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl.replace(" ", "\n", 1) for lbl in TYPE_LABELS], fontsize=9)
    ax.set_ylabel("Predicted likes per comment (marginal mean, 95% CI)")
    ax.set_title("NB-GLMM marginal means by discourse type\n"
                 "(video random intercepts, covariate-adjusted, n=29,955)", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "fig_rq2_marginal_means.png")
    plt.close(fig)
    print(f"[OK] {FIG_DIR / 'fig_rq2_marginal_means.png'}")


def fig_robustness():
    rob = pd.read_csv(P2_DIR / "rq2_robustness.csv")
    mi_file = P2_DIR / "rq2_mi_pooled.csv"
    if mi_file.exists():
        mi = pd.read_csv(mi_file)
        mi = mi.rename(columns={"se_pooled": "std.error"})
        mi["model"] = "MI_pooled(M=20)"
        rob = pd.concat([rob, mi[["term", "estimate", "conf.low", "conf.high",
                                  "p.value", "model"]]], ignore_index=True)

    nb_models = [("nbinom2_main", CAT["blue"]), ("zinb", CAT["aqua"]),
                 ("nbinom1", CAT["yellow"]), ("nbinom2_ge5videos", CAT["green"]),
                 ("MI_pooled(M=20)", CAT["violet"])]
    nb_models = [(m, c) for m, c in nb_models if m in set(rob["model"])]
    terms = [f"dt{t}" for t in (1, 2, 3, 4)]
    term_lbl = {f"dt{t}": TYPE_LABELS[t] for t in (1, 2, 3, 4)}

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [3, 1.4]})

    ys = np.arange(len(terms))[::-1]
    n_m = len(nb_models)
    for k, (mname, color) in enumerate(nb_models):
        sub = rob[rob["model"] == mname].set_index("term")
        off = (k - (n_m - 1) / 2) * 0.14
        for yi, t in zip(ys, terms):
            if t not in sub.index:
                continue
            r = sub.loc[t]
            ax1.errorbar(r["estimate"], yi + off,
                         xerr=[[r["estimate"] - r["conf.low"]],
                               [r["conf.high"] - r["estimate"]]],
                         fmt="o", ms=5, color=color, capsize=2.5, lw=1.4,
                         label=mname if yi == ys[0] else None, zorder=3)
    ax1.axvline(0, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([term_lbl[t] for t in terms], fontsize=9)
    ax1.set_xlabel("log(IRR) vs type 0 (game review)")
    ax1.set_title("Count-model specifications", fontsize=10)
    ax1.legend(frameon=False, fontsize=8, loc="best")
    ax1.grid(axis="y", visible=False)

    ihs = rob[rob["model"] == "ihs_lmer"].set_index("term")
    for yi, t in zip(ys, terms):
        if t not in ihs.index:
            continue
        r = ihs.loc[t]
        ax2.errorbar(r["estimate"], yi,
                     xerr=[[r["estimate"] - r["conf.low"]],
                           [r["conf.high"] - r["estimate"]]],
                     fmt="s", ms=5, color=CAT["red"], capsize=2.5, lw=1.4, zorder=3)
    ax2.axvline(0, color=MUTED, lw=1.2, ls="--", zorder=2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([])
    ax2.set_xlabel("IHS-lmer coefficient")
    ax2.set_title("IHS robustness\n(different scale)", fontsize=10)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Discourse-type effects on likes across model specifications "
                 "(95% CI, ref = type 0)", fontsize=12, y=1.0)
    fig.savefig(FIG_DIR / "fig_rq2_robustness.png")
    plt.close(fig)
    print(f"[OK] {FIG_DIR / 'fig_rq2_robustness.png'}")


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_marginal_means()
    fig_robustness()


if __name__ == "__main__":
    main()
