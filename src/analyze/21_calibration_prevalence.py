# -*- coding: utf-8 -*-
"""
21_calibration_prevalence.py — Phase 1：校准 + 语料构成（prevalence）估计（回应 R3-2）

输入: results/phase1/oof_predictions.parquet, results/phase1/sampleA_probs.npz
输出: results/phase1/calibration_metrics.json      方法阶梯 × 模型 的 ECE/Brier/NLL
      results/phase1/prevalence_estimates.json     CC/PCC/ACC/PACC (+bootstrap CI)
      results/phase1/sampleA_probs_calibrated.npz  校准后 Sample A 概率（Phase 2 复用）
      results/phase1/figures/fig_reliability.png   校准前后 reliability diagram（海报主图）
      results/phase1/figures/fig_prevalence.png    构成估计 vs 无偏基准

协议：
  - 校准评估：random_from_A（n=743，无偏）上的 out-of-fold 预测，内部 5 折
    交叉拟合（校准器不见被评样本）；方法阶梯 = temperature → vector → matrix
    （R3 点名）→ Platt per-class。统一以 log-prob 为校准输入，flat 与层级
    合成概率同一口径。
  - Prevalence：Sample A（30k）折间平均概率上做 CC/PCC/ACC/PACC
    （ACC/BBSE：Lipton et al. 2018；quantification 框架：Hopkins & King 2010）；
    误差率/校准器均从 random_from_A OOF 估计；bootstrap（B=1000）重估 CI；
    另做 2 折交叉拟合验证（拟合半样本、对照另半样本经验分布）以避免自证循环。
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
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parents[2]
P1_DIR = SCRIPT_DIR / "results" / "phase1"
FIG_DIR = P1_DIR / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_style import CAT, GRID, INK, INK_2, MUTED, apply_style  # noqa: E402

RNG = np.random.default_rng(42)
CLASS_EN = ["type0_game", "type1_emotional", "type2_political",
            "type3_nationalist", "type4_neutral"]
EPS = 1e-12


# ─── 基础 ────────────────────────────────────────────────
def softmax_np(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def logprob(p):
    return np.log(p + EPS)


# ─── 校准器（统一在 log-prob 上拟合）─────────────────────
def fit_temperature(z, y):
    t = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zt / t.clamp(min=1e-3), yt)
        loss.backward()
        return loss
    opt.step(closure)
    T = float(t.detach().clamp(min=1e-3))
    return {"kind": "temperature", "T": T}


def fit_vector(z, y):
    k = z.shape[1]
    w = torch.nn.Parameter(torch.ones(k, dtype=torch.float64))
    b = torch.nn.Parameter(torch.zeros(k, dtype=torch.float64))
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.LBFGS([w, b], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(zt * w + b, yt)
        loss.backward()
        return loss
    opt.step(closure)
    return {"kind": "vector", "w": w.detach().numpy(), "b": b.detach().numpy()}


def fit_matrix(z, y):
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(z, y)
    return {"kind": "matrix", "lr": lr}


def fit_platt(z, y):
    k = z.shape[1]
    models = []
    for c in range(k):
        lr = LogisticRegression(C=1e6, max_iter=5000)
        lr.fit(z[:, [c]], (y == c).astype(int))
        models.append(lr)
    return {"kind": "platt", "models": models}


def apply_cal(cal, z):
    if cal["kind"] == "none":
        return softmax_np(z)
    if cal["kind"] == "temperature":
        return softmax_np(z / cal["T"])
    if cal["kind"] == "vector":
        return softmax_np(z * cal["w"] + cal["b"])
    if cal["kind"] == "matrix":
        return cal["lr"].predict_proba(z)
    if cal["kind"] == "platt":
        p = np.column_stack([m.predict_proba(z[:, [c]])[:, 1] for c, m in enumerate(cal["models"])])
        return p / p.sum(axis=1, keepdims=True)
    raise ValueError(cal["kind"])


FITTERS = {"uncalibrated": None, "temperature": fit_temperature, "vector": fit_vector,
           "matrix": fit_matrix, "platt_per_class": fit_platt}


# ─── 校准指标 ────────────────────────────────────────────
def ece_top(p, y, n_bins=15):
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def ece_classwise(p, y, n_bins=15):
    k = p.shape[1]
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for c in range(k):
        pc, yc = p[:, c], (y == c).astype(float)
        e = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (pc > lo) & (pc <= hi)
            if m.sum():
                e += m.mean() * abs(yc[m].mean() - pc[m].mean())
        total += e
    return total / k


def cal_metrics(p, y):
    onehot = np.eye(p.shape[1])[y]
    return {
        "ece_top": round(float(ece_top(p, y)), 4),
        "ece_classwise": round(float(ece_classwise(p, y)), 4),
        "brier": round(float(((p - onehot) ** 2).sum(axis=1).mean()), 4),
        "nll": round(float(-np.log(p[np.arange(len(y)), y] + EPS).mean()), 4),
        "accuracy": round(float(accuracy_score(y, p.argmax(axis=1))), 4),
        "macro_f1": round(float(f1_score(y, p.argmax(axis=1), average="macro")), 4),
    }


def crossfit_calibrated(z, y, fitter, n_splits=5, seed=123):
    """内部交叉拟合：每个样本的校准概率来自未见过它的校准器。"""
    out = np.zeros((len(y), z.shape[1]))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(z, y):
        cal = {"kind": "none"} if fitter is None else fitter(z[tr], y[tr])
        out[te] = apply_cal(cal, z[te])
    return out


# ─── Prevalence 估计器 ───────────────────────────────────
def solve_simplex(M, p_hat):
    """min ||M^T π − p̂||²  s.t. π ∈ simplex（ACC/PACC 的约束解）。"""
    k = M.shape[0]

    def obj(pi):
        return ((M.T @ pi - p_hat) ** 2).sum()
    cons = [{"type": "eq", "fun": lambda pi: pi.sum() - 1}]
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=cons, options={"maxiter": 500})
    return res.x


def conf_rates(y, pred, k=5):
    """M[i,j] = P(pred=j | true=i)（行随真类归一化）。"""
    M = np.zeros((k, k))
    for i in range(k):
        m = y == i
        if m.sum():
            M[i] = np.bincount(pred[m], minlength=k) / m.sum()
    return M


def soft_conf_rates(y, p, k=5):
    """PACC 版：M[i,·] = E[p | true=i]。"""
    M = np.zeros((k, k))
    for i in range(k):
        m = y == i
        if m.sum():
            M[i] = p[m].mean(axis=0)
    return M


def prevalence_suite(y_val, p_val_raw, z_val, p_sa_raw, z_sa, fit_cal_fn, full_M=None):
    """给定验证集（OOF）与 Sample A 概率，返回全部估计器的点估计。
    fit_cal_fn(z, y) → calibrator（每次调用重新拟合，供 bootstrap 复用）。
    full_M=(y_all, p_all)：用全部标注（含关键词池）估计混淆率的 BBSE 变体——
    label-shift 假设下合法（P(pred|true) 与类先验无关），type3 的行估计
    从 n=6 提升到 n=149；代价是关键词池的类内文本分布与总体不同，
    class-conditional invariance 可能被削弱，作为稳健性对照报告。"""
    cal = fit_cal_fn(z_val, y_val)
    p_sa_cal = apply_cal(cal, z_sa)
    pred_val_raw = p_val_raw.argmax(axis=1)
    est = {
        "CC_raw": np.bincount(p_sa_raw.argmax(axis=1), minlength=5) / len(p_sa_raw),
        "PCC_raw": p_sa_raw.mean(axis=0),
        "CC_calibrated": np.bincount(p_sa_cal.argmax(axis=1), minlength=5) / len(p_sa_cal),
        "PCC_calibrated": p_sa_cal.mean(axis=0),
        "ACC_BBSE": solve_simplex(conf_rates(y_val, pred_val_raw),
                                  np.bincount(p_sa_raw.argmax(axis=1), minlength=5) / len(p_sa_raw)),
        "PACC": solve_simplex(soft_conf_rates(y_val, p_val_raw), p_sa_raw.mean(axis=0)),
    }
    if full_M is not None:
        y_all, p_all = full_M
        pred_all = p_all.argmax(axis=1)
        est["ACC_BBSE_fullM"] = solve_simplex(
            conf_rates(y_all, pred_all),
            np.bincount(p_sa_raw.argmax(axis=1), minlength=5) / len(p_sa_raw))
        est["PACC_fullM"] = solve_simplex(soft_conf_rates(y_all, p_all), p_sa_raw.mean(axis=0))
    return {k_: v.round(4).tolist() for k_, v in est.items()}


def main():
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(P1_DIR / "oof_predictions.parquet")
    sa = np.load(P1_DIR / "sampleA_probs.npz", allow_pickle=True)

    y_all = df["discourse_type"].to_numpy()
    is_rand = (df["sample_b_source"] == "random_from_A").to_numpy()

    # 统一口径：log-prob 作为校准输入
    def flat_probs(frame_idx):
        z = df.loc[frame_idx, [f"flat5_logit{j}" for j in range(5)]].to_numpy()
        return softmax_np(z)

    p_flat_all = flat_probs(df.index)
    pa = softmax_np(df[[f"stageA_logit{j}" for j in range(2)]].to_numpy())
    pb = softmax_np(df[[f"stageB_logit{j}" for j in range(2)]].to_numpy())
    pc = softmax_np(df[[f"stageC_logit{j}" for j in range(3)]].to_numpy())
    p_hier_all = np.zeros_like(p_flat_all)
    p_hier_all[:, 2] = pa[:, 1] * pb[:, 0]
    p_hier_all[:, 3] = pa[:, 1] * pb[:, 1]
    p_hier_all[:, 0] = pa[:, 0] * pc[:, 0]
    p_hier_all[:, 1] = pa[:, 0] * pc[:, 1]
    p_hier_all[:, 4] = pa[:, 0] * pc[:, 2]

    y = y_all[is_rand]
    models = {"flat5": p_flat_all[is_rand], "hierarchical": p_hier_all[is_rand]}
    print(f"校准验证集 random_from_A: n={len(y)}, 分布={np.bincount(y, minlength=5).tolist()}")

    # ── 1) 校准方法阶梯（内部 5 折交叉拟合）──
    cal_results = {}
    crossfit_probs = {}
    for mname, p_raw in models.items():
        z = logprob(p_raw)
        cal_results[mname] = {}
        for method, fitter in FITTERS.items():
            p_cal = crossfit_calibrated(z, y, fitter)
            cal_results[mname][method] = cal_metrics(p_cal, y)
            crossfit_probs[(mname, method)] = p_cal
        print(f"[{mname}] " + " | ".join(
            f"{m}: ECE={r['ece_top']:.3f}" for m, r in cal_results[mname].items()))

    (P1_DIR / "calibration_metrics.json").write_text(
        json.dumps(cal_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 校准指标 → {P1_DIR / 'calibration_metrics.json'}")

    # ── 2) Reliability diagram：flat5 校准前 vs matrix 后（海报主图）──
    def reliability_panel(ax_main, ax_hist, p, label, color):
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y).astype(float)
        bins = np.linspace(0, 1, 16)
        mids, accs, fracs = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum():
                mids.append((lo + hi) / 2)
                accs.append(correct[m].mean())
                fracs.append(m.mean())
        mids, accs = np.array(mids), np.array(accs)
        ax_main.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, zorder=2)
        ax_main.bar(mids, accs, width=1 / 15 * 0.92, color=color, zorder=3)
        for x0, a in zip(mids, accs):  # gap 标示
            ax_main.plot([x0, x0], [a, x0], color=CAT["red"], lw=2, alpha=0.65, zorder=4)
        e = ece_top(p, y)
        ax_main.set_title(f"{label}\nECE = {e:.3f}", fontsize=10)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel("Observed accuracy", fontsize=9)
        ax_hist.bar(np.linspace(1 / 30, 1 - 1 / 30, 15),
                    np.histogram(conf, bins=bins)[0], width=1 / 15 * 0.92,
                    color=GRID, zorder=3)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_ylabel("n", fontsize=8)
        ax_hist.set_xlabel("Predicted confidence (top label)", fontsize=9)

    fig = plt.figure(figsize=(9.5, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], hspace=0.12, wspace=0.22)
    panels = [("uncalibrated", "Before calibration", CAT["blue"]),
              ("matrix", "After matrix scaling (R3)", CAT["aqua"])]
    for col, (method, label, color) in enumerate(panels):
        reliability_panel(fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col]),
                          crossfit_probs[("flat5", method)], label, color)
    fig.suptitle("Reliability diagram — flat 5-way, random_from_A (n=743), cross-fitted",
                 fontsize=12, y=1.02, color=INK)
    fig.savefig(FIG_DIR / "fig_reliability.png")
    plt.close(fig)
    print(f"[OK] 图 → {FIG_DIR / 'fig_reliability.png'}")

    # ── 3) Prevalence 估计（Sample A 30k）──
    z_val_flat = logprob(models["flat5"])
    z_val_hier = logprob(models["hierarchical"])
    p_sa_flat = sa["flat5_probs_mean"]
    p_sa_hier = sa["hier_probs_mean"]
    z_sa_flat, z_sa_hier = logprob(p_sa_flat), logprob(p_sa_hier)

    bench_counts = np.bincount(y, minlength=5)
    bench = bench_counts / bench_counts.sum()

    prev = {"benchmark_random_from_A": {
        "n": int(bench_counts.sum()),
        "prevalence": bench.round(4).tolist(),
        "wilson_ci95": [
            [round(x, 4) for x in _wilson(bench_counts[c], bench_counts.sum())] for c in range(5)],
    }}

    suites = {}
    model_inputs = {
        "flat5": (models["flat5"], z_val_flat, p_sa_flat, z_sa_flat, p_flat_all),
        "hierarchical": (models["hierarchical"], z_val_hier, p_sa_hier, z_sa_hier, p_hier_all),
    }
    for mname, (p_val, z_val, p_sa_m, z_sa_m, p_all_m) in model_inputs.items():
        suites[mname] = prevalence_suite(y, p_val, z_val, p_sa_m, z_sa_m, fit_matrix,
                                         full_M=(y_all, p_all_m))

        # bootstrap CI（重采样验证集与全标注集，重估 M 与校准器）
        boots = {k_: [] for k_ in suites[mname]}
        for _ in range(1000):
            idx = RNG.integers(0, len(y), len(y))
            idx_all = RNG.integers(0, len(y_all), len(y_all))
            if len(np.unique(y[idx])) < 5 or len(np.unique(y_all[idx_all])) < 5:
                continue
            b = prevalence_suite(y[idx], p_val[idx], z_val[idx], p_sa_m, z_sa_m, fit_matrix,
                                 full_M=(y_all[idx_all], p_all_m[idx_all]))
            for k_ in boots:
                boots[k_].append(b[k_])
        prev[mname] = {}
        for k_, v in suites[mname].items():
            arr = np.array(boots[k_])
            prev[mname][k_] = {
                "estimate": v,
                "ci95_low": np.percentile(arr, 2.5, axis=0).round(4).tolist(),
                "ci95_high": np.percentile(arr, 97.5, axis=0).round(4).tolist(),
            }
        print(f"[{mname}] prevalence 点估计与 bootstrap CI 完成")

    # 交叉拟合验证：半样本拟合 → 另半样本经验分布为准（20 次重复）
    xfit = {m: {k_: [] for k_ in suites[m]} for m in suites}
    for rep in range(20):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=rep)
        for fit_i, hold_i in skf.split(z_val_flat, y):
            hold_dist = np.bincount(y[hold_i], minlength=5) / len(hold_i)
            for mname, (p_val, z_val, p_sa_m, z_sa_m, p_all_m) in model_inputs.items():
                est = prevalence_suite(y[fit_i], p_val[fit_i], z_val[fit_i],
                                       p_sa_m, z_sa_m, fit_matrix,
                                       full_M=(y_all, p_all_m))
                for k_, v in est.items():
                    xfit[mname][k_].append(np.abs(np.array(v) - hold_dist).mean())
    prev["crossfit_validation_MAE"] = {
        m: {k_: round(float(np.mean(v)), 4) for k_, v in d.items()} for m, d in xfit.items()}

    # 上下文参照：3 月 best_model 的原始 CC（有训练泄漏 + 未校准，仅作 before 对照）
    march = pd.read_csv(SCRIPT_DIR / "results" / "finetune" / "sample_A_predicted.csv",
                        usecols=["pred_discourse_type"])
    prev["march_best_model_CC_reference"] = (
        np.bincount(march["pred_discourse_type"], minlength=5) / len(march)).round(4).tolist()

    (P1_DIR / "prevalence_estimates.json").write_text(
        json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] prevalence → {P1_DIR / 'prevalence_estimates.json'}")

    # 校准后 Sample A 概率存档（Phase 2 测量误差插补复用）
    cal_flat = fit_matrix(z_val_flat, y)
    cal_hier = fit_matrix(z_val_hier, y)
    np.savez_compressed(
        P1_DIR / "sampleA_probs_calibrated.npz",
        comment_id=sa["comment_id"],
        flat5_probs_calibrated=apply_cal(cal_flat, z_sa_flat).astype(np.float32),
        hier_probs_calibrated=apply_cal(cal_hier, z_sa_hier).astype(np.float32),
    )
    print(f"[OK] 校准后概率 → {P1_DIR / 'sampleA_probs_calibrated.npz'}")

    # ── 4) Prevalence 图：估计器 vs 无偏基准 ──
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    xs = np.arange(5)
    band = prev["benchmark_random_from_A"]["wilson_ci95"]
    for c in range(5):
        ax.fill_between([c - 0.42, c + 0.42], band[c][0], band[c][1],
                        color=GRID, alpha=0.9, zorder=1,
                        label="Benchmark (random n=743, Wilson 95%)" if c == 0 else None)
        ax.plot([c - 0.42, c + 0.42], [bench[c]] * 2, color=INK_2, lw=1.4, zorder=2)
    show = [("CC_raw", CAT["blue"], "CC (uncalibrated)"),
            ("PCC_calibrated", CAT["aqua"], "PCC (matrix-calibrated)"),
            ("ACC_BBSE", CAT["violet"], "ACC/BBSE")]
    for off, (key, color, label) in zip([-0.22, 0.0, 0.22], show):
        d = prev["flat5"][key]
        est = np.array(d["estimate"])
        lo, hi = np.array(d["ci95_low"]), np.array(d["ci95_high"])
        ax.errorbar(xs + off, est, yerr=[est - lo, hi - est], fmt="o", ms=6,
                    color=color, capsize=3, lw=1.6, label=label, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in CLASS_EN], fontsize=9)
    ax.set_ylabel("Estimated prevalence in Sample A")
    ax.set_title("Corpus composition: uncalibrated CC vs calibrated / quantification estimators (flat 5-way)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG_DIR / "fig_prevalence.png")
    plt.close(fig)
    print(f"[OK] 图 → {FIG_DIR / 'fig_prevalence.png'}")


def _wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


if __name__ == "__main__":
    main()
