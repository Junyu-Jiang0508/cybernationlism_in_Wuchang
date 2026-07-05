# -*- coding: utf-8 -*-
"""
聚类超参扫描：双模型 embedding → PCA(50) → KMeans(k∈{10,20,50,100}) + DBSCAN 对比。

输入:
  embeddings/embeddings_minilm.npy
  embeddings/embeddings_text2vec.npy
  embeddings/comment_ids.npy

输出:
  results/sweep_results.csv
  results/sweep_log.txt
  results/pca_variance_{minilm,text2vec}.csv   # 累计解释方差（含 80% 对应维数）
  results/kdistance_{minilm,text2vec}_ms{5,10}.png
  results/labels_{minilm,text2vec}_k{k}.npy    # 各 k 的标签（可选 --save-all-k）
  results/labels_{minilm,text2vec}_best.npy    # 按 Silhouette 自动选最优 k

用法:
  python 12_clustering_sweep.py
  python 12_clustering_sweep.py --embeddings-dir embeddings --outdir results --save-all-k
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError as e:
    raise SystemExit(
        "需要 scikit-learn: pip install scikit-learn\n" + str(e)
    ) from e

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


def log(msg: str, log_lines: List[str]) -> None:
    print(msg)
    log_lines.append(msg)


def pca_cumulative_variance(
    X: np.ndarray, max_components: int, name: str, out_csv: Path, log_lines: List[str]
) -> Tuple[PCA, np.ndarray, int]:
    """拟合 PCA 至 max_components，写累计方差 CSV，返回 n_80（达 80% 的最小维数）。"""
    n_comp = min(max_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    idx80 = int(np.searchsorted(cum, 0.8))
    n_80 = (idx80 + 1) if idx80 < len(cum) else None
    rows = []
    for i, (r, c) in enumerate(
        zip(pca.explained_variance_ratio_, cum), start=1
    ):
        rows.append(
            {
                "n_components": i,
                "explained_variance_ratio": float(r),
                "cumulative_ratio": float(c),
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    if n_80 is not None:
        log(
            f"  [{name}] PCA 1..{n_comp}: 累计达 80% 约需 {n_80} 维 "
            f"（{n_comp} 维累计={cum[-1]:.4f}）",
            log_lines,
        )
    else:
        log(
            f"  [{name}] PCA 1..{n_comp}: 前 {n_comp} 维累计方差未达 80% "
            f"（最大累计={cum[-1]:.4f}，见 CSV）",
            log_lines,
        )
    return pca, cum, n_80 or n_comp


def fit_pca_fixed(
    X: np.ndarray, n_components: int, random_state: int = 42
) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(X)


def knee_eps_from_kdist(k_dist: np.ndarray) -> float:
    """标准 k-distance 图（降序）上，到首尾连线的最大垂距点 → 拐点 eps。"""
    y = np.sort(np.asarray(k_dist, dtype=np.float64))[::-1]
    n = len(y)
    if n < 10:
        return float(np.median(k_dist))
    x = np.arange(n, dtype=np.float64)
    x1, y1 = 0.0, float(y[0])
    x2, y2 = float(n - 1), float(y[-1])
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy + 1e-12
    t = np.clip(((x - x1) * dx + (y - y1) * dy) / denom, 0, 1)
    px = x1 + t * dx
    py = y1 + t * dy
    dist = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    i = int(np.argmax(dist))
    return float(max(y[i], 1e-10))


def plot_kdistance(
    k_dist: np.ndarray, eps: float, title: str, out_png: Path
) -> None:
    if plt is None:
        return
    s = np.sort(k_dist)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(s)), s, lw=0.8, color="steelblue")
    ax.axhline(eps, color="crimson", ls="--", label=f"eps={eps:.6f}")
    ax.set_xlabel("points (sorted by k-distance desc)")
    ax.set_ylabel("k-distance")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def cluster_size_ratio(labels: np.ndarray, k: int) -> float:
    sizes = np.bincount(labels, minlength=k)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return float("nan")
    return float(sizes.max() / max(sizes.min(), 1))


def run_kmeans_sweep(
    X_pca: np.ndarray,
    model_slug: str,
    ks: List[int],
    silhouette_sample: int,
    random_state: int,
    log_lines: List[str],
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    rows = []
    label_by_k: Dict[int, np.ndarray] = {}
    n = len(X_pca)
    sil_sample = min(silhouette_sample, n)

    for k in ks:
        t0 = time.perf_counter()
        km = KMeans(
            n_clusters=k,
            n_init=10,
            max_iter=300,
            random_state=random_state,
            algorithm="lloyd",
        )
        lab = km.fit_predict(X_pca)
        elapsed = time.perf_counter() - t0
        label_by_k[k] = lab

        sil = silhouette_score(
            X_pca,
            lab,
            metric="euclidean",
            sample_size=sil_sample,
            random_state=random_state,
        )
        dbi = davies_bouldin_score(X_pca, lab)
        ratio = cluster_size_ratio(lab, k)

        rows.append(
            {
                "method": "KMeans",
                "model": model_slug,
                "k_or_ms": k,
                "silhouette_sample_size": sil_sample,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "max_min_cluster_ratio": ratio,
                "n_clusters_effective": k,
                "noise_ratio": "",
                "max_cluster_size": int(np.bincount(lab, minlength=k).max()),
                "seconds": round(elapsed, 3),
                "notes": "",
            }
        )
        log(
            f"  KMeans k={k}: Silhouette={sil:.4f} DBI={dbi:.4f} "
            f"max/min_size={ratio:.2f} t={elapsed:.1f}s",
            log_lines,
        )

    return pd.DataFrame(rows), label_by_k


def run_dbscan(
    X_pca: np.ndarray,
    model_slug: str,
    min_samples: int,
    out_png: Path | None,
    log_lines: List[str],
) -> Dict:
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_pca)
    dist, _ = nn.kneighbors(X_pca)
    k_dist = dist[:, -1]
    eps = knee_eps_from_kdist(k_dist)
    if out_png and plt:
        plot_kdistance(
            k_dist,
            eps,
            f"{model_slug} DBSCAN kNN min_samples={min_samples}",
            out_png,
        )

    t0 = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X_pca)
    elapsed = time.perf_counter() - t0
    lab = db.labels_
    n_noise = int((lab == -1).sum())
    noise_ratio = n_noise / len(lab)
    uniq = set(lab.tolist())
    uniq.discard(-1)
    n_clu = len(uniq)
    if n_clu == 0:
        max_sz = 0
    else:
        mask = lab >= 0
        if not mask.any():
            max_sz = 0
        else:
            max_sz = int(
                max(np.sum(lab == c) for c in uniq)
            )

    log(
        f"  DBSCAN min_samples={min_samples}: eps={eps:.6f} n_clusters={n_clu} "
        f"noise={100 * noise_ratio:.2f}% max_cluster={max_sz} t={elapsed:.1f}s",
        log_lines,
    )
    return {
        "method": "DBSCAN",
        "model": model_slug,
        "k_or_ms": min_samples,
        "silhouette_sample_size": "",
        "silhouette": "",
        "davies_bouldin": "",
        "max_min_cluster_ratio": "",
        "n_clusters_effective": n_clu,
        "noise_ratio": round(noise_ratio, 6),
        "max_cluster_size": max_sz,
        "seconds": round(elapsed, 3),
        "notes": f"eps={eps:.8f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="聚类超参扫描 KMeans + DBSCAN")
    ap.add_argument(
        "--embeddings-dir",
        type=Path,
        default=SCRIPT_DIR / "embeddings",
    )
    ap.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "results")
    ap.add_argument(
        "--pca-dim",
        type=int,
        default=50,
        help="KMeans/DBSCAN 使用的 PCA 维数（对称实验）",
    )
    ap.add_argument(
        "--pca-scan-max",
        type=int,
        default=80,
        help="仅用于方差曲线扫描的最大 PCA 维数",
    )
    ap.add_argument("--k-list", default="10,20,50,100")
    ap.add_argument("--silhouette-sample", type=int, default=3000)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument(
        "--save-all-k",
        action="store_true",
        help="保存每个 k 的 labels_*.npy；否则只保存 best",
    )
    args = ap.parse_args()

    emb_dir = args.embeddings_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    path_m = emb_dir / "embeddings_minilm.npy"
    path_t = emb_dir / "embeddings_text2vec.npy"
    path_ids = emb_dir / "comment_ids.npy"

    for p in (path_m, path_t, path_ids):
        if not p.is_file():
            print(f"缺失文件: {p}", file=sys.stderr)
            sys.exit(1)

    X_m = np.load(path_m)
    X_t = np.load(path_t)
    ids = np.load(path_ids, allow_pickle=True)
    n = len(ids)
    if X_m.shape[0] != n or X_t.shape[0] != n:
        raise SystemExit(
            f"行数不一致: ids={n} minilm={X_m.shape[0]} text2vec={X_t.shape[0]}"
        )

    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    log_lines: List[str] = []

    log(f"样本数 n={n}  MiniLM shape={X_m.shape}  text2vec shape={X_t.shape}", log_lines)
    log(f"PCA 固定维数={args.pca_dim}（KMeans/DBSCAN）", log_lines)

    all_rows: List[Dict] = []

    for slug, X_raw in [("minilm", X_m.astype(np.float64)), ("text2vec", X_t.astype(np.float64))]:
        log(f"\n=== {slug} ===", log_lines)
        var_csv = outdir / f"pca_variance_{slug}.csv"
        _, _, n80 = pca_cumulative_variance(
            X_raw, args.pca_scan_max, slug, var_csv, log_lines
        )
        log(f"  （参考）80% 方差约 {n80} 维，见 {var_csv.name}", log_lines)

        X_pca = fit_pca_fixed(X_raw, args.pca_dim, args.random_state)
        df_km, label_by_k = run_kmeans_sweep(
            X_pca,
            slug,
            ks,
            args.silhouette_sample,
            args.random_state,
            log_lines,
        )
        all_rows.extend(df_km.to_dict("records"))

        # 按 Silhouette 选最优 k
        best_k = max(ks, key=lambda k: df_km.loc[df_km["k_or_ms"] == k, "silhouette"].iloc[0])
        best_lab = label_by_k[best_k]
        np.save(outdir / f"labels_{slug}_best.npy", best_lab)
        log(f"  → 最优 k={best_k}（按 Silhouette），已写 labels_{slug}_best.npy", log_lines)

        if args.save_all_k:
            for k, lab in label_by_k.items():
                np.save(outdir / f"labels_{slug}_k{k}.npy", lab)

        for ms in (5, 10):
            png = outdir / f"kdistance_{slug}_ms{ms}.png"
            row = run_dbscan(
                X_pca, slug, ms, png if plt else None, log_lines
            )
            all_rows.append(row)

    sweep_csv = outdir / "sweep_results.csv"
    pd.DataFrame(all_rows).to_csv(sweep_csv, index=False, encoding="utf-8-sig")
    log(f"\n已写入 {sweep_csv}", log_lines)

    log_path = outdir / "sweep_log.txt"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"日志: {log_path}")

    if plt is None:
        print(
            "提示: 未安装 matplotlib，已跳过 k-distance 图；"
            "pip install matplotlib 可生成 PNG",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
