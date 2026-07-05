# -*- coding: utf-8 -*-
"""
双模型 SBERT 全量 embedding + 分块断点续传。

读取 sample_A_30k.csv → MiniLM / text2vec 各一份向量 + comment_ids.npy

用法:
  python 11_sbert_embed_dual_models.py --csv data/sample_A_30k.csv --outdir embeddings
  python 11_sbert_embed_dual_models.py --only minilm --batch-size 128
  python 11_sbert_embed_dual_models.py --chunk-size 3000   # 默认 3000，约 10 块/3万条

中断后原命令重跑会跳过已存在的 chunk，最后合并为 embeddings_*.npy。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CSV = SCRIPT_DIR / "data" / "sample_A_30k.csv"

MODEL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_TEXT2VEC = "shibing624/text2vec-base-chinese"

SLUG = {"minilm": "minilm", "text2vec": "text2vec"}


def load_texts_and_ids(
    csv_path: Path, text_col: str
) -> tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    id_col = "comment_id" if "comment_id" in df.columns else None
    if id_col is None:
        raise SystemExit("CSV 需包含 comment_id 列")

    if text_col not in df.columns:
        if "content" in df.columns:
            text_col = "content"
        else:
            raise SystemExit(f"无文本列，现有: {list(df.columns)}")

    texts = df[text_col].fillna("").astype(str).tolist()
    ids = df[id_col].astype(str).to_numpy()
    return texts, ids


def run_model_chunks(
    model_name: str,
    slug_key: str,
    texts: List[str],
    outdir: Path,
    chunk_size: int,
    batch_size: int,
    merge_only: bool,
) -> Path:
    chunk_dir = outdir / f"chunks_{slug_key}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_chunks = (n + chunk_size - 1) // chunk_size
    final_path = outdir / f"embeddings_{slug_key}.npy"
    all_done = all(
        (chunk_dir / f"chunk_{k}.npy").is_file() for k in range(n_chunks)
    )

    if merge_only:
        print(f"\n[{slug_key}] 仅合并 chunk → {final_path.name}")
    elif all_done:
        print(
            f"\n[{slug_key}] 全部 {n_chunks} 个 chunk 已存在，"
            f"跳过加载模型，直接合并 → {final_path.name}"
        )
    else:
        print(f"\n[{slug_key}] 模型: {model_name}")
        model = SentenceTransformer(model_name)
        for k in range(n_chunks):
            chunk_path = chunk_dir / f"chunk_{k}.npy"
            if chunk_path.is_file():
                print(f"  跳过已存在 chunk {k}/{n_chunks}")
                continue
            lo = k * chunk_size
            hi = min(lo + chunk_size, n)
            batch = texts[lo:hi]
            t0 = time.time()
            emb = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            np.save(chunk_path, emb.astype(np.float32))
            print(
                f"  chunk {k}: 行 {lo}-{hi - 1}, "
                f"shape {emb.shape}, {time.time() - t0:.1f}s"
            )

    # 合并
    parts = []
    for k in range(n_chunks):
        p = chunk_dir / f"chunk_{k}.npy"
        if not p.is_file():
            raise SystemExit(
                f"缺少 {p}，请先完成该模型各 chunk 编码（勿加 --merge-only）"
            )
        parts.append(np.load(p))
    full = np.vstack(parts)
    if full.shape[0] != n:
        raise SystemExit(f"合并行数 {full.shape[0]} != 文本数 {n}")
    np.save(final_path, full)
    size_mb = final_path.stat().st_size / (1024 * 1024)
    print(f"[{slug_key}] 已写 {final_path}  shape={full.shape}  ({size_mb:.1f} MB)")
    return final_path


def main() -> None:
    ap = argparse.ArgumentParser(description="双模型 SBERT embedding + 分块续传")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "embeddings")
    ap.add_argument("--text-col", default="content_clean")
    ap.add_argument("--chunk-size", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--only",
        choices=("both", "minilm", "text2vec"),
        default="both",
    )
    ap.add_argument(
        "--merge-only",
        action="store_true",
        help="仅合并已有 chunk，不加载模型编码",
    )
    ap.add_argument(
        "--skip-ids",
        action="store_true",
        help="不写入 comment_ids.npy（已存在时）",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"找不到: {args.csv}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    ids_path = args.outdir / "comment_ids.npy"

    texts, ids = load_texts_and_ids(args.csv, args.text_col)
    if not (args.skip_ids and ids_path.is_file()):
        np.save(ids_path, ids)
        print(f"已写 {ids_path}  n={len(ids)}")
    else:
        print(f"保留已有 {ids_path}")

    t_all = time.time()
    if args.only in ("both", "minilm"):
        run_model_chunks(
            MODEL_MINILM,
            SLUG["minilm"],
            texts,
            args.outdir,
            args.chunk_size,
            args.batch_size,
            args.merge_only,
        )
    if args.only in ("both", "text2vec"):
        run_model_chunks(
            MODEL_TEXT2VEC,
            SLUG["text2vec"],
            texts,
            args.outdir,
            args.chunk_size,
            args.batch_size,
            args.merge_only,
        )

    print(f"\n总耗时: {(time.time() - t_all) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
