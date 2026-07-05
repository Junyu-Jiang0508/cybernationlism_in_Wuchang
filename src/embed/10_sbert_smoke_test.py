# -*- coding: utf-8 -*-
"""
SBERT 冒烟测试：10 条文本 + MiniLM，估算全量耗时。
用法:
  python 10_sbert_smoke_test.py
  python 10_sbert_smoke_test.py --csv data/sample_A_30k.csv --nrows 10 --batch-size 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CSV = SCRIPT_DIR / "data" / "sample_A_30k.csv"
MODEL_MINILM = "paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    ap = argparse.ArgumentParser(description="SBERT 冒烟测试")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="样本 CSV 路径")
    ap.add_argument("--nrows", type=int, default=10, help="读取行数")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--text-col",
        default="content_clean",
        help="文本列名（若无则尝试 content）",
    )
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"找不到 CSV: {args.csv}")

    df = pd.read_csv(args.csv, encoding="utf-8-sig", nrows=args.nrows)
    col = args.text_col
    if col not in df.columns:
        if "content" in df.columns:
            col = "content"
        else:
            raise SystemExit(f"CSV 无列 {args.text_col!r}，现有列: {list(df.columns)}")

    texts = df[col].fillna("").astype(str).tolist()
    print(f"测试文本数: {len(texts)}")
    if texts:
        print(f"示例: {texts[0][:80]}...")

    print(f"加载模型: {MODEL_MINILM} ...")
    model = SentenceTransformer(MODEL_MINILM)

    start = time.time()
    embeddings = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True
    )
    elapsed = time.time() - start

    print(f"Embedding 维度: {embeddings.shape}")
    print(f"{len(texts)} 条耗时: {elapsed:.2f}s")
    if len(texts) > 0:
        est_min = elapsed / len(texts) * 30_000 / 60
        print(f"估算 3 万条耗时: 约 {est_min:.1f} 分钟")
    print("✓ 冒烟测试通过")


if __name__ == "__main__":
    main()
