# Scripted Rage as Commodity — Chinese Cyber-Nationalism in the *Wuchang* Controversy

Computational communication research on how Chinese cyber-nationalist discourse is
produced, circulated, and rewarded across platforms, using the 2025 controversy around
the video game *WUCHANG: Fallen Feathers* as the focal case.

**Status (July 2026)**: Accepted as a Research-in-Progress paper at AEJMC 2026
(CTAM division); presenting August 5–8 in New Orleans. Full-paper development underway.

## Overview

- **Theoretical framework**: Swidler's cultural toolkit — nationalist discourse as a
  *modular repertoire* of recyclable rhetorical moves rather than a coherent ideology.
- **Pipeline**: crawling (MediaCrawler / custom spiders) → cleaning and deduplication →
  stratified sampling → GPT-4o-mini batch annotation with human validation
  (human–human κ = 0.81, human–model κ = 0.74) → RoBERTa-wwm-ext fine-tuning
  (binary F1 = 0.82) → dual-model SBERT embedding → clustering and n-gram template analysis.
- **Corpora**: 623k cleaned Bilibili comments · 41.5k Zhihu comments + 398 long-form posts ·
  33.4k Simplified-Chinese Steam reviews.

## Repository layout

```text
├── README.md
├── requirements*.txt / requirements.lock.txt
├── src/                     ← analysis pipeline (crawl → clean → sample → annotate
│   │                          → embed → cluster → finetune → analyze)
│   └── README.md            ← script map and path conventions
├── results/                 ← aggregate outputs (figures, cluster labels, summaries)
├── Document/                ← paper versions and submission materials
│   ├── AEJMC_RIP/           ← AEJMC Research-in-Progress submission (LaTeX)
│   └── COGS181_AAAI_report/ ← earlier course report (AAAI format)
├── archive/                 ← superseded materials (indexed in archive/README.md)
└── MediaCrawler-main/       ← third-party crawler (dependency)
```

Raw scraped data, annotation files, embeddings, and fine-tuned model weights are kept
out of the repository by design (privacy, research ethics, and file-size limits); the
pipeline in `src/` regenerates all derived artifacts from raw data.

## Quick start

```bash
pip install -r requirements.lock.txt
python3 src/embed/10_sbert_smoke_test.py                            # smoke test
python3 src/embed/11_sbert_embed_dual_models.py --batch-size 128    # rebuild embeddings
```

See `src/README.md` for the full script map and notes on mirror endpoints for
HuggingFace downloads.

## Related documents

- Early conceptual paper: `Document/Performing the Nation as Commodity.pdf`
- AEJMC RIP submission: `Document/AEJMC_RIP/report.pdf`

## Authors

Junyu Jiang (UC Davis / UCSD) · Wenhan Xie (Communication University of China)
