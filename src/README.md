# src/ — Analysis Pipeline

Reorganized from the repository root on 2026-07-05 (`git mv`, history preserved).
Scripts are grouped by pipeline stage:

| Directory | Scripts | Stage |
|---|---|---|
| `crawl/` | `01_bilibili_crawler_config.py`, `01_zhihu_crawler_config.py`, `02_run_zhihu_crawler.py`, `03_run_bilibili_crawler.py`, `04_bilibili_subtitle_and_transcript.py`, `steam_spider.py` | Data collection |
| `clean/` | `clean_bilibili_comments.py`, `dedup_bilibili_data.py`, `05_time_detect.py`, `06_time_analysis.py`, `07_time_analysis.py` | Cleaning and time-period diagnostics |
| `sample/` | `08_data_sampling_pipeline.py` | Stratified sampling (Samples A/B) |
| `annotate/` | `09_openai_batch_annotate.py` | GPT batch annotation |
| `embed/` | `10_sbert_smoke_test.py`, `11_sbert_embed_dual_models.py` | SBERT embeddings |
| `cluster/` | `12_clustering_sweep.py` | Clustering sweep |
| `finetune/` | `13_finetune_classifier.py` | Classifier fine-tuning |
| `analyze/` | `15_rq1_analysis.py`, `16_rq2_analysis.py`, `18_visualizations.py` | RQ analyses and figures |

## Path convention

Every script resolves data paths from the **repository root**
(`Path(__file__).resolve().parents[2]`), so they can be run from any working directory:

```bash
python3 src/embed/11_sbert_embed_dual_models.py --batch-size 128
python3 src/embed/10_sbert_smoke_test.py --nrows 5
```

Data directories (`data/`, `01_data/`, `00_output/`, `embeddings/`, `results/`) stay at
the repository root; raw data is not tracked in the public repository.

## Notes

- `crawl/02_*.py` and `crawl/03_*.py` **execute on import** (they chdir into
  MediaCrawler and launch the crawler) — do not run them casually; their config files
  (`01_*_config.py`) live in the same directory.
- Direct HuggingFace downloads of large files may stall on some networks; use a mirror:
  `HF_ENDPOINT=https://hf-mirror.com python3 src/embed/11_...`
  If the Python client still fails, download the weights manually with curl into
  `~/.cache/local_models/`.
- Environment: system `python3` (3.11) + `requirements.lock.txt` (full `pip freeze`).
- Script numbers `14_` and `17_` never existed in the pipeline — they are not missing.
