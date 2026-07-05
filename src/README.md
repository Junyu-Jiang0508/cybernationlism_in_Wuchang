# src/ — 分析管线脚本

2026-07-05 从仓库根目录重组而来（`git mv`，历史可追溯）。按管线阶段分目录：

| 目录 | 脚本 | 阶段 |
|---|---|---|
| `crawl/` | `01_bilibili_crawler_config.py`、`01_zhihu_crawler_config.py`、`02_run_zhihu_crawler.py`、`03_run_bilibili_crawler.py`、`04_bilibili_subtitle_and_transcript.py`、`steam_spider.py` | 数据采集 |
| `clean/` | `clean_bilibili_comments.py`、`dedup_bilibili_data.py`、`05_time_detect.py`、`06_time_analysis.py`、`07_time_analysis.py` | 清洗与时段诊断 |
| `sample/` | `08_data_sampling_pipeline.py` | 分层抽样（Sample A/B） |
| `annotate/` | `09_openai_batch_annotate.py` | GPT 批量标注 |
| `embed/` | `10_sbert_smoke_test.py`、`11_sbert_embed_dual_models.py` | SBERT 嵌入 |
| `cluster/` | `12_clustering_sweep.py` | 聚类扫描 |
| `finetune/` | `13_finetune_classifier.py` | 分类器微调 |
| `analyze/` | `15_rq1_analysis.py`、`16_rq2_analysis.py`、`18_visualizations.py` | RQ 分析与可视化 |

## 路径约定

所有脚本以**仓库根目录**为数据路径基准（`Path(__file__).resolve().parents[2]`），
因此**从任何工作目录运行都可以**：

```bash
python3 src/embed/11_sbert_embed_dual_models.py --batch-size 128
python3 src/embed/10_sbert_smoke_test.py --nrows 5
```

数据目录（`data/`、`01_data/`、`00_output/`、`embeddings/`、`results/`）仍在仓库根，未移动。

## 注意事项

- `crawl/02_*.py` 与 `crawl/03_*.py` **导入即执行**（会 chdir 到 MediaCrawler 并启动爬虫），
  不要在无意图时运行；其配置文件（`01_*_config.py`）与启动脚本同目录。
- HuggingFace 直连下载可能卡死；国内网络用镜像：
  `HF_ENDPOINT=https://hf-mirror.com python3 src/embed/11_...`
- 环境：系统 `python3`（3.11）+ `requirements.lock.txt`；仓库内 `venv/` 为
  Windows 结构，WSL 下不可用（见 `docs/DATA_STATEMENT.md` §四）。
- `14_` 与 `17_` 编号在原管线中即不存在，非丢失。
