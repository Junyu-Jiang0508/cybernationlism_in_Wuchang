# 数据声明（Data Statement）

> 用途：全文版论文附录底稿 + 复现性审计依据。所有数字为 2026-07-05
> 对仓库内实际文件重新核算的结果；与论文/提交稿中数字不一致处均在 §五 明确标注。
> 事件背景：《明末：渊虚之羽》（WUCHANG: Fallen Feathers）2025-07-23 发售引发的民族主义争议。

---

## 一、语料清单

### 1. Bilibili 评论（主语料，Phase 1 已分析）

| 项 | 值 |
|---|---|
| 采集工具 | MediaCrawler（仓库内 `MediaCrawler-main/`），启动脚本 `src/crawl/03_run_bilibili_crawler.py` |
| 采集方式 | 关键词搜索：**"明末渊虚之羽"**，遍历命中视频的评论区（含二级评论） |
| 原始规模 | 1,508,478 条评论（提交稿口径；原始 CSV 约 174 万行，含引号内换行） |
| 清洗后 | **623,396 条**，覆盖 **1,079 个有评论的视频**（见 §五 口径差异 1） |
| 时间窗口 | 2025-07-23 ~ 2025-12-31（北京时间；窗口外评论在清洗时过滤） |
| 清洗流程 | `src/clean/clean_bilibili_comments.py`：comment_id 全局去重 → 时间过滤 → 内容过滤（len≥阈值）→ 文本标准化（OpenCC 繁→简）→ 机器人标记 → 关键词筛选 → 三路输出 |
| 结构快照 | 一级评论 179,676（28.8%）；like_count=0 占 46.71%；2025-07-24~08-31 高峰期占 84.7% |
| 存储 | `01_data/comments_clean.parquet`（分析用）、`01_data/search_comments_all.csv`（原始） |

**时段划分**（`01_data/time_diagnose/period_boundaries.json`，当前版本）：
- 事件窗口：2025-07-23 00:00 ~ 2025-12-31 23:59（+08:00）
- burst（爆发期）结束：2025-08-19；transition（过渡期）结束：2025-10-14

### 2. Bilibili 衍生样本

| 样本 | 规模 | 构造 | 文件 |
|---|---|---|---|
| Sample A（嵌入/聚类/推理） | 30,000 | 分层抽样：一级评论、content_clean 长度≥5（eligible=169,058→时段内 158,262）；early 池 129,977 抽 15,000 / mid 池 23,165 抽 9,000 / late 池 5,120 抽 6,000；合并去重 29,120 后随机补足至 30,000；Top10 视频评论占比 16.9%，未触发视频上限截断 | `data/sample_A_30k.csv` |
| Sample B（标注/训练） | 2,000 | 1,200 条来自关键词池（宽口径疑似池 3,577）+ 800 条随机取自 Sample A。**注意**：关键词池过采样导致训练集 Type 2 占 38.3% vs 无偏子集（random_from_A，n=743）中的 15.2% | `data/sample_B_2000.csv` |
| 抽样全记录 | — | 详细数字与随机种子过程 | `data/sampling_log.txt` |

### 3. Bilibili 创作者与视频元数据

- `01_data/search_videos_all.csv`（视频元数据，提交稿口径 13,124 个搜索命中视频）
- `01_data/search_creators_all.csv`（创作者元数据：粉丝、账号类型等，Phase 3 创作者层分析备用）
- 视频音频转写：`00_output/01_Raw_data/02_Bilibili/transcripts/`（8 个视频，
  `src/crawl/04_bilibili_subtitle_and_transcript.py` 生成，定性佐证用）

### 4. Zhihu（Phase 2 待分析）

| 项 | 值 |
|---|---|
| 采集工具 | MediaCrawler，启动脚本 `src/crawl/02_run_zhihu_crawler.py`，关键词 **"明末渊虚之羽"** |
| 规模 | **41,518 条评论** + **398 篇内容**（回答/文章，含全文与问题元数据） |
| 时间范围 | 评论 publish_time 跨 **约 2025-01-20 ~ 2026-03-14**——**早于事件窗口**：关键词命中了发售前的预热/前瞻讨论。Phase 2 分析时需决定是否截断至事件窗口，或把"事件前话语"作为基线利用（设计机会，见 RESEARCH_DESIGN_GUIDE §一时间维度） |
| 特有字段 | `ip_location`（属地标识）、`dislike_count`——B 站语料没有的维度 |
| 存储 | `00_output/01_Raw_data/01_Zhihu/csv/` |

### 5. Steam（Phase 2 待分析）

| 项 | 值 |
|---|---|
| 采集工具 | 自研 `src/crawl/steam_spider.py`，官方 appreviews API（store.steampowered.com） |
| 目标 | App ID 2277560（WUCHANG: Fallen Feathers） |
| 过滤 | `language=schinese`（仅简中评测）、`filter=recent`、请求间隔 0.35s |
| 规模 | **33,415 条评测**（API 当时报告总数 33,426，覆盖率 99.97%） |
| 时间范围 | 2025-07-24 ~ 2026-03-17（UTC）——**比 B 站窗口长约 2.5 个月** |
| 特有字段 | 是否推荐（好评率 51.6%）、游玩时长、评测点赞——支持"话语 vs 实际游玩行为"交叉分析 |
| 存储 | `00_output/01_Raw_data/03_Steam/steam_reviews.csv`（2026-07-05 从根目录归位，爬虫输出路径已同步更新） |

### 6. GPT 标注数据（2026-03 生成，不可重新调用）

| 项 | 值 |
|---|---|
| 标注模型 | gpt-4o-mini，OpenAI Batch API，temperature=0（2026-03 执行） |
| 维度 | discourse_type（0-4, 99）、othering_intensity（0-3）、affect_intensity（0-3） |
| 人工校验 | 200 条分层子样本双人独立编码：人–人 κ=0.81；人–GPT κ=0.74（最弱边界：Type 1 κ=0.61 / Type 2 κ=0.63） |
| 存档 | `data/batch_input.jsonl`、`data/batch_output.jsonl`、`data/annotations_raw.json`、`data/annotations_merged.csv`。**下游分析一律从存档出发，不重新调用 API**（闭源模型不可复现，见 §四） |

---

## 二、审查与可观测性限制（censorship-aware 声明）

所有跨平台对比 **conditional on surviving text**——以下过程使观测语料 ≠ 潜在话语总体：

1. **删除不可观测**：三个国内可见平台（B 站、知乎）的平台删帖、用户自删、
   账号封禁后的内容消失均发生在采集之前或之间，无法从单次采集中识别；
   本仓库数据为**单次采集快照**，无时间点差分。
2. **平台特有机制**：B 站评论区折叠/关闭（部分高争议视频评论区被关闭，
   这些视频的评论只有关闭前的部分）；知乎回答折叠与"建议修改"隐藏；
   Steam 无此类政治性审查，但有"离题评测"（review bomb）标记机制，
   API 的 `filter=recent` 不受其影响。
3. **采集侧缺口**：B 站二级评论受 API 翻页深度限制，超长评论树尾部可能缺失；
   知乎关键词搜索只触及被搜索索引的内容，被限流/降权内容系统性缺席。
4. **已知节制事件记录（待补全）**：全文版附录需要一张"事件时间线 × 可观测节制行为"
   表（游戏版本更新公告、制作组声明、平台热搜撤除报道等），来源：新闻存档 +
   视频转写。当前状态：**未系统整理**——列为 Phase 2 任务。

---

## 三、伦理与合规

- 仅采集公开可见内容；未采集私信、未注册登录态之外的受限内容。
- 分析与发表以聚合形式呈现；引用具体评论文本时改写脱敏，不保留可回溯的
  用户名/UID 组合。创作者层分析（Phase 3）只报告聚合统计。
- `bilibili_cookie.txt` 与 `.env` 含账号凭据，均已加入 `.gitignore`。
  经核验，凭据**从未进入公开远端历史**（含凭据的本地提交未曾推送，
  已隔离在本地备份分支，勿推送该分支）。
- IRB：公开数据二手分析，UCSD 豁免类别待确认（Phase 2 任务）。

---

## 四、复现入口与运行环境

```
依赖锁定:  requirements.lock.txt（pip freeze 全量，220 包）
管线脚本:  src/{crawl,clean,sample,annotate,embed,cluster,finetune,analyze}/
           （运行位置无关：脚本内部以仓库根为基准解析路径）
嵌入向量:  python3 src/embed/11_sbert_embed_dual_models.py --batch-size 128
           （输出 embeddings/embeddings_{minilm,text2vec}.npy + comment_ids.npy）
分析产物:  results/（2026-03 原始存档 + 2026-07 重跑）
```

**运行环境（2026-07-05 记录）**

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4060 Laptop (8 GB)，驱动 595.79，CUDA 13.0 |
| 系统 | Windows 11 + WSL2（内核 6.6.114.1） |
| Python | 3.11.15（系统 python3） |
| 关键包 | torch 2.11.0+cu130 · transformers 5.5.0 · sentence-transformers 5.3.0 · pandas 3.0.2 · numpy 2.4.4 · scikit-learn 1.8.0 · statsmodels 0.14.6 |

**复现性警示与核对**

- 2026-03 原始分析环境未锁版本；本记录为 2026-07-05 重建环境。重跑结果与
  3 月存档有微小差异时，首先怀疑 transformers / sentence-transformers 大版本变化。
- **复现性核对（MiniLM）**：重建向量经 PCA(50, seed=42)+KMeans(k=10, seed=42) 与
  2026-03 存档标签对比：**NMI=0.980，ARI=0.988**——版本漂移影响可忽略，
  3 月聚类结论在新环境下成立。
- GPT-4o-mini 标注为闭源 API 调用，**不可精确复现**；下游分析一律从
  `data/` 存档出发（见 §一.6）。
- HuggingFace 直连大文件下载在本网络易卡死：用 `HF_ENDPOINT=https://hf-mirror.com`，
  Python 客户端仍失败时用 curl 手动下载到 `~/.cache/local_models/`（详见 `src/README.md`）。

---

## 五、口径差异与待办（诚实记录，勿删）

1. **"13,124 视频" vs "1,079 视频"**：提交稿称评论跨 13,124 个视频——这是
   **搜索命中的视频元数据总数**（`search_videos_all.csv`）；清洗后评论语料
   实际覆盖 **1,079 个有评论的视频**（`data/sampling_log.txt`）。全文版必须
   统一口径：建议表述为"搜索命中 13,124 个视频，其中 1,079 个视频的评论区
   贡献了清洗后语料"。**在改稿前需重新核实两个数字的确切定义。**
2. **一级评论占比**：提交稿/课程报告底稿（本地存档）称 33.4%（318,044），sampling_log
   记录 28.8%（179,676）——前者疑为原始语料口径、后者为清洗后口径，待核实。
3. **凭据文件**：~~`bilibili_cookie.txt` 被 git 跟踪~~ **已处理**（2026-07-05
   脱离跟踪并加入 `.gitignore`；核验确认从未推送到公开远端）。
4. **Zhihu/Steam 时间窗口与 B 站不对齐**（知乎起点早 6 个月，Steam 终点晚 2.5 个月）：
   跨平台对比时需显式选择统一窗口或论证不对齐的合理性。
5. **`venv/` 为 Windows 结构**——**已处理**（2026-07-05 确认后删除，省 1.1GB）；
   实际环境见 §四。
