# Scripted Rage as Commodity — Chinese Cyber-Nationalism in the *Wuchang* Controversy

Computational communication research on how Chinese cyber-nationalist discourse is
produced, circulated, and rewarded across platforms, using the 2025 controversy around
the game *WUCHANG: Fallen Feathers*（《明末：渊虚之羽》）as the focal case.

**Status（2026-07）**: AEJMC 2026 Research-in-Progress accepted（CTAM division,
reviewer average 35.67/40, RiP-05-7669）; presenting August 5–8, New Orleans.
Full-paper development underway — see `docs/RESEARCH_PLAN_2026H2.md`.

## 研究概要

- **理论框架**：Swidler 文化工具箱 → 民族主义话语作为**模块化剧目**（modular repertoire）
- **管线**：MediaCrawler/自研爬虫 → 清洗去重 → 分层抽样 → GPT-4o-mini 批量标注（+人工校验 κ=0.74/0.81）
  → RoBERTa-wwm-ext 微调（binary F1=0.82）→ SBERT 双模型嵌入 → 聚类与 n-gram 模板分析
- **语料**：Bilibili 62.3 万清洗评论 · Zhihu 4.2 万评论 + 398 篇长文 · Steam 3.3 万简中评测

## 仓库结构

```text
├── README.md                ← 本文件
├── requirements*.txt/.lock  ← 依赖（实际环境见 docs/DATA_STATEMENT.md §四）
├── src/                     ← 分析管线（crawl→clean→sample→annotate→embed→cluster→finetune→analyze）
│   └── README.md            ← 脚本目录映射与运行约定
├── docs/                    ← 研究文档
│   ├── RESEARCH_PLAN_2026H2.md      ← 执行层推进计划（评审意见→行动）
│   ├── RESEARCH_DESIGN_GUIDE.md     ← 设计层指南（RQ/理论/标注/平台扩展）
│   └── DATA_STATEMENT.md            ← 数据声明（口径、审查限制、伦理、环境与复现）
├── 00_output/01_Raw_data/   ← 原始数据（01_Zhihu / 02_Bilibili / 03_Steam）
├── 01_data/                 ← 清洗后数据与时段诊断
├── data/                    ← 抽样样本与 GPT 标注存档（Sample A/B）
├── embeddings/              ← SBERT 向量（gitignored，可用 src/embed/11 重新生成）
├── results/                 ← 分析产物（微调模型、聚类标签、图表）
├── Document/                ← 论文版本与投稿材料
│   ├── AEJMC_RIP/           ← RIP 投稿（LaTeX + reviewer_feedback/ 评审意见）
│   ├── COGS181_AAAI_report/ ← 课程期末版（AAAI 格式）
│   └── ...
├── archive/                 ← 已归档（COGS181 课程页面、LaTeX 构建产物等）
└── MediaCrawler-main/       ← 第三方爬虫（依赖）
```

## 快速开始

```bash
pip install -r requirements.lock.txt          # 或按 requirements.txt 装最新版
python3 src/embed/10_sbert_smoke_test.py      # 冒烟测试
python3 src/embed/11_sbert_embed_dual_models.py --batch-size 128   # 重建嵌入
```

国内网络下 HuggingFace 模型下载见 `src/README.md` 的镜像说明。

## 相关文本

- 早期概念论文：`Document/Performing the Nation as Commodity.pdf`
- AEJMC RIP 提交稿：`Document/AEJMC_RIP/report.pdf`

## 作者

Junyu Jiang（UC Davis / UCSD）· Wenhan Xie（Communication University of China）
