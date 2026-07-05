# 研究推进计划指南（2026 下半年）

## Scripted Rage as Commodity — 从 AEJMC RIP 到完整论文

> 依据：AEJMC 评审意见（RiP-05-7669，均分 35.67/40）、提交稿全文、本仓库现有数据与代码。
> 撰写日期：2026-07-05。AEJMC 年会：2026-08-05 ~ 08-08（新奥尔良）。

---

## 一、现状盘点

### 1.1 评审结果解读

| 评分项           | 均分             | 信号                            |
| ---------------- | ---------------- | ------------------------------- |
| 1 目的清晰       | 4.33             | 良好                            |
| 2 文献           | 4.33             | 良好                            |
| 3 概念论证       | 4.67             | 强项                            |
| 4 方法           | 4.33（R3 给 3）  | R3 对方法有具体保留意见，见 §二 |
| 5 契合 CTAM      | 4.67             | 投对了分会                      |
| 6 研究方向前景   | 4.67             | 强项                            |
| 7**表达与组织**  | **3.67（最低）** | 全文写作是最大短板              |
| 8 进展与反馈需求 | 5.00             | 满分，RIP 格式运用得当          |

**结论**：理论框架（Swidler 工具箱 → 模块化剧目）和研究方向被三位评审一致认可；短板集中在
(a) 写作表达（评分项 7），(b) 方法层面的三个技术缺口（R3 详列），(c) 核心概念
"modularity" 解释不足与案例背景缺失（R2，受 RIP 篇幅所限）。

### 1.2 数据资产（已到手，无需再爬）

| 语料                  | 规模                                           | 位置                                                    | 状态                       |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------- | -------------------------- |
| Bilibili 评论         | 1,508,478 条原始 / 623,396 清洗后              | `01_data/comments_clean.parquet`                        | Phase 1 已完成分析         |
| Sample A              | 30,000 条（已推理预测）                        | `results/finetune/sample_A_predicted.csv`               | 已有预测标签               |
| Sample B              | 2,000 条（GPT 标注 + 200 条双人工校验 κ=0.74） | `data/sample_B_2000.csv`, `data/annotations_merged.csv` | 训练集                     |
| Bilibili 创作者元数据 | —                                              | `01_data/search_creators_all.csv`                       | Phase 3 备用               |
| 视频字幕/转写         | 8+ 视频                                        | `00_output/01_Raw_data/02_Bilibili/transcripts/`        | 可做定性佐证               |
| **Zhihu**             | 41,538 条评论 + 398 篇内容                     | `00_output/01_Raw_data/01_Zhihu/csv/`                   | **待分析（Phase 2 核心）** |
| **Steam**             | 33,415 条评测（含推荐标记、游玩时长、点赞） | `00_output/01_Raw_data/03_Steam/steam_reviews.csv`                                     | **待分析（Phase 2 核心）** |

### 1.3 代码资产（注意：需从 git 恢复）

编号脚本 01–18（爬虫→清洗→抽样→GPT 标注→SBERT→聚类→微调→RQ1/RQ2→可视化）
在当前工作区已被删除，但**完整保留在 git HEAD 中**，随时可恢复：

```bash
# 恢复全部管线脚本（不会覆盖未删除的文件）
git checkout HEAD -- '*.py'
# 或单个恢复
git checkout HEAD -- 13_finetune_classifier.py 16_rq2_analysis.py
```

微调最优模型权重在 `results/finetune/best_model/`；`embeddings/` 目录为空，
SBERT 向量需用 `11_sbert_embed_dual_models.py` 重新生成（约 30k×2 模型，GPU 数小时内）。

---

## 二、评审意见 → 行动映射表

| #    | 评审意见                                                                     | 行动                                                                                                    | 阶段 |
| ---- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---- |
| R3-1 | Type 1/2 边界混淆（五分类 macro-F1=0.62 vs 二分类 0.82）→ 改**两步层级分类** | 先政治/非政治，再在政治子集内细分；对比层级 vs 五路 softmax                                             | P1   |
| R3-2 | 过采样导致决策边界偏移 →**Matrix Calibration** + 校准前后曲线                | 温度/向量/矩阵缩放 + Platt 对照；报告 ECE 与 reliability diagram；用 random_from_A（n=743）做无偏验证集 | P1   |
| R3-3 | RQ2 用**负二项 GLMM 或 IHS 变换**，而非 log 变换 OLS                         | 随机截距（视频层）负二项混合模型；IHS 做稳健性检验                                                      | P2   |
| R3-4 | Zhihu 长文本超 512 token →**Longformer 或滑窗+均值池化**                     | 主方案滑窗+池化（保留 RoBERTa 骨干、跨平台可比），Longformer 做稳健性                                   | P3   |
| R2   | modularity 概念解释不足；缺案例背景                                          | 全文版扩写理论章节（Swidler + Tilly repertoire + meme/template 文献）；增设《渊虚之羽》事件时间线小节   | P4   |
| R1   | 给非量化读者加"一句话数字总结"                                               | 每个结果段落首句用 plain-language 概括；这直接回应评分项 7（3.67 最低分）                               | P4   |
| R1   | LLM 标注标准、跨平台验证设计值得会上讨论                                     | AEJMC 会上主动向 CTAM 同行征询这两点（提交稿"Areas Needing Feedback"的延续）                            | P0   |

---

## 三、分阶段执行计划

### Phase 0（7/5 – 8/8）：仓库整备 + AEJMC 会议准备

**目标：能复现、能展示。**

1. **仓库恢复与重组**（1–2 天）
   - `git checkout HEAD -- '*.py'` 恢复管线脚本；建议重组为
     `src/{crawl,clean,sample,annotate,embed,cluster,finetune,analyze}/`；
   - 固定环境：`pip freeze > requirements.lock.txt`；记录 GPU/CUDA 版本；
   - 补一份 `DATA_STATEMENT.md`：各语料采集窗口、采集工具、清洗规则、已知审查事件
     （提交稿"censorship-aware"承诺的落实，也是全文版必备附录）。
2. **重新生成 SBERT 向量**（`embeddings/` 为空，恢复 `11_` 脚本后重跑）。

### Phase 1（7 月中 – 8 月底）：分类器升级与校准（回应 R3-1、R3-2）

**目标：把 macro-F1=0.62 的五分类问题拆解掉，把 prevalence 估计做成无偏的。**

1. **两步层级分类**
   - Stage A：政治（type 2+3）vs 非政治（0/1/4）——现有二分类 F1=0.82 就是这一步的基线；
   - Stage B：政治子集内 nationalist(3) vs politicized-criticism(2)；
   - 同时试另一种切法（{2,3,4} 实质性 vs {0,1} 非实质性 → 三分）做对照；
   - 评价指标：hierarchical F1 + 与五路 softmax 的逐类对比；预期 Type 1/2 边界错误
     被隔离在 Stage A 内部，Stage B 的类内区分度上升。
   - 可选增强：把 othering_intensity、affect_intensity 作为辅助任务做 multi-task
     fine-tuning（标注已有，零额外成本）。
2. **校准与患病率（prevalence）估计**
   - 验证集：`data/annotations_merged.csv` 中 random_from_A 子集（n=743，无偏分布）；
   - 方法阶梯：temperature scaling → vector scaling → **matrix scaling**（R3 点名）
     → Platt per-class 对照；报告 ECE、Brier score、reliability diagram（前/后）；
   - **关键升级**：估计语料构成本质上是 _quantification_ 问题，比逐条分类更稳健的做法是
     Adjusted Classify & Count / BBSE（Lipton et al. 2018）或 Hopkins & King (2010)
     readme 方法——communication 领域评审对 Hopkins & King 非常熟悉，引用它能加分；
   - 训练侧修正：已知抽样设计（1200 关键词池 + 800 随机），可按逆抽样概率加权损失重训一版。
3. **边界定性错误分析**（提交稿承诺的 Phase 2 内容）
   - 从混淆矩阵抽 Type1↔2 双向错分各 50 条，两位编码者独立归因：
     标注噪声 / 真实语义模糊 / 模型容量不足；产出一个错误类型学小节——
     这会成为全文版方法讨论中最有说服力的部分。
4. **扩充人工校验**：现有 200 条 κ=0.74；建议在边界层加 100–150 条定向双编码，
   把"最难边界"的 κ 单独报告（回应 LLM 标注标准的讨论）。

### Phase 2（8 月中 – 9 月中）：RQ2 参与度建模（回应 R3-3）

**目标：把被搁置的 RQ2 用站得住的模型重新做出来。**

1. **模型规格**（建议用 R 的 `glmmTMB`；Python 侧 `statsmodels` 无 NB-GLMM）
   - 主模型：`likes ~ discourse_type + comment_covars + (1 | video_id)`，nbinom2 家族；
   - 空模型先行，报告 ICC（预期很高——DW=0.14 已经暗示了）；
   - 零膨胀检验：NB vs ZINB，AIC/BIC 比较；
   - 稳健性：IHS 变换 + `lme4` 线性混合模型对照（R3 给的两条路都走）；
   - 视频层协变量：视频播放量、UP 主粉丝数、视频发布距评论时差；
     评论层：文本长度、是否一级评论、发布时段。
2. **推断注意事项**：type 2/3/4 之间的对比是核心检验
   （"分析性包装的民族主义内容是否获得超额参与"），用 marginal effects +
   多重比较校正呈现；预期结论仍是"算法奖励实质性内容而非意识形态方向"，
   但这次是可辩护的版本。
3. 若想加分：把"预测标签的测量误差"传导进回归（多重插补：按校准后的类别概率
   抽样标签重复估计），直接回应"用预测变量做回归"的经典批评（communication
   方法圈近年热点，CTAM 评审会认）。

### Phase 3（9 月 – 10 月中）：跨平台验证（提交稿 Phase 2 承诺）

**目标：RQ1 的可推广性检验——模块化是话语属性还是平台伪影。**

1. **Zhihu 轨道**（数据已备：41.5k 评论 + 398 长文）
   - 先做承诺过的 codebook 迁移验证：分层抽 200–300 条，双人独立编码，
     目标 κ ≥ 0.75 再放量标注；
   - 长文本（R3-4）：主方案 = 512-token 滑窗（stride 256）+ mean-pooling，
     保持 RoBERTa-wwm-ext 骨干不变以保证跨平台几何可比；
     稳健性 = 中文 Longformer（如 Erlangshen-Longformer）对照一版；
   - 报告尾部信息损失检验：只用前 512 token vs 滑窗全文的标签一致率。
2. **Steam 轨道**（数据已备：33.4k 评测）
   - XLM-RoBERTa-base 微调（提交稿已承诺，避免语言检测级联）；
   - 需一轮 Steam 专用标注（同样 200–300 条双编码——Steam 短文本 + 中英夹杂，
     codebook 需增补 code-switching 判例）；
   - Steam 独有优势：有"是否推荐 + 游玩时长 + 点赞"元数据，可做
     "民族主义话语 vs 实际游玩行为"的独特交叉分析（其他平台做不了，是亮点）。
3. **跨平台几何比较**（RQ1 主检验）
   - 统一指标：intra-class cosine（相对差值版，对校准偏移不敏感）、
     top-500 n-gram Jaccard、模板句跨账号复现率；
   - 审查敏感呈现：每个平台单列采集窗口、已知删帖事件、可观测的缺口率
     （Bilibili 楼中楼缺失、Zhihu 折叠等），所有跨平台对比都注明
     "conditional on surviving text"。
4. **模板迁移检测**（新增，成本低、故事强）：
   - 把 Bilibili 上发现的高频模板句（如"汉族一样尊重少数民族"句式）在
     Zhihu/Steam 语料中做模糊匹配（编辑距离/嵌入近邻），
     直接量化"同一剧目跨平台传播"——这是对 modular repertoire 最直观的证据。

### Phase 4（10 月中 – 11 月初，视目标可延至 2027 春）：创作者层 + 全文写作

1. **创作者层分析**（提交稿 Phase 3，可做可缓）
   - 用 `search_creators_all.csv`：高产民族主义账号 vs 高产非民族主义账号的
     词汇多样性（用 MTLD，别用 TTR——对长度敏感）、模板复用率、账号类型分布；
   - 若时间紧，此部分留给期刊版，会议版聚焦 P1–P3。
2. **全文写作**（起草后先做多轮内审再投稿）
   - 针对评分项 7（3.67）：每个结果小节首句 = 一句 plain-language 数字总结（R1 建议）；
   - 针对 R2：理论章节扩写 modularity（Swidler 1986 + Tilly 2008 repertoire +
     计算宣传的 templated messaging 文献），单列《渊虚之羽》事件背景小节（含时间线图，
     `results/figures/fig8_timeline.png` 可复用）；
   - 方法章节以"校准前后"与"层级 vs 扁平分类"的对比为叙事主轴——
     这正是三位评审都认可的"诚实呈现失败模式"风格的延续；
   - 投稿前做一次独立的可复现性审计，顺手产出复现包（OSF 上传）。
3. **合规与伦理**
   - 按 AEJMC/期刊要求准备 AI 使用披露声明；
   - 爬取数据伦理小节：只分析公开内容、聚合呈现、不引用可回溯到个人的原文
     （引用模板句时做改写脱敏）；确认 UCSD IRB 对二手公开数据的豁免类别；
   - 建议在 OSF 对 P2/P3 的验证性分析做轻量预注册（收集已完成不影响，
     注册的是分析决策——多层模型规格、κ 阈值、几何指标），CTAM 会买账。

---

## 四、投稿路线与时间线

| 时间                                                                                                             | 节点               | 交付物                                                                     |
| ---------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| 7/5 – 7/20                                                                                                       | P0 + P1 启动       | 仓库恢复、校准初步结果                                                     |
| 7/20 – 8/3                                                                                                       | 会议材料           | 海报 + 5 分钟讲稿（含校准前后图更佳）                                      |
| **8/5 – 8/8**                                                                                                    | **AEJMC 新奥尔良** | RIP 展示；记录 CTAM 反馈（LLM 标注标准、跨平台设计）                       |
| 8/10 – 9/15                                                                                                      | P1 收尾 + P2       | 层级分类器、校准报告、NB-GLMM 结果                                         |
| 9/15 – 10/15                                                                                                     | P3                 | Zhihu/Steam 验证、跨平台几何、模板迁移                                     |
| 10/15 – 11 月初                                                                                                  | 全文 v1            | **ICA 2027 投稿**（截稿历年为 11 月初，9 月起盯 icahdq.org 确认分会 call； |
| 建议投 Computational Methods 或 Political Communication 分会）                                                   |                    |                                                                            |
| 11 月 – 2027/3                                                                                                   | 扩展 + P4          | 创作者层分析并入期刊版                                                     |
| 2027/4/1                                                                                                         | 备选               | AEJMC 2027 完整论文（CTAM open competition）                               |
| 2027 上半年                                                                                                      | 期刊               | 首选*Computational Communication Research*（开放、方法友好）；             |
| 备选*New Media & Society* / _JCMC_ / _Chinese Journal of Communication_ / _Information, Communication & Society_ |                    |                                                                            |

**优先级裁剪原则**：若 ICA 截稿前时间不够，砍 P4（创作者层）和 Steam 的
XLM-R 微调（可先只报 Steam 的嵌入几何 + 模板迁移，把微调留给期刊版）；
P1 的校准和 P2 的多层模型**不可砍**——那是 R3 的两条硬意见，全文版评审大概率还会查。

---

## 五、风险与对策

| 风险                                              | 对策                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| 层级分类 Stage B 样本太少（type 3 仅 149 条标注） | 用校准后的高置信预测做半监督扩充，或对 type 3 增补 100–200 条定向标注      |
| Zhihu κ 达不到 0.75                               | 先修 codebook 再重测；κ 不达标就缩小声明范围（只报嵌入几何，不报分类结果） |
| GLMM 不收敛（视频数 13k+，极端不平衡嵌套）        | 换 nbinom1 家族 / 对每视频评论数设下限（≥5 条）做敏感性分析                |
| GPU 资源不足以重跑 embedding + 三平台微调         | MiniLM 向量 CPU 可跑；微调用 Colab/校内集群；XLM-R 只在 Steam 用 base 版   |
| ICA 截稿与 P3 撞车                                | 见上：优先级裁剪原则                                                       |

---

## 六、本周即可动手的三件事

1. `git checkout HEAD -- '*.py'` 恢复管线，跑通 `13_finetune_classifier.py` 的推理端，确认 `best_model` 可加载；
2. 用 random_from_A（n=743）做温度缩放 + 矩阵缩放，画出第一张校准前后 reliability diagram；
3. 开始做 AEJMC 海报框架（理论图 + 管线图 + 校准图三栏）。
