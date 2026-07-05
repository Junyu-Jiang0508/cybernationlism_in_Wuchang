# 接下来的操作指南：从聚类到完稿

> **路径变更（2026-07-05）**：管线脚本已从仓库根目录重组到 `src/` 分类子目录
> （映射表见 `src/README.md`）。本指南中的脚本名不变，运行时加上对应目录前缀，
> 如 `13_finetune_classifier.py` → `src/finetune/13_finetune_classifier.py`。

> 适用于：渊虚之羽 B站评论民族主义话语分析项目
> 当前进度：数据爬取 ✓ → 清洗抽样 ✓ → GPT标注 ✓ → SBERT Embedding ✓ → 聚类扫描 ✓
> 剩余工作：监督分类器 → 全量推理 → RQ分析 → 论文撰写

---

## 阶段一：环境准备与数据预处理（~1小时）

### 1.1 安装依赖

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn
pip install accelerate   # 混合精度训练
```

如果你有 GPU（推荐），确认 CUDA 可用：

```python
import torch
print(torch.cuda.is_available())       # 应为 True
print(torch.cuda.get_device_name(0))   # 显示 GPU 型号
```

如果没有本地 GPU，使用 Google Colab（免费 T4）或 AutoDL/恒源云（推荐 3090/4090，约 2-5 元/小时）。

### 1.2 整理标注数据

你的 `annotations_merged.csv` 有 2000 条标注数据，分布如下：

| discourse_type | 含义           | 数量 | 占比  |
| -------------- | -------------- | ---- | ----- |
| 0              | 纯游戏评价     | 440  | 22.0% |
| 1              | 情绪化非政治化 | 315  | 15.8% |
| 2              | 政治化游戏批评 | 739  | 37.0% |
| 3              | 显性民族主义   | 149  | 7.5%  |
| 4              | 中立/分析/科普 | 288  | 14.4% |
| 99             | 无法判断       | 69   | 3.5%  |

**关键处理决策**（在报告中需说明理由）：

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/annotations_merged.csv', encoding='utf-8-sig')

# 方案A（推荐）：去掉 type=99，保留 5 类（0-4）
df_clean = df[df['discourse_type'] != 99].copy()
df_clean['label'] = df_clean['discourse_type'].astype(int)
# 最终: 1931 条, 5 类

# 方案B（二分类辅助实验）：nationalist vs non-nationalist
df_clean['label_binary'] = (df_clean['discourse_type'].isin([2, 3])).astype(int)
# 888 (46%) vs 1043 (54%)，比较均衡

# 保存
df_clean.to_csv('data/train_ready.csv', index=False, encoding='utf-8-sig')
```

### 1.3 数据划分

```python
from sklearn.model_selection import StratifiedKFold, train_test_split

texts = df_clean['content_clean'].astype(str).tolist()
labels = df_clean['label'].tolist()

# 方案一（推荐）：5-fold 交叉验证（2000 条太少，CV 更稳健）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(texts, labels))
# 每 fold: ~1545 train / ~386 val

# 方案二：固定 70/15/15 划分
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)
```

---

## 阶段二：核心深度学习——微调分类器（~3-5小时含调参）

这是**课程评分最关键的部分**（占 30+20 分），必须做得充分。

### 2.1 基线模型选择

你需要对比至少 **2-3 个预训练模型**：

| 模型              | HuggingFace ID                | 参数量 | 特点                                |
| ----------------- | ----------------------------- | ------ | ----------------------------------- |
| BERT-base-Chinese | `bert-base-chinese`           | 110M   | 基线，Google 原版中文BERT           |
| RoBERTa-wwm-ext   | `hfl/chinese-roberta-wwm-ext` | 110M   | **推荐主力**，全词掩码，中文NLP常用 |
| MacBERT           | `hfl/chinese-macbert-base`    | 110M   | MLM改进，可能更好处理口语化文本     |

（可选加分项）如果时间充裕，加一个轻量模型做对比：
| MiniRBT | `hfl/minirbt-h256` | 16M | 蒸馏小模型，展示效率vs精度权衡 |

### 2.2 训练脚本框架

以下是完整的训练脚本结构，你可以直接使用：

```python
# 13_finetune_classifier.py
import os, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    accuracy_score
)

# ─── 配置 ───────────────────────────────────────────────
MODELS = {
    'bert':    'bert-base-chinese',
    'roberta': 'hfl/chinese-roberta-wwm-ext',
    'macbert': 'hfl/chinese-macbert-base',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_LABELS = 5  # discourse_type 0-4（去掉99）
MAX_LEN = 128   # B站评论中位长度40字，128 tokens绰绰有余

# ─── Dataset ─────────────────────────────────────────────
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx],
        }

# ─── 训练一个 fold ────────────────────────────────────────
def train_fold(model_name, hf_id, train_texts, train_labels,
               val_texts, val_labels, lr, epochs, batch_size,
               warmup_ratio, weight_decay, fold_idx, outdir):
    """返回 val metrics dict"""

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=NUM_LABELS
    ).to(DEVICE)

    train_ds = CommentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds   = CommentDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size*2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_f1 = 0
    best_state = None
    history = []

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0
        for batch in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        # ── Validate ──
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        history.append({
            'epoch': epoch+1, 'train_loss': avg_loss,
            'val_acc': acc, 'val_f1_macro': f1_macro,
            'val_f1_weighted': f1_weighted,
        })
        print(f'  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} '
              f'acc={acc:.4f} f1_macro={f1_macro:.4f}')

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存最佳模型
    if best_state:
        model.load_state_dict(best_state)
        save_path = os.path.join(outdir, f'{model_name}_fold{fold_idx}')
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # 用最佳模型重新跑 val，生成完整 report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch['labels'].cpu().tolist())

    report = classification_report(
        all_labels, all_preds,
        target_names=['type0_game','type1_emotional','type2_political',
                      'type3_nationalist','type4_neutral'],
        output_dict=True
    )

    return {
        'model': model_name, 'fold': fold_idx,
        'lr': lr, 'epochs': epochs, 'batch_size': batch_size,
        'warmup_ratio': warmup_ratio, 'weight_decay': weight_decay,
        'best_val_f1_macro': best_f1,
        'val_acc': report['accuracy'],
        'history': history,
        'report': report,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }
```

### 2.3 超参搜索网格

这是评分要求最高的部分，需要做**系统性**的超参扫描：

```python
# ─── 超参搜索 ────────────────────────────────────────────
HYPERPARAM_GRID = {
    # 第一轮：粗搜索（每个模型跑一次 5-fold CV）
    'round1': {
        'lr':            [1e-5, 2e-5, 3e-5, 5e-5],
        'epochs':        [3, 5, 8, 10],
        'batch_size':    [16, 32],
        'warmup_ratio':  [0.0, 0.1],
        'weight_decay':  [0.01],
    },
    # 第二轮：基于最佳 lr 精细搜索
    'round2': {
        'warmup_ratio':  [0.0, 0.05, 0.1, 0.15],
        'weight_decay':  [0.0, 0.01, 0.05],
    },
}
```

**实际执行策略**（节省时间）：

1. **第一轮**：固定 `batch_size=16, warmup=0.1, wd=0.01`，对 3 个模型 × 4 个 lr × 4 个 epoch 长度，只跑 1 fold（不是 5-fold），快速筛选最佳 lr 和 epoch → 约 48 次实验
2. **第二轮**：选定最佳模型+lr+epoch，做完整 5-fold CV，微调 warmup 和 wd → 约 12×5=60 次实验
3. **第三轮**：最终最佳配置做 5-fold，记录完整指标用于报告

```python
def run_full_experiment(outdir='results/finetune'):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv('data/train_ready.csv', encoding='utf-8-sig')
    texts = df['content_clean'].astype(str).tolist()
    labels = df['label'].tolist()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(texts, labels))
    all_results = []

    # ── 第一轮：粗搜索（只用 fold 0）────────────────────
    for model_name, hf_id in MODELS.items():
        for lr in [1e-5, 2e-5, 3e-5, 5e-5]:
            for ep in [3, 5, 8, 10]:
                train_idx, val_idx = folds[0]
                t_texts = [texts[i] for i in train_idx]
                t_labels = [labels[i] for i in train_idx]
                v_texts = [texts[i] for i in val_idx]
                v_labels = [labels[i] for i in val_idx]

                result = train_fold(
                    model_name, hf_id,
                    t_texts, t_labels, v_texts, v_labels,
                    lr=lr, epochs=ep, batch_size=16,
                    warmup_ratio=0.1, weight_decay=0.01,
                    fold_idx=0, outdir=outdir
                )
                all_results.append(result)
                # 每次保存，防止中断丢失
                with open(f'{outdir}/sweep_round1.json', 'w') as f:
                    json.dump(all_results, f, indent=2)

    # ── 选出 top 配置后做 5-fold（手动查看 round1 结果后执行）──
    # best_model, best_lr, best_ep = ...  # 从 round1 结果中选
    # for fold_idx in range(5):
    #     ...
```

### 2.4 必须记录的指标（报告用）

每个实验必须记录：

| 指标                | 说明              | 用途                         |
| ------------------- | ----------------- | ---------------------------- |
| Accuracy            | 总准确率          | 基本指标                     |
| Macro F1            | 各类 F1 取平均    | **主指标**（应对类别不平衡） |
| Weighted F1         | 加权 F1           | 辅助参考                     |
| Per-class F1        | 每类单独的 P/R/F1 | 分析哪类难分                 |
| Confusion Matrix    | 混淆矩阵          | 错误分析                     |
| Training Loss Curve | 每 epoch 的 loss  | 判断是否过拟合               |
| 训练时间            | 每个配置的耗时    | 效率分析                     |

### 2.5 附加实验（加分项）

**实验 A：二分类 vs 多分类**

```python
# 将 discourse_type 2+3 合并为 "nationalist"，0+1+4 为 "non-nationalist"
# 用同一模型训练二分类，对比 F1 提升幅度
# → 论文可讨论：粗粒度分类是否更可靠
```

**实验 B：多任务学习（同时预测3个维度）**

```python
# 一个模型同时输出 discourse_type + othering_intensity + affect_intensity
# 共享 BERT encoder，3 个分类头
class MultiTaskModel(torch.nn.Module):
    def __init__(self, hf_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden = self.encoder.config.hidden_size
        self.head_discourse  = torch.nn.Linear(hidden, 5)
        self.head_othering   = torch.nn.Linear(hidden, 4)
        self.head_affect     = torch.nn.Linear(hidden, 4)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS]
        return (
            self.head_discourse(cls),
            self.head_othering(cls),
            self.head_affect(cls),
        )
# → 论文可讨论：多任务是否产生正则化效果
```

**实验 C：MAX_LEN 消融**

```python
# 测试 MAX_LEN = 64, 128, 256, 512
# B站评论中位长度仅 40 字 → 预期 128 和 256 差异很小
# → 论文可展示效率 vs 精度权衡
```

**实验 D：冻结层数消融**

```python
# 冻结 BERT 前 N 层，只微调后面的层
# freeze_layers = [0, 4, 8, 10]（共 12 层）
for name, param in model.bert.named_parameters():
    layer_num = int(name.split('.')[2]) if 'layer' in name else -1
    if layer_num < freeze_n:
        param.requires_grad = False
# → 论文可讨论：中文 BERT 的层级表征对口语化文本的适应性
```

---

## 阶段三：全量推理——伪标签 Sample A（~30分钟）

用最佳模型对 30k 样本做推理：

```python
# 14_inference_sample_a.py
def inference_full(model_path, csv_path, out_path, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    texts = df['content_clean'].fillna('').astype(str).tolist()

    all_preds = []
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, padding='max_length',
            max_length=128, return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    df['pred_discourse_type'] = all_preds
    df['pred_confidence'] = [max(p) for p in all_probs]

    # 各类概率也保存（后续分析可能用到）
    for i in range(5):
        df[f'prob_type{i}'] = [p[i] for p in all_probs]

    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'推理完成: {len(df)} 条 → {out_path}')

    # 打印分布
    print(df['pred_discourse_type'].value_counts().sort_index())
    print(f'平均置信度: {df["pred_confidence"].mean():.4f}')
```

---

## 阶段四：回答 RQ 的定量分析（~2-3小时）

### 4.1 RQ1 分析：语义同质化与情感饱和

**分析 1：聚类 × 分类标签交叉**

```python
# 15_rq1_analysis.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score
)

# 加载数据
df_a = pd.read_csv('results/sample_A_predicted.csv', encoding='utf-8-sig')
labels_best = np.load('results/labels_minilm_best.npy')  # 聚类标签
df_a['cluster'] = labels_best
df_a['pred_dt'] = df_a['pred_discourse_type']

# 指标1: NMI（归一化互信息）
nmi = normalized_mutual_info_score(df_a['cluster'], df_a['pred_dt'])
ari = adjusted_rand_score(df_a['cluster'], df_a['pred_dt'])
homo = homogeneity_score(df_a['cluster'], df_a['pred_dt'])
print(f'NMI={nmi:.4f}  ARI={ari:.4f}  Homogeneity={homo:.4f}')

# 指标2: 各聚类内 discourse_type 分布
for c in sorted(df_a['cluster'].unique()):
    sub = df_a[df_a['cluster'] == c]
    dist = sub['pred_dt'].value_counts(normalize=True)
    dominant = dist.idxmax()
    purity = dist.max()
    print(f'Cluster {c} (n={len(sub)}): dominant=type{dominant} '
          f'purity={purity:.2f}')

# 指标3: 余弦相似度（同类 vs 跨类）
from sklearn.metrics.pairwise import cosine_similarity
embeddings = np.load('embeddings/embeddings_minilm.npy')

for dt in range(5):
    mask = df_a['pred_dt'] == dt
    if mask.sum() < 10:
        continue
    emb_sub = embeddings[mask.values]
    # 随机抽1000对计算平均余弦相似度
    n = min(1000, len(emb_sub))
    idx = np.random.choice(len(emb_sub), n, replace=False)
    sim = cosine_similarity(emb_sub[idx])
    avg_sim = sim[np.triu_indices(n, k=1)].mean()
    print(f'Type {dt}: avg intra-class cosine sim = {avg_sim:.4f}')
```

**分析 2：Script 模板检测**

```python
# 找出高频"模板化"表达
from collections import Counter
import re

# 对 type 2+3 的评论，提取高频 n-gram
nationalist = df_a[df_a['pred_dt'].isin([2, 3])]['content_clean'].astype(str)

def extract_ngrams(texts, n=3):
    all_grams = []
    for t in texts:
        chars = list(t)
        for i in range(len(chars) - n + 1):
            all_grams.append(''.join(chars[i:i+n]))
    return Counter(all_grams)

trigrams = extract_ngrams(nationalist, 3)
print('Top 30 三字词频:')
for gram, count in trigrams.most_common(30):
    print(f'  {gram}: {count}')

# 计算 type 2+3 vs type 0+1+4 的 n-gram 重叠率
# → 如果 nationalist 评论共享大量相同短语，支持"工业化脚本"假设
```

### 4.2 RQ2 分析：平台算法与可见性分配

```python
# 16_rq2_analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, negativebinomial

df = pd.read_csv('results/sample_A_predicted.csv', encoding='utf-8-sig')
df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
df['log_likes'] = np.log1p(df['like_count'])

# ── 分析1: 各 discourse_type 的点赞分布（Kruskal-Wallis 检验）──
groups = [df[df['pred_discourse_type']==t]['log_likes'].values for t in range(5)]
H, p = stats.kruskal(*groups)
print(f'Kruskal-Wallis H={H:.2f}, p={p:.6f}')

# 事后检验：Mann-Whitney U（两两比较）
from itertools import combinations
for i, j in combinations(range(5), 2):
    U, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
    print(f'  Type {i} vs {j}: U={U:.0f}, p={p:.6f}')

# ── 分析2: 负二项回归（因为 like_count 是过度离散的计数变量）──
df['is_political'] = (df['pred_discourse_type'].isin([2, 3])).astype(int)
df['is_nationalist'] = (df['pred_discourse_type'] == 3).astype(int)

# 简单模型
model1 = negativebinomial(
    'like_count ~ is_political',
    data=df[df['like_count'] >= 0]
).fit(disp=0)
print(model1.summary())

# 带控制变量的模型（控制发布时间、视频）
df['post_hour'] = pd.to_datetime(
    pd.to_numeric(df['create_time'], errors='coerce'), unit='s'
).dt.hour
df['is_weekend'] = pd.to_datetime(
    pd.to_numeric(df['create_time'], errors='coerce'), unit='s'
).dt.dayofweek.isin([5, 6]).astype(int)

# 如果使用 predicted affect/othering（从多任务模型或GPT标注对齐）
# model2 = negativebinomial(
#     'like_count ~ C(pred_discourse_type) + post_hour + is_weekend',
#     data=df
# ).fit(disp=0)
```

### 4.3 时间动态分析（加分项）

```python
# 17_temporal_analysis.py
# 分析民族主义话语随时间的演变

df['create_dt'] = pd.to_datetime(
    pd.to_numeric(df['create_time'], errors='coerce'), unit='s'
)

# 按周聚合各 discourse_type 的比例
df['week'] = df['create_dt'].dt.to_period('W')
weekly = df.groupby('week')['pred_discourse_type'].value_counts(normalize=True)
weekly = weekly.unstack(fill_value=0)

# 绘制堆叠面积图
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
weekly.plot.area(ax=ax, stacked=True, alpha=0.7)
ax.set_ylabel('Proportion')
ax.set_title('Discourse Type Distribution Over Time')
plt.tight_layout()
plt.savefig('results/temporal_discourse.png', dpi=150)
```

---

## 阶段五：可视化（报告用图表）

你需要在报告中放置以下图表：

### 必需图表清单

| 图编号 | 内容                                                 | 类型                         |
| ------ | ---------------------------------------------------- | ---------------------------- |
| Fig 1  | 数据管线总览                                         | 流程图（手动绘制或 Mermaid） |
| Fig 2  | discourse_type 分布（GPT标注 vs 模型预测）           | 并排柱状图                   |
| Fig 3  | 超参扫描：不同 lr × epoch 的 F1 热力图               | Heatmap                      |
| Fig 4  | 模型对比：3 模型的 5-fold CV F1 箱线图               | Box plot                     |
| Fig 5  | 最佳模型的混淆矩阵                                   | Heatmap                      |
| Fig 6  | Training/validation loss 曲线                        | 折线图                       |
| Fig 7  | t-SNE/UMAP 可视化（embedding 着色按 discourse_type） | 散点图                       |
| Fig 8  | 各 discourse_type 的 like_count 箱线图               | Box plot                     |
| Fig 9  | 时间线：各话语类型占比变化                           | 堆叠面积图                   |

```python
# 18_visualizations.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Fig 3: LR × Epoch 热力图 ──
results = json.load(open('results/finetune/sweep_round1.json'))
# 筛选 roberta 的结果
rob = [r for r in results if r['model'] == 'roberta']
lrs = sorted(set(r['lr'] for r in rob))
eps = sorted(set(r['epochs'] for r in rob))
mat = np.zeros((len(lrs), len(eps)))
for r in rob:
    i = lrs.index(r['lr'])
    j = eps.index(r['epochs'])
    mat[i, j] = r['best_val_f1_macro']

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(mat, annot=True, fmt='.3f', xticklabels=eps,
            yticklabels=[f'{lr:.0e}' for lr in lrs], cmap='YlOrRd', ax=ax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Learning Rate')
ax.set_title('RoBERTa: Macro F1 by LR × Epoch')
plt.tight_layout()
plt.savefig('results/fig3_lr_epoch_heatmap.png', dpi=150)

# ── Fig 7: t-SNE 可视化 ──
from sklearn.manifold import TSNE

embeddings = np.load('embeddings/embeddings_minilm.npy')
# 随机抽 5000 条（t-SNE 太慢跑 3 万）
idx = np.random.choice(len(embeddings), 5000, replace=False)
emb_sub = embeddings[idx]
labels_sub = df_a['pred_discourse_type'].values[idx]

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
emb_2d = tsne.fit_transform(emb_sub)

fig, ax = plt.subplots(figsize=(10, 8))
type_names = ['Game review', 'Emotional', 'Politicized',
              'Nationalist', 'Neutral']
colors = ['#4CAF50', '#FF9800', '#E53935', '#9C27B0', '#607D8B']
for t in range(5):
    mask = labels_sub == t
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=colors[t], s=5, alpha=0.5, label=type_names[t])
ax.legend(markerscale=5)
ax.set_title('t-SNE of SBERT Embeddings (colored by predicted discourse type)')
plt.tight_layout()
plt.savefig('results/fig7_tsne.png', dpi=150)
```

---

## 阶段六：论文撰写（NeurIPS 格式）

### 6.1 推荐结构

| 章节                        | 页数  | 内容要点                                                                                                                                                   |
| --------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Abstract**                | 0.3页 | 问题、方法、关键发现（一句话总结每个RQ的结论）                                                                                                             |
| **1. Introduction**         | 1页   | 背景（渊虚之羽事件）→ 研究缺口 → RQ1/RQ2 → 贡献声明                                                                                                        |
| **2. Related Work**         | 0.7页 | 平台民族主义、仇恨话语检测、BERT中文NLP、游戏争议分析                                                                                                      |
| **3. Data & Methodology**   | 1.5页 | 3.1 数据收集（150万→清洗→抽样）；3.2 GPT标注方案（prompt设计 + 标注维度）；3.3 模型架构（BERT/RoBERTa + 分类头）；3.4 聚类（SBERT + KMeans）；3.5 统计分析 |
| **4. Experiments**          | 2页   | 4.1 实验设置；4.2 超参扫描结果（含热力图）；4.3 模型对比（含5-fold CV结果表）；4.4 消融实验（MAX_LEN/冻结层/二分类vs多分类）                               |
| **5. Results & Discussion** | 1.5页 | 5.1 RQ1 分析结果（聚类纯度、余弦相似度、模板检测）；5.2 RQ2 分析结果（回归系数、效应量）；5.3 时间动态                                                     |
| **6. Conclusion**           | 0.5页 | 总结发现 + 局限性 + 未来工作                                                                                                                               |
| **References**              | 0.5页 | ~15-25 篇                                                                                                                                                  |
| **Appendix**                | 可选  | GPT prompt 完整版、额外图表                                                                                                                                |

### 6.2 Bonus Points 部分

在报告末尾单独写一个 **Bonus Points** 小节，包含：

1. **Novel dataset（原创数据集）**: 自主爬取 150 万条 B 站评论，覆盖 13,124 个视频，时间跨度 5 个月。这不是标准 benchmark，是真实世界的大规模中文社交媒体数据。
2. **Multi-stage pipeline（多阶段管线）**: 关键词抽样 → GPT-4o 标注 → SBERT 双模型 embedding → 无监督聚类 → 监督微调 → 全量推理 → 统计分析，每个环节都有methodological justification。
3. **Computational social science application（计算社会科学应用）**: 将深度学习技术应用于政治话语检测，这不是标准分类任务，而是需要处理隐含立场、文化语境、讽刺等复杂语言现象。
4. **Comprehensive experiments（充分的实验）**: 3 个模型 × 16+ 超参组合 × 5-fold CV，加消融实验（MAX_LEN、冻结层、任务粒度）。

---

## 时间安排建议

| 天    | 任务                                           | 预计耗时         |
| ----- | ---------------------------------------------- | ---------------- |
| Day 1 | 环境准备 + 数据预处理 + 第一轮粗搜索（跑过夜） | 3-4h 操作 + 过夜 |
| Day 2 | 分析 Round 1 结果 → 第二轮精细搜索 → 5-fold CV | 4-5h             |
| Day 3 | 消融实验（MAX_LEN/二分类/冻结层） + 全量推理   | 3-4h             |
| Day 4 | RQ1 + RQ2 定量分析 + 可视化                    | 4-5h             |
| Day 5 | 撰写报告初稿                                   | 5-6h             |
| Day 6 | 修改 + 排版 + 检查                             | 3-4h             |

---

## 关键提醒

1. **先跑通再调优**：第一步是让一个模型在一个 fold 上成功训练并输出 F1，确认管线无 bug。然后再大规模扫描。
2. **保存所有结果**：每次实验都写入 JSON/CSV。报告需要呈现完整的扫描过程，而不仅是最终结果。
3. **GPU 内存管理**：如果 OOM，减 batch_size 为 8，并使用梯度累积：
   ```python
   accumulation_steps = 4
   loss = loss / accumulation_steps
   loss.backward()
   if (step + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```
4. **不要忽略错误分析**：在报告中展示 confusion matrix 并讨论哪些类型容易混淆（预计 type 1 vs type 2 边界最模糊），这本身就是有价值的发现。
5. **class weight**：由于 type 3 只有 149 条（7.5%），考虑在 loss 中加权：
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
   loss_fn = torch.nn.CrossEntropyLoss(
       weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
   )
   ```
