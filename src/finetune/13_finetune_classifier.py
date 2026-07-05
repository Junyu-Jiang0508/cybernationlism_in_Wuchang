# -*- coding: utf-8 -*-
"""
13_finetune_classifier.py
中文 BERT 微调分类器：discourse_type 5分类（0-4）

用法:
  # 第一步：快速验证管线（1个模型, 1个fold, 3 epochs）
  python 13_finetune_classifier.py --mode smoke

  # 第二步：粗搜索（3模型 × 4lr × 4epoch, 只用fold0）
  python 13_finetune_classifier.py --mode sweep_round1

  # 第三步：精细搜索（最佳模型, 5-fold CV, 指定配置）
  python 13_finetune_classifier.py --mode sweep_round2 \
      --model roberta --lr 2e-5 --epochs 5

  # 第四步：最终训练 + 保存最佳模型
  python 13_finetune_classifier.py --mode final \
      --model roberta --lr 2e-5 --epochs 5

  # 第五步：全量推理 Sample A
  python 13_finetune_classifier.py --mode inference \
      --model-path results/finetune/best_model

依赖: pip install torch transformers scikit-learn pandas
"""
from __future__ import annotations

import os
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"  # 关闭 HF 自动转换后台线程，消除 403 报错

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results" / "finetune"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_LABELS = 5
TYPE_NAMES = ['type0_game', 'type1_emotional', 'type2_political',
              'type3_nationalist', 'type4_neutral']
TYPE_NAMES_BINARY = ['non_nationalist', 'nationalist']

# ─── 模型注册表 ──────────────────────────────────────────
MODELS = {
    'bert':    'bert-base-chinese',
    'roberta': 'hfl/chinese-roberta-wwm-ext',
    'macbert': 'hfl/chinese-macbert-base',
}

# ─── 延迟导入 transformers（启动更快）────────────────────
_tokenizer_cls = None
_model_cls = None
_scheduler_fn = None

def _import_transformers():
    global _tokenizer_cls, _model_cls, _scheduler_fn
    if _tokenizer_cls is None:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        _tokenizer_cls = AutoTokenizer
        _model_cls = AutoModelForSequenceClassification
        _scheduler_fn = get_linear_schedule_with_warmup


# ─── Dataset ─────────────────────────────────────────────
class CommentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer, max_len: int = 128):
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


# ─── 数据加载 ─────────────────────────────────────────────
def load_data(csv_path: Path, binary: bool = False) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 过滤 discourse_type == 99
    df = df[df['discourse_type'] != 99].copy()

    if binary:
        # 二分类：type 2+3 → 1（nationalist），0+1+4 → 0（non-nationalist）
        df['label'] = df['discourse_type'].isin([2, 3]).astype(int)
        texts = df['content_clean'].fillna('').astype(str).tolist()
        labels = df['label'].tolist()
        print(f'加载 {len(texts)} 条（二分类模式）')
        for t, name in enumerate(TYPE_NAMES_BINARY):
            n = labels.count(t)
            print(f'  {name}: {n} ({n/len(labels)*100:.1f}%)')
    else:
        # 重映射标签为 0-4 连续整数
        label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        df['label'] = df['discourse_type'].map(label_map)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        texts = df['content_clean'].fillna('').astype(str).tolist()
        labels = df['label'].tolist()
        print(f'加载 {len(texts)} 条（去除 type=99 后）')
        for t in range(5):
            n = labels.count(t)
            print(f'  type {t} ({TYPE_NAMES[t]}): {n} ({n/len(labels)*100:.1f}%)')

    return texts, labels


# ─── 单次训练 ─────────────────────────────────────────────
def train_one(
    model_slug: str,
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    lr: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 16,
    max_len: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    use_class_weight: bool = True,
    freeze_layers: int = 0,
    fold_idx: int = 0,
    save_dir: Optional[Path] = None,
    binary: bool = False,
) -> Dict:
    _import_transformers()
    hf_id = MODELS[model_slug]
    n_labels = 2 if binary else NUM_LABELS
    t_names = TYPE_NAMES_BINARY if binary else TYPE_NAMES

    print(f'\n{"="*60}')
    print(f'模型: {model_slug} ({hf_id})')
    print(f'配置: lr={lr}, epochs={epochs}, bs={batch_size}, '
          f'max_len={max_len}, warmup={warmup_ratio}, wd={weight_decay}')
    print(f'Fold: {fold_idx}, freeze_layers: {freeze_layers}, binary: {binary}')
    print(f'Device: {DEVICE}')
    print(f'{"="*60}')

    tokenizer = _tokenizer_cls.from_pretrained(hf_id)
    model = _model_cls.from_pretrained(hf_id, num_labels=n_labels).to(DEVICE)

    # 冻结前 N 层
    if freeze_layers > 0:
        encoder = getattr(model, 'bert', None) or getattr(model, 'roberta', None)
        if encoder is not None:
            for name, param in encoder.named_parameters():
                parts = name.split('.')
                if 'layer' in parts:
                    layer_idx = int(parts[parts.index('layer') + 1])
                    if layer_idx < freeze_layers:
                        param.requires_grad = False
            frozen = sum(1 for p in encoder.parameters() if not p.requires_grad)
            total = sum(1 for p in encoder.parameters())
            print(f'  冻结 {frozen}/{total} 个参数')

    # class weight
    loss_fn = None
    if use_class_weight:
        weights = compute_class_weight(
            'balanced', classes=np.unique(train_labels), y=train_labels
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        )
        print(f'  Class weights: {[f"{w:.2f}" for w in weights]}')

    train_ds = CommentDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = CommentDataset(val_texts, val_labels, tokenizer, max_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    total_steps = len(train_dl) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _scheduler_fn(optimizer, warmup_steps, total_steps)

    best_f1 = 0.0
    best_state = None
    history = []

    t_start = time.time()

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            if loss_fn is not None:
                loss = loss_fn(outputs.logits, batch['labels'])
            else:
                # 使用 HF 内置的 loss（需要传 labels）
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── Validate ──
        model.eval()
        all_preds, all_labels_v = [], []
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                )
                if loss_fn is not None:
                    vl = loss_fn(outputs.logits, batch['labels'])
                else:
                    vl = torch.nn.CrossEntropyLoss()(outputs.logits, batch['labels'])
                val_loss += vl.item()
                n_val += 1
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels_v.extend(batch['labels'].cpu().tolist())

        avg_val_loss = val_loss / max(n_val, 1)
        acc = accuracy_score(all_labels_v, all_preds)
        f1_macro = f1_score(all_labels_v, all_preds, average='macro')
        f1_weighted = f1_score(all_labels_v, all_preds, average='weighted')

        history.append({
            'epoch': epoch + 1,
            'train_loss': round(avg_loss, 5),
            'val_loss': round(avg_val_loss, 5),
            'val_acc': round(acc, 4),
            'val_f1_macro': round(f1_macro, 4),
            'val_f1_weighted': round(f1_weighted, 4),
        })

        marker = ' *' if f1_macro > best_f1 else ''
        print(f'  Epoch {epoch+1}/{epochs}: '
              f'train_loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} '
              f'acc={acc:.4f} f1_macro={f1_macro:.4f}{marker}')

        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t_start

    # 用最佳模型评估
    if best_state:
        model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    all_preds, all_labels_v = [], []
    all_probs = []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            ).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels_v.extend(batch['labels'].cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    report = classification_report(
        all_labels_v, all_preds,
        target_names=t_names, output_dict=True
    )
    cm = confusion_matrix(all_labels_v, all_preds)

    # 保存模型
    if save_dir is not None:
        suffix = '_binary' if binary else ''
        save_path = save_dir / f'{model_slug}_fold{fold_idx}{suffix}'
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f'  模型已保存: {save_path}')

    result = {
        'model': model_slug,
        'hf_id': hf_id,
        'fold': fold_idx,
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'max_len': max_len,
        'warmup_ratio': warmup_ratio,
        'weight_decay': weight_decay,
        'freeze_layers': freeze_layers,
        'use_class_weight': use_class_weight,
        'binary': binary,
        'best_val_f1_macro': round(best_f1, 4),
        'val_acc': round(report['accuracy'], 4),
        'val_f1_weighted': round(report['weighted avg']['f1-score'], 4),
        'per_class_f1': {
            t_names[i]: round(report[t_names[i]]['f1-score'], 4)
            for i in range(n_labels)
        },
        'confusion_matrix': cm.tolist(),
        'history': history,
        'elapsed_seconds': round(elapsed, 1),
        'n_train': len(train_texts),
        'n_val': len(val_texts),
    }

    print(f'\n  最佳 Macro F1: {best_f1:.4f} | 耗时: {elapsed:.0f}s')
    print(f'  Per-class F1: {result["per_class_f1"]}')
    print(f'  Confusion Matrix:\n{cm}')

    return result


# ─── 模式: smoke test ────────────────────────────────────
def mode_smoke(args):
    texts, labels = load_data(args.csv)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(texts, labels)))

    t_texts = [texts[i] for i in train_idx]
    t_labels = [labels[i] for i in train_idx]
    v_texts = [texts[i] for i in val_idx]
    v_labels = [labels[i] for i in val_idx]

    result = train_one(
        'roberta', t_texts, t_labels, v_texts, v_labels,
        lr=2e-5, epochs=3, batch_size=16,
        save_dir=RESULTS_DIR,
    )

    out = RESULTS_DIR / 'smoke_result.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n[OK] Smoke test done -> {out}')


# ─── 模式: sweep round 1（粗搜索）────────────────────────
def mode_sweep_round1(args):
    texts, labels = load_data(args.csv)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(texts, labels)))

    t_texts = [texts[i] for i in train_idx]
    t_labels = [labels[i] for i in train_idx]
    v_texts = [texts[i] for i in val_idx]
    v_labels = [labels[i] for i in val_idx]

    out_file = RESULTS_DIR / 'sweep_round1.json'
    existing = []
    if out_file.exists():
        existing = json.loads(out_file.read_text(encoding='utf-8'))
        print(f'已有 {len(existing)} 条结果，断点续传')

    done_keys = set()
    for r in existing:
        done_keys.add(f"{r['model']}_{r['lr']}_{r['epochs']}")

    lrs = [1e-5, 2e-5, 3e-5, 5e-5]
    epoch_list = [3, 5, 8, 10]

    total = len(MODELS) * len(lrs) * len(epoch_list)
    done = len(existing)

    for model_slug in MODELS:
        for lr in lrs:
            for ep in epoch_list:
                key = f"{model_slug}_{lr}_{ep}"
                if key in done_keys:
                    continue
                done += 1
                print(f'\n[{done}/{total}] {key}')

                result = train_one(
                    model_slug, t_texts, t_labels, v_texts, v_labels,
                    lr=lr, epochs=ep, batch_size=16,
                    fold_idx=0,
                )
                existing.append(result)

                # 每次保存
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f'\n[OK] Round 1 done: {len(existing)} results -> {out_file}')

    # 打印 Top 10
    existing.sort(key=lambda x: x['best_val_f1_macro'], reverse=True)
    print('\nTop 10 配置:')
    for i, r in enumerate(existing[:10]):
        print(f'  {i+1}. {r["model"]} lr={r["lr"]:.0e} ep={r["epochs"]} '
              f'F1={r["best_val_f1_macro"]:.4f}')


# ─── 模式: sweep round 2（5-fold CV）─────────────────────
def mode_sweep_round2(args):
    binary = args.binary
    texts, labels = load_data(args.csv, binary=binary)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(texts, labels))

    suffix = '_binary' if binary else f'_maxlen{args.max_len}'
    out_file = RESULTS_DIR / f'sweep_round2_{args.model}{suffix}.json'
    all_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        t_texts = [texts[i] for i in train_idx]
        t_labels = [labels[i] for i in train_idx]
        v_texts = [texts[i] for i in val_idx]
        v_labels = [labels[i] for i in val_idx]

        result = train_one(
            args.model, t_texts, t_labels, v_texts, v_labels,
            lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
            warmup_ratio=args.warmup, weight_decay=args.wd,
            max_len=args.max_len,
            fold_idx=fold_idx,
            binary=binary,
        )
        all_results.append(result)

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

    f1s = [r['best_val_f1_macro'] for r in all_results]
    tag = 'binary' if binary else f'maxlen={args.max_len}'
    print(f'\n{"="*60}')
    print(f'5-Fold CV 结果: {args.model} [{tag}]')
    print(f'  F1 per fold: {[f"{x:.4f}" for x in f1s]}')
    print(f'  Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')
    print(f'{"="*60}')


# ─── 模式: final（最终训练，保存模型）────────────────────
def mode_final(args):
    texts, labels = load_data(args.csv)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(skf.split(texts, labels)))

    t_texts = [texts[i] for i in train_idx]
    t_labels = [labels[i] for i in train_idx]
    v_texts = [texts[i] for i in val_idx]
    v_labels = [labels[i] for i in val_idx]

    save_dir = RESULTS_DIR / 'best_model'
    result = train_one(
        args.model, t_texts, t_labels, v_texts, v_labels,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
        warmup_ratio=args.warmup, weight_decay=args.wd,
        max_len=args.max_len,
        save_dir=save_dir,
    )

    with open(save_dir / 'training_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\n[OK] Final model saved -> {save_dir}')


# ─── 模式: inference（全量推理 Sample A）──────────────────
def mode_inference(args):
    _import_transformers()
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f'模型路径不存在: {model_path}', file=sys.stderr)
        sys.exit(1)

    tokenizer = _tokenizer_cls.from_pretrained(model_path)
    model = _model_cls.from_pretrained(model_path).to(DEVICE)
    model.eval()

    csv_in = Path(args.inference_csv)
    df = pd.read_csv(csv_in, encoding='utf-8-sig')
    texts = df['content_clean'].fillna('').astype(str).tolist()

    infer_max_len = args.max_len
    # MAX_LEN=256 时 batch_size 减半防显存溢出
    batch_size = 32 if infer_max_len >= 256 else 64
    print(f'推理 {len(texts)} 条 (max_len={infer_max_len}, batch_size={batch_size})...')
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, padding='max_length',
            max_length=infer_max_len, return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        if (i // batch_size) % 50 == 0:
            print(f'  {i+len(batch_texts)}/{len(texts)}')

    df['pred_discourse_type'] = all_preds
    df['pred_confidence'] = [max(p) for p in all_probs]
    for j in range(5):
        df[f'prob_type{j}'] = [p[j] for p in all_probs]

    out_path = RESULTS_DIR / 'sample_A_predicted.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f'\n[OK] Inference done -> {out_path}')
    print(f'预测分布:')
    print(df['pred_discourse_type'].value_counts().sort_index())
    print(f'平均置信度: {df["pred_confidence"].mean():.4f}')


# ─── Main ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Chinese BERT 微调 discourse_type 分类器')
    ap.add_argument('--csv', type=Path,
                    default=DATA_DIR / 'annotations_merged.csv')
    ap.add_argument('--mode', required=True,
                    choices=['smoke', 'sweep_round1', 'sweep_round2',
                             'final', 'inference'])
    ap.add_argument('--model', default='roberta',
                    choices=list(MODELS.keys()))
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--warmup', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--model-path', type=str, default='')
    ap.add_argument('--inference-csv', type=str,
                    default=str(DATA_DIR / 'sample_A_30k.csv'))
    ap.add_argument('--binary', action='store_true',
                    help='二分类模式：nationalist(type2+3) vs non-nationalist(0+1+4)')
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == 'smoke':
        mode_smoke(args)
    elif args.mode == 'sweep_round1':
        mode_sweep_round1(args)
    elif args.mode == 'sweep_round2':
        mode_sweep_round2(args)
    elif args.mode == 'final':
        mode_final(args)
    elif args.mode == 'inference':
        mode_inference(args)


if __name__ == '__main__':
    main()
