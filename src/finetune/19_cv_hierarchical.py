# -*- coding: utf-8 -*-
"""
19_cv_hierarchical.py — Phase 1：统一 5 折 CV（回应 AEJMC R3-1 / R3-2 的数据前置）

在同一个 StratifiedKFold(5, shuffle, seed=42) 切分下训练 6 组任务，
产出全部 out-of-fold logits（校准与层级对比的无泄漏基础）：

  flat5   五路 softmax 基线（与 best_model 同配置：lr=3e-5, ep=3, max_len=256）
  stageA  层级第一步：政治(2,3)=1 vs 非政治(0,1,4)=0
  stageB  层级第二步：政治子集内 type2=0 vs type3=1
  stageC  非政治子集内 type0/1/4 三分（用于合成完整五类层级预测）
  altA    对照切法第一步:实质性(2,3,4)=1 vs 非实质性(0,1)=0
  altB    对照切法第二步:实质性子集内 type2/3/4 三分

注意：
  - 子集任务（stageB/C, altB）只在子集上训练与选 epoch，但对整个验证折推理，
    这样层级预测可以在任意 Stage A 路由下合成完整概率。
  - flat5 / stageA / stageB / stageC 的每折模型还对 Sample A（30k）全量推理，
    折间平均后用于 prevalence 估计（21_ 脚本）。
  - 断点续传：每个 (task, fold) 的 OOF 与 Sample A 概率各存一个 npz chunk，
    已存在即跳过；--mode merge 合并为分析用文件。

用法:
  python 19_cv_hierarchical.py --mode cv               # 训练 + OOF + Sample A 推理
  python 19_cv_hierarchical.py --mode cv --skip-sample-a
  python 19_cv_hierarchical.py --mode merge            # 合并 chunks → parquet/npz

依赖: torch transformers scikit-learn pandas pyarrow
"""
from __future__ import annotations

import os

os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR / "results" / "phase1"
OOF_DIR = OUT_DIR / "oof_chunks"
SA_DIR = OUT_DIR / "sampleA_chunks"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# 基座模型：优先本地缓存（本网络 HF 直连不通，见 docs/DATA_STATEMENT §四）
_LOCAL_BASE = Path.home() / ".cache" / "local_models" / "chinese-roberta-wwm-ext"
BASE_MODEL = str(_LOCAL_BASE) if _LOCAL_BASE.exists() else "hfl/chinese-roberta-wwm-ext"

# ─── 任务注册表 ───────────────────────────────────────────
# label_map: 原 discourse_type → 任务标签；不在 map 中的类型不参与该任务的训练/选型
TASKS = {
    "flat5": {
        "label_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        "class_names": ["type0", "type1", "type2", "type3", "type4"],
        "epochs": 3,
        "sample_a": True,
    },
    "stageA": {
        "label_map": {0: 0, 1: 0, 4: 0, 2: 1, 3: 1},
        "class_names": ["nonpolitical", "political"],
        "epochs": 3,
        "sample_a": True,
    },
    "stageB": {
        "label_map": {2: 0, 3: 1},
        "class_names": ["type2", "type3"],
        "epochs": 5,
        "sample_a": True,
    },
    "stageC": {
        "label_map": {0: 0, 1: 1, 4: 2},
        "class_names": ["type0", "type1", "type4"],
        "epochs": 5,
        "sample_a": True,
    },
    "altA": {
        "label_map": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
        "class_names": ["nonsubstantive", "substantive"],
        "epochs": 3,
        "sample_a": False,
    },
    "altB": {
        "label_map": {2: 0, 3: 1, 4: 2},
        "class_names": ["type2", "type3", "type4"],
        "epochs": 5,
        "sample_a": False,
    },
}

TRAIN_CFG = dict(lr=3e-5, batch_size=16, max_len=256, warmup_ratio=0.1, weight_decay=0.01)


class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_annotations() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "annotations_merged.csv", encoding="utf-8-sig")
    df = df[df["discourse_type"] != 99].copy().reset_index(drop=True)
    df["content_clean"] = df["content_clean"].fillna("").astype(str)
    n_random = (df["sample_b_source"] == "random_from_A").sum()
    print(f"标注集: {len(df)} 条（去 type=99），其中 random_from_A = {n_random}")
    return df


def make_folds(df: pd.DataFrame):
    """全部任务共用的折切分：在五类标签上分层，与 13_ 脚本协议一致。"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(df.index, df["discourse_type"]))


def train_fold_model(task: str, train_texts, train_labels, val_texts, val_labels, tokenizer, fold_idx: int):
    """训练单折模型，按验证折 macro-F1 选最佳 epoch，返回模型与历史。"""
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    cfg = TASKS[task]
    n_labels = len(cfg["class_names"])
    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=n_labels).to(DEVICE)

    weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

    train_ds = CommentDataset(train_texts, train_labels, tokenizer, TRAIN_CFG["max_len"])
    val_ds = CommentDataset(val_texts, val_labels, tokenizer, TRAIN_CFG["max_len"])
    train_dl = DataLoader(train_ds, batch_size=TRAIN_CFG["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=TRAIN_CFG["batch_size"] * 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    total_steps = len(train_dl) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * TRAIN_CFG["warmup_ratio"]), total_steps)

    best_f1, best_state, history = -1.0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        preds, labels_v = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                labels_v.extend(batch["labels"].cpu().tolist())
        f1 = f1_score(labels_v, preds, average="macro")
        history.append({"epoch": epoch + 1, "train_loss": round(total_loss / max(n_batches, 1), 5),
                        "val_f1_macro": round(f1, 4)})
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()
    return model, best_f1, history


@torch.no_grad()
def predict_logits(model, tokenizer, texts, max_len=256, batch_size=128, log_every=0):
    """动态 padding 批量推理，返回 logits ndarray [N, C]。"""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size], truncation=True, padding=True, max_length=max_len, return_tensors="pt"
        ).to(DEVICE)
        out.append(model(**enc).logits.cpu().numpy())
        if log_every and (i // batch_size) % log_every == 0:
            print(f"    推理 {i + len(out[-1])}/{len(texts)}", flush=True)
    return np.concatenate(out, axis=0)


def mode_cv(args):
    from transformers import AutoTokenizer

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    SA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_annotations()
    folds = make_folds(df)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    sample_a = None
    if not args.skip_sample_a:
        sample_a = pd.read_csv(DATA_DIR / "sample_A_30k.csv", encoding="utf-8-sig")
        sample_a["content_clean"] = sample_a["content_clean"].fillna("").astype(str)
        print(f"Sample A: {len(sample_a)} 条待推理")

    metrics_file = OUT_DIR / "cv_train_metrics.json"
    metrics = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else []
    done_keys = {(m["task"], m["fold"]) for m in metrics}

    for task, cfg in TASKS.items():
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            oof_file = OOF_DIR / f"{task}_fold{fold_idx}.npz"
            sa_file = SA_DIR / f"{task}_fold{fold_idx}.npz"
            need_sa = cfg["sample_a"] and sample_a is not None and not sa_file.exists()
            if oof_file.exists() and not need_sa and (task, fold_idx) in done_keys:
                print(f"[skip] {task} fold{fold_idx} 已完成")
                continue

            t0 = time.time()
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # 子集任务：训练/选型只用 label_map 覆盖的类型
            tr_mask = train_df["discourse_type"].isin(cfg["label_map"])
            va_mask = val_df["discourse_type"].isin(cfg["label_map"])
            tr_sub, va_sub = train_df[tr_mask], val_df[va_mask]

            print(f"\n[{task} fold{fold_idx}] train={len(tr_sub)} val_sel={len(va_sub)} "
                  f"val_all={len(val_df)}", flush=True)

            model, best_f1, history = train_fold_model(
                task,
                tr_sub["content_clean"].tolist(),
                tr_sub["discourse_type"].map(cfg["label_map"]).tolist(),
                va_sub["content_clean"].tolist(),
                va_sub["discourse_type"].map(cfg["label_map"]).tolist(),
                tokenizer, fold_idx,
            )
            print(f"  best val macro-F1 = {best_f1:.4f}  ({time.time() - t0:.0f}s)", flush=True)

            # OOF：对整个验证折推理（含子集外样本，供层级合成）
            logits = predict_logits(model, tokenizer, val_df["content_clean"].tolist(),
                                    max_len=TRAIN_CFG["max_len"])
            np.savez_compressed(oof_file, comment_id=val_df["comment_id"].values,
                                row_idx=val_idx, logits=logits)

            if need_sa:
                t1 = time.time()
                sa_logits = predict_logits(model, tokenizer, sample_a["content_clean"].tolist(),
                                           max_len=TRAIN_CFG["max_len"], log_every=100)
                np.savez_compressed(sa_file, comment_id=sample_a["comment_id"].values, logits=sa_logits)
                print(f"  Sample A 推理完成 ({time.time() - t1:.0f}s)", flush=True)

            metrics = [m for m in metrics if not (m["task"] == task and m["fold"] == fold_idx)]
            metrics.append({"task": task, "fold": fold_idx, "best_val_f1_macro": round(best_f1, 4),
                            "n_train": len(tr_sub), "n_val_selected": len(va_sub),
                            "history": history, "elapsed_s": round(time.time() - t0, 1)})
            metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

            del model
            torch.cuda.empty_cache()

    print("\n[OK] CV 全部完成 →", metrics_file)


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def mode_merge(_args):
    df = load_annotations()
    folds = make_folds(df)

    # ── OOF 合并：宽表，每任务 logits 列 ──
    wide = df[["comment_id", "content_clean", "discourse_type", "sample_b_source",
               "keyword_pool_tier", "othering_intensity", "affect_intensity",
               "video_id", "like_count"]].copy()
    wide["fold"] = -1
    for fold_idx, (_tr, val_idx) in enumerate(folds):
        wide.loc[val_idx, "fold"] = fold_idx

    for task, cfg in TASKS.items():
        n_c = len(cfg["class_names"])
        mat = np.full((len(df), n_c), np.nan, dtype=np.float32)
        for fold_idx in range(5):
            chunk = np.load(OOF_DIR / f"{task}_fold{fold_idx}.npz", allow_pickle=True)
            mat[chunk["row_idx"]] = chunk["logits"]
        assert not np.isnan(mat).any(), f"{task} OOF 覆盖不完整"
        for j in range(n_c):
            wide[f"{task}_logit{j}"] = mat[:, j]

    out_parquet = OUT_DIR / "oof_predictions.parquet"
    wide.to_parquet(out_parquet, index=False)
    print(f"[OK] OOF 宽表 → {out_parquet}  ({wide.shape})")

    # ── Sample A 合并：折间平均概率 + 层级合成 ──
    sa_tasks = [t for t, c in TASKS.items() if c["sample_a"]]
    first = np.load(SA_DIR / f"{sa_tasks[0]}_fold0.npz", allow_pickle=True)
    comment_ids = first["comment_id"]
    arrays = {"comment_id": comment_ids}
    probs_mean = {}
    for task in sa_tasks:
        per_fold = []
        for fold_idx in range(5):
            f = SA_DIR / f"{task}_fold{fold_idx}.npz"
            if not f.exists():
                print(f"[warn] 缺 {f.name}，跳过 Sample A 合并")
                per_fold = None
                break
            per_fold.append(_softmax(np.load(f, allow_pickle=True)["logits"]))
        if per_fold is None:
            continue
        stacked = np.stack(per_fold)                      # [5, N, C]
        arrays[f"{task}_probs_folds"] = stacked.astype(np.float32)
        probs_mean[task] = stacked.mean(axis=0)
        arrays[f"{task}_probs_mean"] = probs_mean[task].astype(np.float32)

    if {"flat5", "stageA", "stageB", "stageC"} <= probs_mean.keys():
        # 层级五类概率：P(2)=P(pol)·P(B=2|·), P(3)=P(pol)·P(B=3|·);
        #               P(0/1/4)=P(nonpol)·P(C=·|·)
        pa, pb, pc = probs_mean["stageA"], probs_mean["stageB"], probs_mean["stageC"]
        hier = np.zeros_like(probs_mean["flat5"])
        hier[:, 2] = pa[:, 1] * pb[:, 0]
        hier[:, 3] = pa[:, 1] * pb[:, 1]
        hier[:, 0] = pa[:, 0] * pc[:, 0]
        hier[:, 1] = pa[:, 0] * pc[:, 1]
        hier[:, 4] = pa[:, 0] * pc[:, 2]
        arrays["hier_probs_mean"] = hier.astype(np.float32)

    out_npz = OUT_DIR / "sampleA_probs.npz"
    np.savez_compressed(out_npz, **arrays)
    print(f"[OK] Sample A 概率 → {out_npz}")


def main():
    ap = argparse.ArgumentParser(description="Phase 1 统一 5 折 CV（层级分类 + OOF logits）")
    ap.add_argument("--mode", required=True, choices=["cv", "merge"])
    ap.add_argument("--skip-sample-a", action="store_true", help="跳过 Sample A 全量推理")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "cv":
        mode_cv(args)
    else:
        mode_merge(args)


if __name__ == "__main__":
    main()
