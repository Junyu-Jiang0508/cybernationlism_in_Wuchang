# -*- coding: utf-8 -*-
"""
OpenAI Batch API：对 sample_B_2000.csv 做多维标注（宽定义 discourse_type 等）。

依赖: pip install openai python-dotenv pandas
环境变量: OPENAI_API_KEY

用法:
  python 09_openai_batch_annotate.py build          # 生成 batch_input.jsonl + batch_meta.json
  python 09_openai_batch_annotate.py test --n 5   # 单次 Chat Completions 测 5 条（必做）
  python 09_openai_batch_annotate.py submit       # 上传并创建 Batch，打印 batch_id
  python 09_openai_batch_annotate.py poll BATCH_ID
  python 09_openai_batch_annotate.py download BATCH_ID   # 下载输出到 batch_output.jsonl
  python 09_openai_batch_annotate.py parse      # 解析 batch_output → annotations_raw.json
  python 09_openai_batch_annotate.py retry      # 对失败批次逐条重试 + 再 parse
  python 09_openai_batch_annotate.py merge      # 合并为 annotations_merged.csv
  python 09_openai_batch_annotate.py all        # submit → poll 至完成 → download → parse → merge（耗时长）

输出目录默认: data/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = SCRIPT_DIR / "data"
BATCH_SIZE = 10
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """你是一名专业的中文计算传播学研究助理，正在执行一项关于中国网络民族主义话语的标注任务。

【任务背景】
2025年，动作RPG游戏《渊虚之羽》（武昌）发布，因开发商将清军替换为超自然瘟疫，引发了大量争议评论。这些评论涉及游戏评价、历史叙事、民族主义话语等多个层次。

【你的任务】
对每条B站评论进行多维度标注。标注采用"宽定义"原则：政治化的游戏批评（即使不包含攻击性词汇）也应被识别，只要评论涉及历史责任、文化尊重或民族情感框架。

【标注维度说明】

discourse_type（话语类型）：
- 0：纯游戏评价（仅涉及产品本身：画面/机制/剧情/音乐/bug等）
- 1：情绪化但非政治化（对游戏质量的愤怒，矛头是产品而非历史/民族）
- 2：政治化游戏批评（对历史处理/文化尊重的批评，包含道德框架，但无明显群体攻击）
- 3：显性民族主义话语（明确的他者化、群体标签、政治定罪）
- 4：中立/分析/科普（解释性、比较性，无明显情感立场）
- 99：无法判断（太短/乱码/缺乏上下文）

othering_intensity（他者化强度）：
- 0：无他者化
- 1：软性（隐含我群/他群区分）
- 2：显性（明确标记异类，有排斥无攻击）
- 3：攻击性（人身攻击/政治标签/道德定罪）

affect_intensity（情感强度）：
- 0：中性/平静
- 1：轻度（轻微不满、遗憾）
- 2：中度（明显愤怒、强烈失望）
- 3：高度（激烈愤慨、存在威胁感、道德义愤）

【边界规则，必须遵守】
1. discourse_type=2 的核心判断标准：评论是否使用了道德/责任/文化/历史框架来批评游戏？即使没有攻击性词汇也算2。
2. discourse_type=2 vs 3 的区别：攻击对象是否具体到人/群体？"删历史令人失望"→2，"支持这游戏的是汉奸"→3。
3. discourse_type和othering_intensity是独立维度，可以有 discourse_type=2 + othering_intensity=1 这样的组合。
4. 如果评论内容少于5个汉字，或无法理解，一律标99/0/0。

【输出格式】
严格输出JSON数组，包含与输入等量的对象，顺序与输入一一对应。
禁止输出任何解释、前言、后语、markdown格式或代码块标记。
每个对象只包含以下字段：
{"id": <原始id字符串>, "discourse_type": <0-4或99>, "othering_intensity": <0-3>, "affect_intensity": <0-3>}"""

FEW_SHOT_USER = """【标注示例，请学习以下案例的判断逻辑】

输入：
[
  {"id": "ex01", "text": "画面真的很漂亮，就是帧率有点问题"},
  {"id": "ex02", "text": "垃圾游戏！剧情一坨，白花我的钱"},
  {"id": "ex03", "text": "以明末为背景却删掉了清军，这是对这段历史的不尊重"},
  {"id": "ex04", "text": "为了能过审把侵略者删了，开发商的自我阉割让人寒心"},
  {"id": "ex05", "text": "买这游戏的都是数典忘祖"},
  {"id": "ex06", "text": "其实开发商删清兵是商业决策，国内市场的政策压力很大"},
  {"id": "ex07", "text": "懂的都懂，有些东西不方便说"},
  {"id": "ex08", "text": "哈哈哈好玩"},
  {"id": "ex09", "text": "国产游戏终于有这个画质了，支持"},
  {"id": "ex10", "text": "作为一个汉族人，看到这种改编我感到羞耻，先辈的血不该被这样消费"},
  {"id": "ex11", "text": "删掉清军这个决定，说明开发商根本不在乎玩家的感受，纯粹是资本媚权"},
  {"id": "ex12", "text": "好游戏，就是剧情有点短，期待DLC"}
]

输出：
[
  {"id": "ex01", "discourse_type": 0, "othering_intensity": 0, "affect_intensity": 0},
  {"id": "ex02", "discourse_type": 1, "othering_intensity": 0, "affect_intensity": 2},
  {"id": "ex03", "discourse_type": 2, "othering_intensity": 0, "affect_intensity": 1},
  {"id": "ex04", "discourse_type": 2, "othering_intensity": 0, "affect_intensity": 2},
  {"id": "ex05", "discourse_type": 3, "othering_intensity": 3, "affect_intensity": 3},
  {"id": "ex06", "discourse_type": 4, "othering_intensity": 0, "affect_intensity": 0},
  {"id": "ex07", "discourse_type": 2, "othering_intensity": 1, "affect_intensity": 1},
  {"id": "ex08", "discourse_type": 0, "othering_intensity": 0, "affect_intensity": 0},
  {"id": "ex09", "discourse_type": 0, "othering_intensity": 0, "affect_intensity": 0},
  {"id": "ex10", "discourse_type": 3, "othering_intensity": 2, "affect_intensity": 3},
  {"id": "ex11", "discourse_type": 2, "othering_intensity": 0, "affect_intensity": 2},
  {"id": "ex12", "discourse_type": 0, "othering_intensity": 0, "affect_intensity": 0}
]

【注意ex07】：隐晦表达"懂的都懂"有暗示性政治化内容，discourse_type=2，othering_intensity=1（软性），affect_intensity=1。
【注意ex10 vs ex04】：ex10有"汉族人"身份表达和道德化语气→3；ex04是对开发商行为的批评，无群体攻击→2。
【注意ex11】：批评的是"资本"和"开发商"而不是玩家群体，没有他者化→discourse_type=2，othering_intensity=0。

---
下面请对【本轮待标注评论】进行标注。
要求：只输出一个 JSON 数组，长度与输入完全一致，顺序一一对应；每个元素含 id、discourse_type、othering_intensity、affect_intensity；id 必须与输入一致（字符串）。
不要输出任何其他文字。"""


def _manual_parse_env_file(path: Path) -> bool:
    """不依赖 dotenv，直接读 .env（兼容 UTF-8 BOM、引号）。"""
    if not path.is_file():
        return False
    try:
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        text = raw.decode("utf-8")
    except Exception:
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            return False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k != "OPENAI_API_KEY" or not v:
            continue
        # 占位符不当作有效密钥（避免误用 .env.example）
        low = v.lower()
        if any(x in low for x in ("请替换", "请在这里", "your_key", "paste", "sk-...")):
            continue
        if v.startswith("sk-你的"):
            continue
        os.environ["OPENAI_API_KEY"] = v
        return True
    return False


def _load_openai_key() -> None:
    """从项目根 .env、当前工作目录 .env 加载 OPENAI_API_KEY。"""
    paths = [
        SCRIPT_DIR / ".env",
        Path.cwd() / ".env",
        SCRIPT_DIR / ".env.local",
    ]
    try:
        from dotenv import load_dotenv

        for env_path in paths:
            if env_path.is_file():
                load_dotenv(env_path, override=True)
                if os.environ.get("OPENAI_API_KEY", "").strip():
                    return
    except ImportError:
        pass
    for env_path in paths:
        if _manual_parse_env_file(env_path):
            return
    # Windows 上常见误保存为 .env.txt
    for ext in (".env.txt", "env.txt"):
        p = SCRIPT_DIR / ext
        if p.is_file() and _manual_parse_env_file(p):
            return


def _print_env_diagnostics() -> None:
    root = SCRIPT_DIR
    env_f = root / ".env"
    env_txt = root / ".env.txt"
    ex = root / ".env.example"
    print("\n【诊断】", file=sys.stderr)
    print(f"  .env 是否存在: {env_f.is_file()}  ← 路径: {env_f}", file=sys.stderr)
    print(f"  .env.txt 是否存在（应改名为 .env）: {env_txt.is_file()}", file=sys.stderr)
    print(f"  .env.example 是否存在: {ex.is_file()}", file=sys.stderr)
    if not env_f.is_file() and ex.is_file():
        print(
            "\n  你很可能还没创建 .env。在 PowerShell 中执行：\n"
            f"    Copy-Item \"{ex}\" \"{env_f}\"\n"
            "  然后用记事本打开 .env，把 sk-你的密钥 改成真实 Key（保存为 UTF-8）。",
            file=sys.stderr,
        )


def _client():
    try:
        from openai import OpenAI
    except ImportError:
        print("请安装: pip install openai", file=sys.stderr)
        sys.exit(1)
    _load_openai_key()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(
            "未检测到 OPENAI_API_KEY。\n"
            "请任选其一：\n"
            f"  1) 在目录创建文件 .env（无后缀！），内容一行：OPENAI_API_KEY=sk-...\n"
            f"     完整路径: {SCRIPT_DIR / '.env'}\n"
            "  2) PowerShell: $env:OPENAI_API_KEY='sk-...'\n"
            "  3) 系统环境变量里设置 OPENAI_API_KEY\n"
            "  （python-dotenv 可选；无安装时也会尝试直接读取 .env）",
            file=sys.stderr,
        )
        _print_env_diagnostics()
        sys.exit(1)
    return OpenAI(api_key=key)


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```\s*$", "", s)
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j <= i:
        raise ValueError("未找到 JSON 数组")
    arr = json.loads(s[i : j + 1])
    if not isinstance(arr, list):
        raise ValueError("根节点不是数组")
    return arr


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x.strip())
    return None


def validate_ann(item: Dict[str, Any]) -> bool:
    try:
        dt = _as_int(item.get("discourse_type"))
        oi = _as_int(item.get("othering_intensity"))
        ai = _as_int(item.get("affect_intensity"))
        if item.get("id") is None:
            return False
        if dt not in (0, 1, 2, 3, 4, 99):
            return False
        if oi not in (0, 1, 2, 3) or ai not in (0, 1, 2, 3):
            return False
        return True
    except Exception:
        return False


def build_user_content(batch: List[Dict[str, str]]) -> str:
    payload = json.dumps(batch, ensure_ascii=False)
    return FEW_SHOT_USER + "\n\n【本轮待标注评论】\n" + payload


def cmd_build(args: argparse.Namespace) -> None:
    path = Path(args.sample)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "content_clean" not in df.columns:
        print("缺少列 content_clean", file=sys.stderr)
        sys.exit(1)
    df["comment_id"] = df["comment_id"].astype(str)
    df["content_clean"] = df["content_clean"].fillna("").astype(str)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "batch_input.jsonl"
    meta_path = out_dir / "batch_meta.json"

    rows: List[Tuple[str, str]] = list(
        zip(df["comment_id"].tolist(), df["content_clean"].tolist())
    )
    batches: Dict[str, List[Dict[str, str]]] = {}
    n_batch = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), BATCH_SIZE):
            chunk = rows[i : i + BATCH_SIZE]
            bid = f"batch_{n_batch:03d}"
            n_batch += 1
            batch_objs = [{"id": cid, "text": txt[:8000]} for cid, txt in chunk]
            batches[bid] = batch_objs
            body = {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 1200,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_content(batch_objs)},
                ],
            }
            line = {
                "custom_id": bid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"batch_size": BATCH_SIZE, "batches": batches}, f, ensure_ascii=False, indent=2)

    print(f"已写 {jsonl_path}（{n_batch} 条请求）与 {meta_path}")


def cmd_test(args: argparse.Namespace) -> None:
    path = Path(args.sample)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.head(args.n)
    batch = [
        {"id": str(r["comment_id"]), "text": str(r["content_clean"])[:8000]}
        for _, r in df.iterrows()
    ]
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(batch)},
        ],
    )
    content = resp.choices[0].message.content or ""
    print("--- 模型原始输出 ---")
    print(content)
    print("--- 解析结果 ---")
    try:
        arr = extract_json_array(content)
        print(json.dumps(arr, ensure_ascii=False, indent=2))
        if len(arr) != len(batch):
            print(f"[警告] 条数不一致: 期望 {len(batch)} 得到 {len(arr)}", file=sys.stderr)
    except Exception as e:
        print(f"解析失败: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_submit(args: argparse.Namespace) -> None:
    client = _client()
    jsonl_path = Path(args.outdir) / "batch_input.jsonl"
    if not jsonl_path.exists():
        print("请先运行 build", file=sys.stderr)
        sys.exit(1)
    with open(jsonl_path, "rb") as f:
        up = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    id_path = Path(args.outdir) / "batch_job_id.txt"
    id_path.write_text(batch.id, encoding="utf-8")
    print(f"batch_id={batch.id}")
    print(f"已写入 {id_path}")


def cmd_poll(args: argparse.Namespace) -> None:
    client = _client()
    bid = args.batch_id or (Path(args.outdir) / "batch_job_id.txt").read_text(encoding="utf-8").strip()
    interval = max(60, args.interval)
    while True:
        b = client.batches.retrieve(bid)
        print(f"{time.strftime('%H:%M:%S')} status={b.status} completed={b.request_counts.completed if b.request_counts else 0}/{b.request_counts.total if b.request_counts else '?'}")
        if b.status in ("completed", "failed", "expired", "cancelled"):
            if b.status != "completed":
                print(f"Batch 结束状态: {b.status}", file=sys.stderr)
            if b.output_file_id:
                (args.outdir / "batch_output_file_id.txt").write_text(b.output_file_id, encoding="utf-8")
            if b.error_file_id:
                (args.outdir / "batch_error_file_id.txt").write_text(b.error_file_id, encoding="utf-8")
            break
        time.sleep(interval)


def cmd_download(args: argparse.Namespace) -> None:
    client = _client()
    out_dir = Path(args.outdir)
    bid = args.batch_id or (out_dir / "batch_job_id.txt").read_text(encoding="utf-8").strip()
    b = client.batches.retrieve(bid)
    if b.status != "completed":
        print(f"Batch 未 completed: {b.status}", file=sys.stderr)
        sys.exit(1)
    if not b.output_file_id:
        print("无 output_file_id", file=sys.stderr)
        sys.exit(1)
    content = client.files.content(b.output_file_id)
    out_path = out_dir / "batch_output.jsonl"
    out_path.write_bytes(content.read())
    print(f"已下载 → {out_path}")
    if b.error_file_id:
        err = client.files.content(b.error_file_id)
        (out_dir / "batch_errors.jsonl").write_bytes(err.read())
        print("已下载 batch_errors.jsonl")


def _extract_batch_annotations(
    rec: Dict[str, Any],
    batch_items: List[Dict[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """解析单条 batch 记录；任一校验环节失败返回 None（整批视为失败）。"""
    if rec.get("error"):
        return None
    resp = rec.get("response") or {}
    if resp.get("status_code") and resp["status_code"] != 200:
        return None
    body = resp.get("body") or {}
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            body = {}
    choices = (body.get("choices") or [{}])[0]
    content = (choices.get("message") or {}).get("content") or ""
    try:
        arr = extract_json_array(content)
    except Exception:
        return None
    if len(arr) != len(batch_items):
        return None
    by_id = {str(x.get("id")): x for x in arr}
    for item in batch_items:
        row = by_id.get(str(item["id"]))
        if not row or not validate_ann(row):
            return None
    return {
        str(item["id"]): {
            "discourse_type": _as_int(item["discourse_type"]),
            "othering_intensity": _as_int(item["othering_intensity"]),
            "affect_intensity": _as_int(item["affect_intensity"]),
        }
        for item in arr
    }


def parse_output_lines(
    lines: List[str],
    meta: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
    """返回 (comment_id -> ann, failed_custom_ids, failed_comment_ids)"""
    ann: Dict[str, Dict[str, Any]] = {}
    failed_cids: List[str] = []
    failed_custom: List[str] = []
    batches = meta.get("batches", {})

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = rec.get("custom_id", "")
        batch_items = batches.get(custom_id, [])
        parsed = _extract_batch_annotations(rec, batch_items)
        if parsed is None:
            failed_custom.append(custom_id)
            failed_cids.extend(str(x["id"]) for x in batch_items)
            continue
        ann.update(parsed)

    return ann, list(set(failed_custom)), list(set(failed_cids))


def cmd_parse(args: argparse.Namespace) -> None:
    out_dir = Path(args.outdir)
    out_path = out_dir / "batch_output.jsonl"
    if not out_path.exists():
        print(f"缺少 {out_path}，先 download", file=sys.stderr)
        sys.exit(1)
    meta = json.loads((out_dir / "batch_meta.json").read_text(encoding="utf-8"))
    lines = out_path.read_text(encoding="utf-8").splitlines()
    ann, failed_custom, failed_cids = parse_output_lines(lines, meta)

    raw_path = out_dir / "annotations_raw.json"
    raw_path.write_text(
        json.dumps(list({"comment_id": k, **v} for k, v in sorted(ann.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "failed_custom_ids.txt").write_text("\n".join(sorted(failed_custom)), encoding="utf-8")
    (out_dir / "failed_ids.txt").write_text("\n".join(sorted(failed_cids)), encoding="utf-8")
    print(f"成功解析 {len(ann)} 条；失败批次 {len(failed_custom)}；失败 comment_id {len(failed_cids)}")
    print(f"→ {raw_path}")


def cmd_retry(args: argparse.Namespace) -> None:
    """对 failed_ids 逐条调用 Chat Completions"""
    client = _client()
    out_dir = Path(args.outdir)
    meta = json.loads((out_dir / "batch_meta.json").read_text(encoding="utf-8"))
    batches = meta["batches"]
    failed_custom = (out_dir / "failed_custom_ids.txt").read_text(encoding="utf-8").strip().split()
    if not failed_custom or failed_custom == [""]:
        print("无失败批次")
        return

    recovered: Dict[str, Dict[str, Any]] = {}
    still_failed: List[str] = []

    for cid_custom in failed_custom:
        items = batches.get(cid_custom, [])
        for it in items:
            one = [it]
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    temperature=0,
                    max_tokens=400,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_content(one)},
                    ],
                )
                content = resp.choices[0].message.content or ""
                arr = extract_json_array(content)
                if len(arr) == 1 and validate_ann(arr[0]) and str(arr[0]["id"]) == str(it["id"]):
                    recovered[str(it["id"])] = {
                        "discourse_type": _as_int(arr[0]["discourse_type"]),
                        "othering_intensity": _as_int(arr[0]["othering_intensity"]),
                        "affect_intensity": _as_int(arr[0]["affect_intensity"]),
                    }
                else:
                    still_failed.append(str(it["id"]))
            except Exception:
                still_failed.append(str(it["id"]))
            time.sleep(float(args.sleep))

    # 合并进 annotations_raw
    raw_path = out_dir / "annotations_raw.json"
    if raw_path.exists():
        existing = {
            str(x["comment_id"]): {
                "discourse_type": x["discourse_type"],
                "othering_intensity": x["othering_intensity"],
                "affect_intensity": x["affect_intensity"],
            }
            for x in json.loads(raw_path.read_text(encoding="utf-8"))
        }
    else:
        existing = {}
    existing.update(recovered)
    for fid in still_failed:
        existing[fid] = {"discourse_type": 99, "othering_intensity": 0, "affect_intensity": 0}

    raw_path.write_text(
        json.dumps(
            [{"comment_id": k, **v} for k, v in sorted(existing.items())],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "failed_ids.txt").write_text("\n".join(sorted(set(still_failed))), encoding="utf-8")
    print(f"重试恢复 {len(recovered)} 条；仍失败已标 99/0/0: {len(still_failed)} → {raw_path}")


def cmd_merge(args: argparse.Namespace) -> None:
    out = Path(args.outdir)
    raw_path = out / "annotations_raw.json"
    if not raw_path.is_file():
        print("找不到 annotations_raw.json，无法 merge。", file=sys.stderr)
        if (out / "batch_output.jsonl").is_file():
            print("你已下载 Batch 输出，请先运行:", file=sys.stderr)
            print("  python 09_openai_batch_annotate.py parse", file=sys.stderr)
        else:
            print(
                "完整顺序: submit → poll → download → parse →（可选 retry）→ merge",
                file=sys.stderr,
            )
        sys.exit(1)
    sample = pd.read_csv(Path(args.sample), encoding="utf-8-sig")
    sample["comment_id"] = sample["comment_id"].astype(str)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    ann_df = pd.DataFrame(raw)
    ann_df = ann_df.rename(columns={"comment_id": "comment_id"})
    merged = sample.merge(ann_df, on="comment_id", how="left")
    out_path = Path(args.outdir) / "annotations_merged.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    miss = merged["discourse_type"].isna().sum()
    print(f"已写 {out_path}；未匹配标注: {miss}")


def cmd_validate_report(args: argparse.Namespace) -> None:
    """分布 + keyword_hit 交叉（需 merge 后或 annotations_raw + sample）"""
    path = Path(args.outdir) / "annotations_merged.csv"
    if not path.exists():
        print("请先 merge", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, encoding="utf-8-sig")
    dt = df["discourse_type"].value_counts(normalize=True).sort_index()
    print("=== discourse_type 分布 ===")
    print((dt * 100).round(2).to_string())
    p23 = df["discourse_type"].isin([2, 3]).mean()
    print(f"\n2+3 合计: {p23:.1%}")
    if "keyword_hit" in df.columns:
        kh1 = df["keyword_hit"] == 1
        kh0 = df["keyword_hit"] == 0
        pol = df["discourse_type"].isin([2, 3])
        print("\n=== keyword_hit 交叉 ===")
        if kh1.any():
            print(f"keyword_hit=1 中 标为 2或3: {(pol & kh1).sum() / kh1.sum():.1%}")
        if kh0.any():
            print(f"keyword_hit=0 中 标为 2或3: {(pol & kh0).sum() / kh0.sum():.1%}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_build(args)
    cmd_submit(args)
    args.batch_id = None
    args.interval = args.interval
    cmd_poll(args)
    b = _client().batches.retrieve(
        (Path(args.outdir) / "batch_job_id.txt").read_text(encoding="utf-8").strip()
    )
    if b.status != "completed":
        sys.exit(1)
    cmd_download(argparse.Namespace(outdir=args.outdir, batch_id=None))
    cmd_parse(args)
    fc_path = args.outdir / "failed_custom_ids.txt"
    if fc_path.exists() and fc_path.read_text(encoding="utf-8").strip():
        cmd_retry(args)
    cmd_merge(args)
    cmd_validate_report(args)


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI Batch 宽定义标注")
    ap.add_argument("--outdir", type=Path, default=DATA_DIR)
    ap.add_argument("--sample", type=Path, default=DATA_DIR / "sample_B_2000.csv")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("build", help="生成 batch_input.jsonl")
    p1.set_defaults(func=cmd_build)

    p2 = sub.add_parser("test", help="非 Batch 测 n 条")
    p2.add_argument("--n", type=int, default=5)
    p2.set_defaults(func=cmd_test)

    p3 = sub.add_parser("submit", help="上传并创建 Batch")
    p3.set_defaults(func=cmd_submit)

    p4 = sub.add_parser("poll", help="轮询 Batch 状态")
    p4.add_argument("batch_id", nargs="?", default=None)
    p4.add_argument("--interval", type=int, default=300)
    p4.set_defaults(func=cmd_poll)

    p5 = sub.add_parser("download", help="下载 batch 输出")
    p5.add_argument("batch_id", nargs="?", default=None)
    p5.set_defaults(func=cmd_download)

    p6 = sub.add_parser("parse", help="解析 batch_output.jsonl")
    p6.set_defaults(func=cmd_parse)

    p7 = sub.add_parser("retry", help="失败批次逐条重试")
    p7.add_argument("--sleep", type=float, default=0.5)
    p7.set_defaults(func=cmd_retry)

    p8 = sub.add_parser("merge", help="合并 annotations_merged.csv")
    p8.set_defaults(func=cmd_merge)

    p9 = sub.add_parser("report", help="分布与 keyword 交叉验证")
    p9.set_defaults(func=cmd_validate_report)

    pa = sub.add_parser("all", help="build+submit+poll+download+parse+merge（长时间）")
    pa.add_argument("--interval", type=int, default=300)
    pa.set_defaults(func=cmd_all)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
