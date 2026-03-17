# -*- coding: utf-8 -*-
"""
B 站视频字幕获取 + 无字幕时音频下载与 Whisper 转写。
移植自 questionmark_danmaku 的 04_get_subtitles.py 与 05_whisper_transcriber.py，
输入为本项目 03_run_bilibili_crawler 产出的视频列表 CSV。

前置条件：
  1. 已运行 03_run_bilibili_crawler.py，得到 00_output/01_Raw_data/02_Bilibili/csv/search_videos_all.csv
  2. 依赖：pip install requests openai-whisper yt-dlp；系统已安装 ffmpeg
  3. GPU 加速（可选）：已安装带 CUDA 的 PyTorch 时，Whisper 会自动用 GPU，转写更快；加 --no-gpu 可强制用 CPU
  4. Cookie：环境变量 BILIBILI_COOKIE 或项目根目录 bilibili_cookie.txt（可选，部分接口需登录）

用法：
  python 04_bilibili_subtitle_and_transcript.py                    # 默认从 02_Bilibili/csv 找视频列表
  python 04_bilibili_subtitle_and_transcript.py --csv path/to.csv   # 指定视频列表 CSV
  python 04_bilibili_subtitle_and_transcript.py --subtitle-only     # 仅抓字幕，不跑 Whisper
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_RAW_DIR = PROJECT_ROOT / "00_output" / "01_Raw_data" / "02_Bilibili"
DEFAULT_CSV_DIR = DEFAULT_RAW_DIR / "csv"
DEFAULT_SUBTITLE_DIR = DEFAULT_RAW_DIR / "subtitles"
DEFAULT_TRANSCRIPT_DIR = DEFAULT_RAW_DIR / "transcripts"
DEFAULT_TRANSCRIPT_JSON_DIR = DEFAULT_RAW_DIR / "transcripts_json"
DEFAULT_AUDIO_DIR = DEFAULT_RAW_DIR / "audio"

DELAY_MIN, DELAY_MAX = 1, 3
WHISPER_MODEL = "base"  # tiny | base | small | medium | large
MAX_RETRIES = 3
DELETE_AUDIO_AFTER_TRANSCRIPT = True

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_cookie() -> str:
    cookie = os.environ.get("BILIBILI_COOKIE", "").strip()
    if cookie:
        return cookie
    cookie_file = PROJECT_ROOT / "bilibili_cookie.txt"
    if cookie_file.exists():
        return cookie_file.read_text(encoding="utf-8").strip()
    return ""


def _load_video_list_from_csv(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 兼容：video_id 或 aid
    if "video_id" not in df.columns and "aid" in df.columns:
        df["video_id"] = df["aid"].astype(str)
    if "video_id" not in df.columns:
        raise ValueError(f"CSV 需包含 video_id 列: {csv_path}")
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["video_id"])
    rows = []
    for _, row in df.iterrows():
        aid = row["video_id"]
        title = str(row.get("title", ""))
        video_url = row.get("video_url", "") or row.get("url", "")
        if not video_url:
            video_url = f"https://www.bilibili.com/video/av{aid}"
        rows.append({"aid": aid, "title": title, "video_url": video_url.strip()})
    return rows


def _find_videos_csv(csv_dir: Path) -> Path | None:
    # 优先 search_videos_all.csv（03 脚本产出）
    candidates = list(csv_dir.glob("search_videos_all.csv"))
    if not candidates:
        candidates = list(csv_dir.glob("*_videos_*.csv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


# ---------------------------------------------------------------------------
# 从 aid 获取 bvid 和 cid（B 站 API，避免爬页面）
# ---------------------------------------------------------------------------
def get_bvid_cid_from_aid(aid: str, headers: dict) -> tuple[str | None, str | None]:
    try:
        url = "https://api.bilibili.com/x/web-interface/view"
        r = requests.get(url, params={"aid": aid}, headers=headers, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            logger.warning("view API error: %s", data.get("message"))
            return None, None
        d = data.get("data", {})
        bvid = d.get("bvid") or d.get("BV")
        if not bvid:
            return None, None
        cid = d.get("cid")
        if cid is None and d.get("pages"):
            cid = d["pages"][0].get("cid")
        if cid is not None:
            cid = str(cid)
        return bvid, cid
    except Exception as e:
        logger.warning("get_bvid_cid_from_aid %s: %s", aid, e)
        return None, None


# ---------------------------------------------------------------------------
# 获取官方字幕（与 questionmark_danmaku 04 一致）
# ---------------------------------------------------------------------------
def get_bilibili_subtitle(bvid: str, cid: str, headers: dict) -> list | None:
    try:
        url = "https://api.bilibili.com/x/player/v2"
        r = __import__("requests").get(
            url, params={"bvid": bvid, "cid": cid}, headers=headers, timeout=15
        )
        data = r.json()
        if data.get("code") != 0:
            return None
        sub_info = data.get("data", {}).get("subtitle", {})
        sub_list = sub_info.get("subtitles", [])
        if not sub_list:
            return None
        sub_url = sub_list[0].get("subtitle_url", "")
        if not sub_url.startswith("http"):
            sub_url = "https:" + sub_url
        sub_r = requests.get(sub_url, timeout=15)
        sub_data = sub_r.json()
        return sub_data.get("body", [])
    except Exception as e:
        logger.warning("get_bilibili_subtitle %s: %s", bvid, e)
        return None


def subtitle_to_text(subtitles: list) -> str:
    if not subtitles:
        return ""
    lines = [s.get("content", "").strip() for s in subtitles if s.get("content")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whisper：下载音频 + 转写（与 questionmark_danmaku 05 一致）
# ---------------------------------------------------------------------------
def check_yt_dlp_ffmpeg() -> bool:
    for cmd in ("yt-dlp", "ffmpeg"):
        try:
            subprocess.run(
                [cmd, "--version"] if cmd == "yt-dlp" else [cmd, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except FileNotFoundError:
            logger.error("未找到 %s，请安装：pip install yt-dlp；ffmpeg 需系统安装", cmd)
            return False
    return True


def check_whisper() -> bool:
    try:
        __import__("whisper")
        __import__("torch")
    except Exception as e:
        logger.error("未找到 openai-whisper 或 torch：pip install openai-whisper torch，错误: %s", e)
        return False
    return True


def download_audio_bvid(bvid: str, video_url: str, audio_dir: Path) -> Path | None:
    out_tpl = str(audio_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3", "-o", out_tpl, video_url
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("下载音频 %s（第 %d/%d 次）", bvid, attempt, MAX_RETRIES)
            before = set(f.name for f in audio_dir.iterdir()) if audio_dir.exists() else set()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                logger.warning("yt-dlp 失败: %s", (proc.stderr or "")[:400])
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                continue
            # 查找刚生成的文件：先按 bvid/av 号匹配，否则取目录中最新的音频
            for ext in ("mp3", "m4a", "webm"):
                for candidate in (bvid, bvid.replace("BV", ""), bvid.lstrip("av")):
                    p = audio_dir / f"{candidate}.{ext}"
                    if p.exists():
                        return p
            after = list(audio_dir.iterdir()) if audio_dir.exists() else []
            new_files = [f for f in after if f.name not in before and f.suffix.lower() in (".mp3", ".m4a", ".webm")]
            if new_files:
                return max(new_files, key=lambda f: f.stat().st_mtime)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp 超时")
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            logger.warning("下载异常: %s", e)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return None


def _whisper_device(use_gpu: bool = True) -> str:
    """优先使用 GPU（CUDA），不可用时回退到 CPU。"""
    if not use_gpu:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_whisper(audio_path: Path, model_name: str, device: str | None = None) -> tuple[str, list]:
    import whisper
    if device is None:
        device = _whisper_device()
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(audio_path),
        language="zh",
        verbose=False,
        initial_prompt="以下是一段中文视频的逐字转录。",
    )
    segments = result.get("segments", [])
    lines = []
    subtitles = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text:
            lines.append(text)
            subtitles.append({
                "from": seg.get("start", 0),
                "to": seg.get("end", 0),
                "content": text,
                "location": 2,
            })
    return "\n".join(lines).strip(), subtitles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="B 站字幕获取 + 无字幕时 Whisper 转写")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="视频列表 CSV 路径（默认从 02_Bilibili/csv 自动查找）",
    )
    parser.add_argument(
        "--subtitle-only",
        action="store_true",
        help="仅抓官方字幕，不执行 Whisper 转写",
    )
    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help="Whisper 模型: tiny/base/small/medium/large",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="强制使用 CPU 跑 Whisper（默认：有 CUDA 时用 GPU 加速）",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = _find_videos_csv(DEFAULT_CSV_DIR)
    if csv_path is None or not csv_path.exists():
        logger.error(
            "未找到视频列表 CSV。请先运行 03_run_bilibili_crawler.py，或使用 --csv 指定路径。"
        )
        sys.exit(1)

    video_list = _load_video_list_from_csv(csv_path)
    logger.info("加载视频列表: %s，共 %d 条", csv_path, len(video_list))

    cookie = _get_cookie()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    }
    if cookie:
        headers["Cookie"] = cookie
    else:
        logger.warning("未设置 BILIBILI_COOKIE，部分接口可能受限。可设置环境变量或 bilibili_cookie.txt")

    DEFAULT_SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    need_whisper = []

    # ---------- 阶段 1：官方字幕 ----------
    for i, item in enumerate(video_list):
        aid, title, video_url = item["aid"], item["title"], item["video_url"]
        logger.info("[%d/%d] %s | %s", i + 1, len(video_list), aid, (title or "")[:50])
        bvid, cid = get_bvid_cid_from_aid(aid, headers)
        if not bvid or not cid:
            logger.warning("  无法获取 bvid/cid，加入 Whisper 队列")
            need_whisper.append({**item, "bvid": aid})
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue
        item["bvid"] = bvid
        item["cid"] = cid
        subs = get_bilibili_subtitle(bvid, cid, headers)
        if subs:
            json_path = DEFAULT_SUBTITLE_DIR / f"{bvid}_subtitle.json"
            txt_path = DEFAULT_SUBTITLE_DIR / f"{bvid}_subtitle.txt"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(subs, f, ensure_ascii=False, indent=2)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(subtitle_to_text(subs))
            logger.info("  已保存官方字幕: %s", json_path.name)
        else:
            need_whisper.append(item)
            logger.info("  无官方字幕，将使用 Whisper 转写")
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    if args.subtitle_only:
        logger.info("仅字幕模式，跳过 Whisper。无字幕视频数: %d", len(need_whisper))
        return

    if not need_whisper:
        logger.info("所有视频均有官方字幕，无需 Whisper。")
        return

    # ---------- 阶段 2：Whisper 转写 ----------
    if not check_yt_dlp_ffmpeg() or not check_whisper():
        logger.error("缺少 yt-dlp/ffmpeg 或 whisper，请安装后重试。")
        sys.exit(1)

    whisper_device = _whisper_device(use_gpu=not args.no_gpu)
    logger.info("Whisper 使用设备: %s（可用 GPU 时转写会快很多）", whisper_device)

    DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_TRANSCRIPT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(need_whisper):
        bvid = item.get("bvid") or item["aid"]
        title = item["title"]
        video_url = item["video_url"]
        logger.info("[Whisper %d/%d] %s | %s", i + 1, len(need_whisper), bvid, (title or "")[:50])
        audio_path = download_audio_bvid(bvid, video_url, DEFAULT_AUDIO_DIR)
        if not audio_path or not audio_path.exists():
            logger.warning("  音频下载失败，跳过")
            continue
        try:
            plain_text, subs = transcribe_whisper(audio_path, args.whisper_model, device=whisper_device)
        except Exception as e:
            logger.warning("  转写失败: %s", e)
            continue
        txt_path = DEFAULT_TRANSCRIPT_DIR / f"{bvid}.txt"
        json_path = DEFAULT_TRANSCRIPT_JSON_DIR / f"{bvid}_subtitle.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(subs, f, ensure_ascii=False, indent=2)
        logger.info("  已保存: %s, %s", txt_path.name, json_path.name)
        if DELETE_AUDIO_AFTER_TRANSCRIPT:
            try:
                audio_path.unlink()
            except Exception:
                pass
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    logger.info("全部完成。字幕目录: %s；转写目录: %s", DEFAULT_SUBTITLE_DIR, DEFAULT_TRANSCRIPT_DIR)


if __name__ == "__main__":
    main()
