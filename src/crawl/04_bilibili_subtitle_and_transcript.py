# -*- coding: utf-8 -*-
"""
B 站视频字幕获取 + 无字幕时音频下载与 Whisper 转写。
移植自 questionmark_danmaku 的 04_get_subtitles.py 与 05_whisper_transcriber.py，
输入为视频列表 CSV（默认优先 01_data/search_videos_all.csv，否则 03 爬虫产出目录）。

前置条件：
  1. 视频名单：01_data/search_videos_all.csv，或已运行 03_run_bilibili_crawler.py 得到 00_output/.../search_videos_all.csv
  2. 依赖：pip install requests openai-whisper yt-dlp；系统已安装 ffmpeg
  3. GPU 加速（可选）：已安装带 CUDA 的 PyTorch 时，Whisper 会自动用 GPU，转写更快；加 --no-gpu 可强制用 CPU
  4. Cookie：环境变量 BILIBILI_COOKIE 或项目根目录 bilibili_cookie.txt（可选，部分接口需登录）

用法：
  python 04_bilibili_subtitle_and_transcript.py                    # 默认优先 01_data/search_videos_all.csv
  python 04_bilibili_subtitle_and_transcript.py --csv path/to.csv   # 指定视频列表 CSV
  python 04_bilibili_subtitle_and_transcript.py --subtitle-only     # 仅抓字幕，不跑 Whisper
  python 04_bilibili_subtitle_and_transcript.py --keep-audio        # 转写后保留音频文件
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SCRIPT_DIR
DEFAULT_RAW_DIR = PROJECT_ROOT / "00_output" / "01_Raw_data" / "02_Bilibili"
DEFAULT_CSV_DIR = DEFAULT_RAW_DIR / "csv"
DEFAULT_VIDEOS_CSV = PROJECT_ROOT / "01_data" / "search_videos_all.csv"
DEFAULT_SUBTITLE_DIR = DEFAULT_RAW_DIR / "subtitles"
DEFAULT_TRANSCRIPT_DIR = DEFAULT_RAW_DIR / "transcripts"
DEFAULT_TRANSCRIPT_JSON_DIR = DEFAULT_RAW_DIR / "transcripts_json"
DEFAULT_AUDIO_DIR = DEFAULT_RAW_DIR / "audio"

DELAY_MIN, DELAY_MAX = 1, 3
WHISPER_MODEL = "base"  # tiny | base | small | medium | large
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class VideoItem:
    aid: str
    title: str
    video_url: str
    bvid: str = field(default="")
    cid: str = field(default="")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _get_cookie() -> str:
    cookie = os.environ.get("BILIBILI_COOKIE", "").strip()
    if cookie:
        return cookie
    cookie_file = PROJECT_ROOT / "bilibili_cookie.txt"
    if cookie_file.exists():
        return cookie_file.read_text(encoding="utf-8").strip()
    return ""


def _normalize_bilibili_aid(raw) -> str:
    """
    CSV 常把大数字 aid 存成浮点，读成 "114946679509769.0" 或科学计数法 "1.1493e+14"，
    view 接口收到非整数字符串会报「请求错误」。
    策略：先尝试字符串分割（无精度损失），失败则用 Decimal 兜底。
    """
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return ""
    # 纯整数，直接返回
    if s.isdigit():
        return s
    # 科学计数法（含 e/E）：Decimal 可精确还原大整数
    if "e" in s.lower():
        try:
            from decimal import Decimal, ROUND_DOWN
            d = Decimal(s).to_integral_value(rounding=ROUND_DOWN)
            result = str(d)
            return result if result.lstrip("-").isdigit() else ""
        except Exception:
            return ""
    # 含小数点：字符串分割，不经过 float
    if "." in s:
        whole, frac = s.split(".", 1)
        if whole.lstrip("-").isdigit() and not frac.strip("0"):
            return whole
        return ""
    return ""


def _avid_from_url(url: str) -> str | None:
    if not url or not isinstance(url, str):
        return None
    m = re.search(r"/video/(?:av)?(\d{6,})", url, re.I)
    if m:
        return m.group(1)
    m = re.search(r"[?&]aid=(\d+)", url, re.I)
    return m.group(1) if m else None


def _bvid_from_url(url: str) -> str | None:
    """BV 号是大小写敏感的 Base58 编码，不使用 re.I。"""
    if not url:
        return None
    m = re.search(r"(BV[\w]+)\b", url)
    return m.group(1) if m else None


def _load_video_list_from_csv(csv_path: Path) -> list[VideoItem]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    if "video_id" not in df.columns and "aid" in df.columns:
        df["video_id"] = df["aid"]
    if "video_id" not in df.columns:
        raise ValueError(f"CSV 需包含 video_id 列: {csv_path}")
    df["video_id"] = df["video_id"].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["video_id"])
    rows: list[VideoItem] = []
    for _, row in df.iterrows():
        aid = _normalize_bilibili_aid(row["video_id"])
        title = str(row.get("title", "") or "")
        video_url = str(row.get("video_url", "") or row.get("url", "") or "").strip()
        if not aid and video_url:
            aid = _avid_from_url(video_url) or aid
        if not video_url and aid:
            video_url = f"https://www.bilibili.com/video/av{aid}"
        rows.append(VideoItem(aid=aid, title=title, video_url=video_url))
    return rows


def _find_videos_csv(csv_dir: Path) -> Path | None:
    candidates = list(csv_dir.glob("search_videos_all.csv"))
    if not candidates:
        candidates = list(csv_dir.glob("*_videos_*.csv"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _resolve_default_videos_csv() -> Path | None:
    """优先 01_data/search_videos_all.csv，否则 00_output/.../02_Bilibili/csv。"""
    if DEFAULT_VIDEOS_CSV.exists():
        return DEFAULT_VIDEOS_CSV
    return _find_videos_csv(DEFAULT_CSV_DIR)


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


# ---------------------------------------------------------------------------
# 依赖检测
# ---------------------------------------------------------------------------
def check_yt_dlp_ffmpeg() -> bool:
    for cmd in ("yt-dlp", "ffmpeg"):
        try:
            result = subprocess.run(
                [cmd, "--version"] if cmd == "yt-dlp" else [cmd, "-version"],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error("命令 %s 返回非零退出码，请检查安装", cmd)
                return False
        except FileNotFoundError:
            logger.error("未找到 %s，请安装：pip install yt-dlp；ffmpeg 需系统安装", cmd)
            return False
    return True


def check_whisper() -> bool:
    try:
        import whisper  # noqa: F401
        import torch    # noqa: F401
        return True
    except ImportError as e:
        logger.error("未找到 openai-whisper 或 torch：pip install openai-whisper torch，错误: %s", e)
        return False


# ---------------------------------------------------------------------------
# 从 aid 获取 bvid 和 cid（B 站 API，带重试）
# ---------------------------------------------------------------------------
def get_bvid_cid_from_aid(
    aid: str,
    session: requests.Session,
    video_url: str = "",
    retries: int = MAX_RETRIES,
) -> tuple[str | None, str | None]:
    api = "https://api.bilibili.com/x/web-interface/view"
    params: dict = {}
    bvid_hint = _bvid_from_url(video_url)
    aid_clean = _normalize_bilibili_aid(aid) if aid else ""
    if bvid_hint:
        params["bvid"] = bvid_hint
    elif aid_clean:
        params["aid"] = aid_clean
    else:
        return None, None

    for attempt in range(1, retries + 1):
        try:
            r = session.get(api, params=params, timeout=15)
            data = r.json()
            if data.get("code") != 0:
                logger.warning(
                    "view API error: %s (aid=%s bvid=%s)",
                    data.get("message"),
                    aid_clean or "-",
                    bvid_hint or "-",
                )
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
            logger.warning("get_bvid_cid_from_aid %s（第 %d 次）: %s", aid, attempt, e)
            if attempt < retries:
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return None, None


# ---------------------------------------------------------------------------
# 获取官方字幕（带重试）
# ---------------------------------------------------------------------------
def get_bilibili_subtitle(
    bvid: str,
    cid: str,
    session: requests.Session,
    retries: int = MAX_RETRIES,
) -> list | None:
    url = "https://api.bilibili.com/x/player/v2"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params={"bvid": bvid, "cid": cid}, timeout=15)
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
            sub_r = session.get(sub_url, timeout=15)
            sub_data = sub_r.json()
            return sub_data.get("body", [])
        except Exception as e:
            logger.warning("get_bilibili_subtitle %s（第 %d 次）: %s", bvid, attempt, e)
            if attempt < retries:
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return None


def subtitle_to_text(subtitles: list) -> str:
    if not subtitles:
        return ""
    lines = [s.get("content", "").strip() for s in subtitles if s.get("content")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Whisper：下载音频 + 转写
# ---------------------------------------------------------------------------
def download_audio_bvid(bvid: str, video_url: str, audio_dir: Path) -> Path | None:
    out_tpl = str(audio_dir / "%(id)s.%(ext)s")
    # --print after_move:filepath 在所有后处理完成后直接输出最终路径，避免目录 diff 竞态
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--print", "after_move:filepath",
        "-o", out_tpl,
        video_url,
    ]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("下载音频 %s（第 %d/%d 次）", bvid, attempt, MAX_RETRIES)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                logger.warning("yt-dlp 失败: %s", (proc.stderr or "")[:400])
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                continue
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            if lines:
                audio_path = Path(lines[-1])
                if audio_path.exists():
                    return audio_path
            logger.warning("yt-dlp 未返回有效路径，stdout: %s", proc.stdout[:200])
            return None
        except subprocess.TimeoutExpired:
            logger.warning("yt-dlp 超时")
            time.sleep(random.uniform(5, 10))
        except Exception as e:
            logger.warning("下载异常: %s", e)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return None


def transcribe_whisper(audio_path: Path, model) -> tuple[str, list]:
    """转写音频文件。model 应在调用层加载一次，避免在循环内重复从磁盘初始化。"""
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
# 断点续传：进度日志
# ---------------------------------------------------------------------------
def _load_progress(progress_path: Path) -> set[str]:
    """读取已完成的 bvid/aid 集合，支持断点续传。"""
    if not progress_path.exists():
        return set()
    done: set[str] = set()
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                done.add(line.split(",")[0])
    return done


def _mark_done(progress_path: Path, key: str, status: str) -> None:
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(f"{key},{status},{int(time.time())}\n")


# ---------------------------------------------------------------------------
# 核心：逐视频流水线（字幕优先，立即 Whisper，支持断点续传）
# ---------------------------------------------------------------------------
def process_all_videos(
    video_list: list[VideoItem],
    session: requests.Session,
    subtitle_dir: Path,
    audio_dir: Path,
    transcript_dir: Path,
    transcript_json_dir: Path,
    subtitle_only: bool = False,
    whisper_model=None,
    keep_audio: bool = False,
    progress_path: Path | None = None,
) -> None:
    """
    逐视频处理：对每个视频先尝试官方字幕，无字幕时立即用 Whisper 转写。
    不再将所有视频收集到 need_whisper 列表后才开始转写，避免 Phase 1
    耗尽数小时后 Phase 2 才能启动。支持断点续传。
    """
    done_keys = _load_progress(progress_path) if progress_path else set()
    total = len(video_list)
    skipped = whisper_queued = subtitle_got = whisper_done = 0

    for i, item in enumerate(video_list):
        key = item.bvid or item.aid
        if not key:
            logger.warning("[%d/%d] 无有效 aid/bvid，跳过", i + 1, total)
            continue

        # ── 断点续传：已处理过的直接跳过 ──────────────────────
        if key in done_keys:
            skipped += 1
            continue

        # ── 同时检查磁盘上是否已有输出文件 ────────────────────
        bvid_key = item.bvid if item.bvid else key
        sub_done  = (subtitle_dir   / f"{bvid_key}_subtitle.json").exists()
        xscr_done = (transcript_dir / f"{bvid_key}.txt").exists()
        if sub_done or xscr_done:
            logger.info("[%d/%d] %s 已有输出，跳过", i + 1, total, bvid_key)
            if progress_path:
                _mark_done(progress_path, key, "skipped_existing")
            done_keys.add(key)
            skipped += 1
            continue

        logger.info("[%d/%d] %s | %s", i + 1, total, key, (item.title or "")[:50])

        # ── 步骤 1：获取 bvid + cid ─────────────────────────
        if not item.bvid or not item.cid:
            bvid, cid = get_bvid_cid_from_aid(item.aid, session, item.video_url)
            if bvid:
                item.bvid = bvid
                item.cid  = cid or ""
                key = bvid  # 更新 key 为更精确的 bvid

        # ── 步骤 2：尝试官方字幕 ─────────────────────────────
        got_subtitle = False
        if item.bvid and item.cid:
            subs = get_bilibili_subtitle(item.bvid, item.cid, session)
            if subs:
                json_path = subtitle_dir / f"{item.bvid}_subtitle.json"
                txt_path  = subtitle_dir / f"{item.bvid}_subtitle.txt"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(subs, f, ensure_ascii=False, indent=2)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(subtitle_to_text(subs))
                logger.info("  [字幕] 已保存: %s", json_path.name)
                got_subtitle = True
                subtitle_got += 1
                if progress_path:
                    _mark_done(progress_path, key, "subtitle")
                done_keys.add(key)

        if got_subtitle:
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        # ── 步骤 3：无官方字幕 → 立即 Whisper ───────────────
        if subtitle_only:
            logger.info("  无字幕（subtitle-only 模式，跳过 Whisper）")
            if progress_path:
                _mark_done(progress_path, key, "no_subtitle_skipped")
            done_keys.add(key)
            whisper_queued += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        if whisper_model is None:
            logger.warning("  无官方字幕且未加载 Whisper 模型，跳过 %s", key)
            if progress_path:
                _mark_done(progress_path, key, "no_whisper_model")
            done_keys.add(key)
            continue

        bvid_for_dl = item.bvid or item.aid
        logger.info("  [Whisper] 下载音频: %s", bvid_for_dl)
        audio_path = download_audio_bvid(bvid_for_dl, item.video_url, audio_dir)
        if not audio_path or not audio_path.exists():
            logger.warning("  音频下载失败，跳过")
            if progress_path:
                _mark_done(progress_path, key, "audio_failed")
            done_keys.add(key)
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        try:
            plain_text, subs = transcribe_whisper(audio_path, whisper_model)
        except Exception as e:
            logger.warning("  转写失败: %s", e)
            if progress_path:
                _mark_done(progress_path, key, "whisper_failed")
            done_keys.add(key)
            continue
        finally:
            if not keep_audio and audio_path.exists():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

        txt_path  = transcript_dir     / f"{bvid_for_dl}.txt"
        json_path = transcript_json_dir / f"{bvid_for_dl}_subtitle.json"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(plain_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(subs, f, ensure_ascii=False, indent=2)
        logger.info("  [Whisper] 已保存: %s", txt_path.name)
        whisper_done += 1
        if progress_path:
            _mark_done(progress_path, key, "whisper")
        done_keys.add(key)
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    logger.info(
        "完成：字幕=%d，Whisper=%d，已跳过=%d，subtitle-only队列=%d，总=%d",
        subtitle_got, whisper_done, skipped, whisper_queued, total,
    )


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
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="转写后保留音频文件（默认：转写完成后删除以节省磁盘空间）",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Whisper 模型缓存目录（默认 ~/.cache/whisper）。C 盘空间不足时可指定其他盘，如 E:\\whisper_models",
    )
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = _resolve_default_videos_csv()
    if csv_path is None or not csv_path.exists():
        logger.error(
            "未找到视频列表 CSV。请先运行 03_run_bilibili_crawler.py，或使用 --csv 指定路径。"
        )
        sys.exit(1)

    video_list = _load_video_list_from_csv(csv_path)
    logger.info("加载视频列表: %s，共 %d 条", csv_path, len(video_list))

    cookie = _get_cookie()
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    }
    if cookie:
        base_headers["Cookie"] = cookie
    else:
        logger.warning("未设置 BILIBILI_COOKIE，部分接口可能受限。可设置环境变量或 bilibili_cookie.txt")

    session = requests.Session()
    session.headers.update(base_headers)

    DEFAULT_SUBTITLE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_TRANSCRIPT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    # 进度日志文件：重启后自动跳过已完成的视频
    progress_path = PROJECT_ROOT / "00_output" / "01_Raw_data" / "02_Bilibili" / "progress.csv"

    # ---------- 预检查 Whisper 依赖（仅非 subtitle-only 模式）----------
    whisper_model = None
    if not args.subtitle_only:
        if not check_yt_dlp_ffmpeg() or not check_whisper():
            logger.error("缺少 yt-dlp/ffmpeg 或 whisper，请安装后重试。")
            sys.exit(1)
        whisper_device = _whisper_device(use_gpu=not args.no_gpu)
        model_dir = args.model_dir
        if model_dir:
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Whisper 模型缓存目录: %s", model_dir)
        logger.info("Whisper 使用设备: %s，预加载模型 %s ...", whisper_device, args.whisper_model)
        import whisper as whisper_lib
        whisper_model = whisper_lib.load_model(
            args.whisper_model,
            device=whisper_device,
            download_root=str(model_dir) if model_dir else None,
        )
        logger.info("Whisper 模型加载完成，开始逐视频流水线处理")
    else:
        logger.info("subtitle-only 模式：仅抓字幕，无字幕视频将被记录但不转写")

    # ---------- 逐视频流水线（字幕优先，立即 Whisper，断点续传）----------
    process_all_videos(
        video_list=video_list,
        session=session,
        subtitle_dir=DEFAULT_SUBTITLE_DIR,
        audio_dir=DEFAULT_AUDIO_DIR,
        transcript_dir=DEFAULT_TRANSCRIPT_DIR,
        transcript_json_dir=DEFAULT_TRANSCRIPT_JSON_DIR,
        subtitle_only=args.subtitle_only,
        whisper_model=whisper_model,
        keep_audio=args.keep_audio,
        progress_path=progress_path,
    )

    logger.info("全部完成。字幕目录: %s；转写目录: %s", DEFAULT_SUBTITLE_DIR, DEFAULT_TRANSCRIPT_DIR)


if __name__ == "__main__":
    main()
