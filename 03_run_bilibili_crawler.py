# -*- coding: utf-8 -*-
"""
在 Final_Project 下运行 B 站爬虫的启动脚本。
不修改 MediaCrawler 源文件，通过本脚本注入 01_bilibili_crawler_config 的配置后调用爬虫。
"""
import importlib.util
import os
import sys
import runpy

# Final_Project 根目录（本脚本所在目录）
FINAL_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# MediaCrawler 项目目录
CRAWLER_DIR = os.path.join(FINAL_PROJECT_ROOT, "MediaCrawler-main", "MediaCrawler-main")

if not os.path.isdir(CRAWLER_DIR):
    print(f"错误：未找到爬虫目录 {CRAWLER_DIR}")
    sys.exit(1)

# 确保能加载同目录下的 01_bilibili_crawler_config
if FINAL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, FINAL_PROJECT_ROOT)
# 将爬虫目录加入路径并切换工作目录，以便相对路径（libs/、data/、.env 等）正确
sys.path.insert(0, CRAWLER_DIR)
os.chdir(CRAWLER_DIR)

# 在导入爬虫 main 之前，先导入 config 并应用 B 站专用配置
import config
config_path = os.path.join(FINAL_PROJECT_ROOT, "01_bilibili_crawler_config.py")
spec = importlib.util.spec_from_file_location("bilibili_crawler_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config_module.apply_bilibili_config(config)

# B 站爬虫输出统一保存到 00_output/01_Raw_data/02_Bilibili
BILIBILI_RAW_DATA_DIR = os.path.join(FINAL_PROJECT_ROOT, "00_output", "01_Raw_data", "02_Bilibili")
os.makedirs(BILIBILI_RAW_DATA_DIR, exist_ok=True)
config.SAVE_DATA_PATH = BILIBILI_RAW_DATA_DIR

# 让 AsyncFileWriter 写入 02_Bilibili 时不再追加一层 platform 子目录，即 base_path = SAVE_DATA_PATH/file_type
import tools.async_file_writer as async_file_writer_module
_original_get_file_path = async_file_writer_module.AsyncFileWriter._get_file_path


def _patched_get_file_path(self, file_type: str, item_type: str) -> str:
    if config.SAVE_DATA_PATH and "02_Bilibili" in config.SAVE_DATA_PATH:
        base_path = f"{config.SAVE_DATA_PATH}/{file_type}"
    else:
        return _original_get_file_path(self, file_type, item_type)
    import pathlib
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    from tools.utils import utils
    file_name = f"{self.crawler_type}_{item_type}_{utils.get_current_date()}.{file_type}"
    return f"{base_path}/{file_name}"


async_file_writer_module.AsyncFileWriter._get_file_path = _patched_get_file_path

# 不按采集时间分类：固定日期后缀为 all，多次运行追加到同一文件
import tools.utils as crawler_utils


def _bilibili_single_file_date():
    return "all"


crawler_utils.get_current_date = _bilibili_single_file_date

# ---------- 评论/视频/创作者去重：重跑时不再重复写入已采集过的数据 ----------
# 用脚本所在目录算绝对路径，避免 chdir 后读到错误文件
import csv as csv_module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_bilibili_csv_dir = os.path.join(_script_dir, "00_output", "01_Raw_data", "02_Bilibili", "csv")
_bilibili_seen_comment_ids = set()
_bilibili_seen_video_ids = set()
_bilibili_seen_creator_ids = set()
_bilibili_video_ids_with_comments = set()  # 已有至少一条评论的视频，重跑时不再请求评论

def _load_csv_ids(file_path: str, id_column: str, into_set: set) -> bool:
    if not os.path.isfile(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv_module.DictReader(f)
            if id_column not in (reader.fieldnames or []):
                return False
            for row in reader:
                vid = row.get(id_column)
                if vid not in (None, ""):
                    into_set.add(str(vid))
        return True
    except Exception:
        return False

_comments_csv_path = os.path.join(_bilibili_csv_dir, "search_comments_all.csv")
if os.path.isfile(_comments_csv_path):
    try:
        with open(_comments_csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv_module.DictReader(f)
            fn = reader.fieldnames or []
            for row in reader:
                cid = row.get("comment_id")
                if cid not in (None, ""):
                    _bilibili_seen_comment_ids.add(str(cid))
                vid = row.get("video_id")
                if vid not in (None, ""):
                    _bilibili_video_ids_with_comments.add(str(vid))
        print(f"[03_run_bilibili_crawler] 已加载 {len(_bilibili_seen_comment_ids)} 条已有评论、{len(_bilibili_video_ids_with_comments)} 个已有评论的视频，重跑时跳过重复且不再请求这些视频的评论。")
    except Exception as e:
        print(f"[03_run_bilibili_crawler] 读取评论 CSV 失败: {e}")
else:
    print(f"[03_run_bilibili_crawler] 未找到已有评论文件: {_comments_csv_path}")

_videos_csv_path = os.path.join(_bilibili_csv_dir, "search_videos_all.csv")
if _load_csv_ids(_videos_csv_path, "video_id", _bilibili_seen_video_ids):
    print(f"[03_run_bilibili_crawler] 已加载 {len(_bilibili_seen_video_ids)} 条已有视频 ID，重跑时将跳过重复。")
else:
    print(f"[03_run_bilibili_crawler] 未找到已有视频文件或读取失败: {_videos_csv_path}")

_creators_csv_path = os.path.join(_bilibili_csv_dir, "search_creators_all.csv")
if _load_csv_ids(_creators_csv_path, "user_id", _bilibili_seen_creator_ids):
    print(f"[03_run_bilibili_crawler] 已加载 {len(_bilibili_seen_creator_ids)} 条已有创作者 ID，重跑时将跳过重复。")
else:
    print(f"[03_run_bilibili_crawler] 未找到已有创作者文件或读取失败: {_creators_csv_path}")

_original_write_to_csv = async_file_writer_module.AsyncFileWriter.write_to_csv

# 进度与剩余时间：仅统计本次运行新写入的评论
import time
_progress_start_time = None
_progress_comment_count = 0
_progress_interval = 100  # 每 N 条新评论打印一次
_estimated_total_comments = getattr(
    config, "CRAWLER_MAX_NOTES_COUNT", 500
) * min(200, getattr(config, "CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES", 100))


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or not (seconds < 86400 * 7):
        return "—"
    m = int(seconds // 60)
    s = int(seconds % 60)
    if m >= 60:
        return f"{m // 60} 小时 {m % 60} 分"
    return f"{m} 分 {s} 秒"


async def _patched_write_to_csv(self, item, item_type):
    global _progress_start_time, _progress_comment_count
    if item_type == "comments":
        cid = item.get("comment_id")
        if cid is not None:
            cid_str = str(cid)
            if cid_str in _bilibili_seen_comment_ids:
                return
            _bilibili_seen_comment_ids.add(cid_str)
    elif item_type == "videos":
        vid = item.get("video_id")
        if vid is not None:
            vid_str = str(vid)
            if vid_str in _bilibili_seen_video_ids:
                return
            _bilibili_seen_video_ids.add(vid_str)
    elif item_type == "creators":
        uid = item.get("user_id")
        if uid is not None:
            uid_str = str(uid)
            if uid_str in _bilibili_seen_creator_ids:
                return
            _bilibili_seen_creator_ids.add(uid_str)
    await _original_write_to_csv(self, item, item_type)
    if item_type == "comments":
        vid = item.get("video_id")
        if vid is not None:
            _bilibili_video_ids_with_comments.add(str(vid))
    # 仅对评论统计进度并打印
    if item_type == "comments":
        if _progress_start_time is None:
            _progress_start_time = time.time()
        _progress_comment_count += 1
        if _progress_comment_count % _progress_interval == 0 or _progress_comment_count == 1:
            elapsed = time.time() - _progress_start_time
            elapsed_str = _format_eta(elapsed)
            line = f"[进度] 已采集 {_progress_comment_count} 条评论，已用时 {elapsed_str}"
            if _progress_comment_count >= _progress_interval and elapsed > 10:
                rate = _progress_comment_count / elapsed
                remaining = max(0, _estimated_total_comments - _progress_comment_count)
                eta_sec = remaining / rate if rate > 0 else 0
                line += f"，预计剩余约 {_format_eta(eta_sec)}"
            print(line)


async_file_writer_module.AsyncFileWriter.write_to_csv = _patched_write_to_csv

# ---------- 屏蔽“每条评论一条”的 store 日志，便于看到真实进度（关键词、页码、视频 ID 等） ----------
import logging

class _BilibiliStoreCommentLogFilter(logging.Filter):
    """过滤掉 store 里每条评论的 INFO，保留爬虫进度类日志"""
    def filter(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if "[store.bilibili.update_bilibili_video_comment]" in msg:
            return False
        return True

_crawler_logger = logging.getLogger("MediaCrawler")
_crawler_logger.addFilter(_BilibiliStoreCommentLogFilter())

# ---------- 楼中楼并行拉取 + 单视频完成进度 + 网络异常容错（不因单视频失败而整程退出） ----------
import asyncio
_SUB_COMMENT_CONCURRENCY = 5  # 每页内同时拉取的楼中楼条数，避免顺序等待
try:
    import media_platform.bilibili.client as _bili_client_module
    from media_platform.bilibili.field import CommentOrderType
    from media_platform.bilibili.exception import DataFetchError
    import random
    _bili_utils = _bili_client_module.utils
    _orig_get_all_comments = _bili_client_module.BilibiliClient.get_video_all_comments
    _orig_get_level_two = _bili_client_module.BilibiliClient.get_video_all_level_two_comments

    async def _patched_get_video_all_comments(self, video_id, crawl_interval=1.0, is_fetch_sub_comments=False, callback=None, max_count=10):
        result = []
        try:
            is_end = False
            next_page = 0
            max_retries = 3
            while not is_end and len(result) < max_count:
                comments_res = None
                for attempt in range(max_retries):
                    try:
                        comments_res = await self.get_video_comments(video_id, CommentOrderType.DEFAULT, next_page)
                        break
                    except DataFetchError as e:
                        if attempt < max_retries - 1:
                            delay = 5 * (2**attempt) + random.uniform(0, 1)
                            _bili_utils.logger.warning(
                                f"[BilibiliClient.get_video_all_comments] Retrying video_id {video_id} in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(delay)
                        else:
                            _bili_utils.logger.error(
                                f"[BilibiliClient.get_video_all_comments] Max retries reached for video_id: {video_id}. Skipping comments. Error: {e}"
                            )
                            is_end = True
                            break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            delay = 5 * (2**attempt) + random.uniform(0, 1)
                            _bili_utils.logger.warning(
                                f"[BilibiliClient.get_video_all_comments] 网络/协议异常，重试 video_id {video_id} 于 {delay:.2f}s 后 (Attempt {attempt + 1}/{max_retries}): {e}"
                            )
                            await asyncio.sleep(delay)
                        else:
                            _bili_utils.logger.error(
                                f"[BilibiliClient.get_video_all_comments] video_id {video_id} 重试 {max_retries} 次后仍失败，跳过该视频: {e}"
                            )
                            is_end = True
                            break
                if not comments_res:
                    break
                cursor_info = comments_res.get("cursor")
                if not cursor_info:
                    _bili_utils.logger.warning(
                        f"[BilibiliClient.get_video_all_comments] Could not find 'cursor' in response for video_id: {video_id}. Skipping."
                    )
                    break
                comment_list = comments_res.get("replies", [])
                if "is_end" not in cursor_info or "next" not in cursor_info:
                    is_end = True
                else:
                    is_end = cursor_info.get("is_end")
                    next_page = cursor_info.get("next")
                if not isinstance(is_end, bool):
                    is_end = True
                if is_fetch_sub_comments and comment_list:
                    sem = asyncio.Semaphore(_SUB_COMMENT_CONCURRENCY)
                    async def _fetch_one_sub(comment):
                        if comment.get("rcount", 0) <= 0:
                            return
                        async with sem:
                            await _orig_get_level_two(
                                self, video_id, comment["rpid"], CommentOrderType.DEFAULT, 10, crawl_interval, callback
                            )
                    outs = await asyncio.gather(*[_fetch_one_sub(c) for c in comment_list], return_exceptions=True)
                    for o in outs:
                        if isinstance(o, BaseException):
                            _bili_utils.logger.warning("[03_run_bilibili_crawler] 楼中楼单条请求失败（已忽略，继续）: %s", o)
                if len(result) + len(comment_list) > max_count:
                    comment_list = comment_list[: max_count - len(result)]
                if callback:
                    await callback(video_id, comment_list)
                await asyncio.sleep(crawl_interval)
                if not is_fetch_sub_comments:
                    result.extend(comment_list)
                    continue
            print(f"[进度] 视频 {video_id} 评论采集完成")
        except Exception as e:
            _bili_utils.logger.error("[03_run_bilibili_crawler] 视频 %s 评论采集异常，跳过该视频继续: %s", video_id, e)
        return result

    _bili_client_module.BilibiliClient.get_video_all_comments = _patched_get_video_all_comments
except Exception as _e:
    print(f"[03_run_bilibili_crawler] 未注入楼中楼并行/进度逻辑: {_e}")

print(f"[03_run_bilibili_crawler] 评论进度将每 {_progress_interval} 条打印一次，并显示已用时间与预计剩余时间（按约 {_estimated_total_comments} 条总量估算）。")
print(f"[03_run_bilibili_crawler] 每个视频会完整采完所有评论；楼中楼并行数={_SUB_COMMENT_CONCURRENCY}，评论并发视频数见 MAX_CONCURRENCY_NUM。")

# ---------- 重跑时跳过“已有评论”的视频，不再请求评论接口 ----------
try:
    import media_platform.bilibili.core as _bili_core_module
    _orig_batch_get_comments = _bili_core_module.BilibiliCrawler.batch_get_video_comments
    async def _patched_batch_get_video_comments(self, video_id_list):
        to_fetch = [vid for vid in video_id_list if str(vid) not in _bilibili_video_ids_with_comments]
        skipped = len(video_id_list) - len(to_fetch)
        if skipped:
            print(f"[03_run_bilibili_crawler] 本批 {len(video_id_list)} 个视频中 {skipped} 个已有评论，跳过不请求；仅对 {len(to_fetch)} 个视频拉取评论。")
        if not to_fetch:
            return
        await _orig_batch_get_comments(self, to_fetch)
    _bili_core_module.BilibiliCrawler.batch_get_video_comments = _patched_batch_get_video_comments
except Exception as _e:
    print(f"[03_run_bilibili_crawler] 未注入“跳过已采视频”逻辑: {_e}")

# 以 __main__ 方式执行爬虫的 main.py，会触发其 if __name__ == "__main__" 逻辑
runpy.run_path(os.path.join(CRAWLER_DIR, "main.py"), run_name="__main__")
     