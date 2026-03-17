# -*- coding: utf-8 -*-
"""
B 站爬虫专用配置（不修改 MediaCrawler 源文件）
在 Final_Project 下使用 03_run_bilibili_crawler.py 启动时会自动应用本配置。
"""
# 抓取平台：B 站（项目内参数名为 bili）
PLATFORM = "bili"

# 搜索关键词：与知乎一致，可改为你的目标关键词（多个用英文逗号分隔）
KEYWORDS = "明末渊虚之羽"

# 爬取类型：关键词搜索
CRAWLER_TYPE = "search"

# 抓取条数上限（视频数），B 站每页 20 条，会至少取 20
CRAWLER_MAX_NOTES_COUNT = 2000

# 必须开启评论抓取
ENABLE_GET_COMMENTS = True

# 单条视频下的一级评论数量上限
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 500

# 是否抓取楼中楼（二级评论）
ENABLE_GET_SUB_COMMENTS = True

# 数据保存格式：csv
SAVE_DATA_OPTION = "csv"

# B 站搜索模式：normal=不限时间；
# all_in_time_range / daily_limit_in_time_range 需配合 START_DAY/END_DAY
BILI_SEARCH_MODE = "normal"

# ---------- 反爬 / 限速（在保证数据完整前提下适度加速） ----------
CRAWLER_MAX_SLEEP_SEC = 2  # 请求间隔 2 秒（原 3 秒），降低被限速概率的同时加快采集
MAX_CONCURRENCY_NUM = 5    # 同时采集评论的视频数（默认 1），多视频并行可显著缩短总时长
ENABLE_IP_PROXY = False
IP_PROXY_POOL_COUNT = 2
ENABLE_CDP_MODE = True
CDP_HEADLESS = False


def apply_bilibili_config(config_module):
    """将本文件的配置应用到 MediaCrawler 的 config 模块。"""
    config_module.PLATFORM = PLATFORM
    config_module.KEYWORDS = KEYWORDS
    config_module.CRAWLER_TYPE = CRAWLER_TYPE
    config_module.CRAWLER_MAX_NOTES_COUNT = CRAWLER_MAX_NOTES_COUNT
    config_module.ENABLE_GET_COMMENTS = ENABLE_GET_COMMENTS
    config_module.CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = (
        CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES
    )
    config_module.ENABLE_GET_SUB_COMMENTS = ENABLE_GET_SUB_COMMENTS
    config_module.SAVE_DATA_OPTION = SAVE_DATA_OPTION
    config_module.BILI_SEARCH_MODE = BILI_SEARCH_MODE
    config_module.CRAWLER_MAX_SLEEP_SEC = CRAWLER_MAX_SLEEP_SEC
    config_module.MAX_CONCURRENCY_NUM = MAX_CONCURRENCY_NUM
    config_module.ENABLE_IP_PROXY = ENABLE_IP_PROXY
    config_module.IP_PROXY_POOL_COUNT = IP_PROXY_POOL_COUNT
    config_module.ENABLE_CDP_MODE = ENABLE_CDP_MODE
    config_module.CDP_HEADLESS = CDP_HEADLESS
