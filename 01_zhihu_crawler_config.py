# -*- coding: utf-8 -*-
"""
知乎爬虫专用配置（不修改 MediaCrawler 源文件）
在 Final_Project 下使用 02_run_zhihu_crawler.py 启动时会自动应用本配置。
"""
# 抓取平台：知乎
PLATFORM = "zhihu"

# 搜索关键词：明末渊虚之羽
KEYWORDS = "明末渊虚之羽"

# 爬取类型：关键词搜索
CRAWLER_TYPE = "search"

# 抓取条数上限（回答/文章/视频等）
CRAWLER_MAX_NOTES_COUNT = 2000

# 开启评论抓取
ENABLE_GET_COMMENTS = True

# 单条内容下的一级评论数量上限
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 500

# 开启二级评论（回复/互动数据）
ENABLE_GET_SUB_COMMENTS = True

# 数据保存格式：csv
SAVE_DATA_OPTION = "csv"

# ---------- 反爬 / 限速（遇 403、验证码、安全验证时优先调整） ----------
# 每次请求间隔（秒），知乎反爬严，建议 5~15，被限流可调到 15~30
CRAWLER_MAX_SLEEP_SEC = 5
# 是否使用代理 IP 池（True 需自行配置快代理/豌豆 HTTP 等，见项目 docs/代理使用.md）
ENABLE_IP_PROXY = False
# 代理池数量（启用代理时生效）
IP_PROXY_POOL_COUNT = 2
# 是否开启 CDP 模式（用本机 Chrome/Edge，更不易被检测，建议 True）
ENABLE_CDP_MODE = True
# 是否无头浏览器（CDP 下建议 False，有窗口更不易触发验证）
CDP_HEADLESS = False


def apply_zhihu_config(config_module):
    """将本文件的配置应用到 MediaCrawler 的 config 模块。"""
    config_module.PLATFORM = PLATFORM
    config_module.KEYWORDS = KEYWORDS
    config_module.CRAWLER_TYPE = CRAWLER_TYPE
    config_module.CRAWLER_MAX_NOTES_COUNT = CRAWLER_MAX_NOTES_COUNT
    config_module.ENABLE_GET_COMMENTS = ENABLE_GET_COMMENTS
    max_comments = CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES
    config_module.CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = max_comments
    config_module.ENABLE_GET_SUB_COMMENTS = ENABLE_GET_SUB_COMMENTS
    config_module.SAVE_DATA_OPTION = SAVE_DATA_OPTION
    # 反爬 / 限速
    config_module.CRAWLER_MAX_SLEEP_SEC = CRAWLER_MAX_SLEEP_SEC
    config_module.ENABLE_IP_PROXY = ENABLE_IP_PROXY
    config_module.IP_PROXY_POOL_COUNT = IP_PROXY_POOL_COUNT
    config_module.ENABLE_CDP_MODE = ENABLE_CDP_MODE
    config_module.CDP_HEADLESS = CDP_HEADLESS
