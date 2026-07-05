# -*- coding: utf-8 -*-
"""
在 Final_Project 下运行知乎爬虫的启动脚本。
不修改 MediaCrawler 源文件，通过本脚本注入 01_zhihu_crawler_config 的配置后调用爬虫。
"""
import importlib.util
import os
import sys
import runpy

# 本脚本所在目录（src/crawl/），配置文件与本脚本同目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Final_Project 仓库根目录（src/crawl/ 上两层）
FINAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
# MediaCrawler 项目目录
CRAWLER_DIR = os.path.join(FINAL_PROJECT_ROOT, "MediaCrawler-main", "MediaCrawler-main")

if not os.path.isdir(CRAWLER_DIR):
    print(f"错误：未找到爬虫目录 {CRAWLER_DIR}")
    sys.exit(1)

# 确保能加载同目录下的 01_zhihu_crawler_config（文件名含数字，用 importlib 加载）
if FINAL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, FINAL_PROJECT_ROOT)
# 将爬虫目录加入路径并切换工作目录，以便相对路径（libs/、data/、.env 等）正确
sys.path.insert(0, CRAWLER_DIR)
os.chdir(CRAWLER_DIR)

# 在导入爬虫 main 之前，先导入 config 并应用知乎专用配置
import config
config_path = os.path.join(SCRIPT_DIR, "01_zhihu_crawler_config.py")
spec = importlib.util.spec_from_file_location("zhihu_crawler_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config_module.apply_zhihu_config(config)

# 知乎爬虫输出统一保存到 00_output/01_Raw_data/01_Zhihu，不按采集时间分文件
ZHIHU_RAW_DATA_DIR = os.path.join(FINAL_PROJECT_ROOT, "00_output", "01_Raw_data", "01_Zhihu")
os.makedirs(ZHIHU_RAW_DATA_DIR, exist_ok=True)
# 直接作为完整数据目录，不再加 platform 子目录
config.SAVE_DATA_PATH = ZHIHU_RAW_DATA_DIR

# 让 AsyncFileWriter 写入 01_Zhihu 时不再追加一层 zhihu，即 base_path = SAVE_DATA_PATH/file_type
import tools.async_file_writer as async_file_writer_module
_original_get_file_path = async_file_writer_module.AsyncFileWriter._get_file_path


def _patched_get_file_path(self, file_type: str, item_type: str) -> str:
    if config.SAVE_DATA_PATH and "01_Zhihu" in config.SAVE_DATA_PATH:
        base_path = f"{config.SAVE_DATA_PATH}/{file_type}"
    else:
        return _original_get_file_path(self, file_type, item_type)
    import pathlib
    pathlib.Path(base_path).mkdir(parents=True, exist_ok=True)
    from tools.utils import utils
    file_name = f"{self.crawler_type}_{item_type}_{utils.get_current_date()}.{file_type}"
    return f"{base_path}/{file_name}"


async_file_writer_module.AsyncFileWriter._get_file_path = _patched_get_file_path

# 不按采集时间分类：固定文件名（如 search_contents_all.csv），多次运行追加到同一文件
import tools.utils as crawler_utils


def _zhihu_single_file_date():
    return "all"


crawler_utils.get_current_date = _zhihu_single_file_date

# 以 __main__ 方式执行爬虫的 main.py，会触发其 if __name__ == "__main__" 逻辑
runpy.run_path(os.path.join(CRAWLER_DIR, "main.py"), run_name="__main__")




























 