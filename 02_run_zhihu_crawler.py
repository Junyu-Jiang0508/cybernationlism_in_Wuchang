# -*- coding: utf-8 -*-
"""
在 Final_Project 下运行知乎爬虫的启动脚本。
不修改 MediaCrawler 源文件，通过本脚本注入 01_zhihu_crawler_config 的配置后调用爬虫。
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

# 确保能加载同目录下的 01_zhihu_crawler_config（文件名含数字，用 importlib 加载）
if FINAL_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, FINAL_PROJECT_ROOT)
# 将爬虫目录加入路径并切换工作目录，以便相对路径（libs/、data/、.env 等）正确
sys.path.insert(0, CRAWLER_DIR)
os.chdir(CRAWLER_DIR)

# 在导入爬虫 main 之前，先导入 config 并应用知乎专用配置
import config
config_path = os.path.join(FINAL_PROJECT_ROOT, "01_zhihu_crawler_config.py")
spec = importlib.util.spec_from_file_location("zhihu_crawler_config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config_module.apply_zhihu_config(config)

# 以 __main__ 方式执行爬虫的 main.py，会触发其 if __name__ == "__main__" 逻辑
runpy.run_path(os.path.join(CRAWLER_DIR, "main.py"), run_name="__main__")
