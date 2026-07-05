# -*- coding: utf-8 -*-
"""
B 站爬取数据去重脚本
对 00_output/01_Raw_data/02_Bilibili/csv 下的 CSV 按主键去重（评论 comment_id、视频 video_id、创作者 user_id），
原文件会先备份为 *.csv.bak，再写回去重后的结果。
"""
import argparse
import os
import shutil
import glob

try:
    import pandas as pd
except ImportError:
    print("请先安装 pandas: pip install pandas")
    raise

# 各类型 CSV 的主键列名
KEY_COLUMNS = {
    "comments": "comment_id",
    "videos": "video_id",
    "creators": "user_id",
    "search_comments_all": "comment_id",  # 与 run 脚本中的评论汇总文件一致
}

# 支持的文件名模式（含主键）
FILE_PATTERNS = [
    "*comments*.csv",
    "*videos*.csv",
    "*creators*.csv",
]


def infer_key_column(df_columns: list) -> str | None:
    """根据列名推断主键列"""
    for key in KEY_COLUMNS.values():
        if key in df_columns:
            return key
    return None


def dedup_csv(file_path: str, key_column: str | None = None, backup: bool = True) -> tuple[int, int]:
    """
    对单个 CSV 去重，保留第一次出现的行。
    返回 (去重前行数, 去重后行数)。
    """
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            break
        except Exception:
            continue
    if df is None or df.empty:
        return 0, 0

    n_before = len(df)
    key_col = key_column or infer_key_column(list(df.columns))
    if not key_col or key_col not in df.columns:
        print(f"  跳过（未识别主键列）: {file_path}")
        return n_before, n_before

    df_dedup = df.drop_duplicates(subset=[key_col], keep="first")
    n_after = len(df_dedup)

    if n_before == n_after:
        return n_before, n_after

    if backup:
        bak_path = file_path + ".bak"
        shutil.copy2(file_path, bak_path)

    df_dedup.to_csv(file_path, index=False, encoding="utf-8-sig")
    return n_before, n_after


def main():
    parser = argparse.ArgumentParser(description="B 站 CSV 数据去重")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="数据目录，默认为当前目录下的 00_output/01_Raw_data/02_Bilibili/csv 或 data/bili/csv",
    )
    parser.add_argument("--no-backup", action="store_true", help="不去重前备份原文件")
    args = parser.parse_args()

    if args.data_dir and os.path.isdir(args.data_dir):
        data_dir = os.path.abspath(args.data_dir)
    else:
        # 默认：项目根下的常见路径
        cwd = os.getcwd()
        for sub in [
            "00_output/01_Raw_data/02_Bilibili/csv",
            "00_output/01_Raw_data/02_Bilibili",
            "data/bili/csv",
            "data/bili",
        ]:
            candidate = os.path.join(cwd, sub)
            if os.path.isdir(candidate):
                data_dir = candidate
                break
        else:
            data_dir = cwd

    if not os.path.isdir(data_dir):
        print(f"错误：目录不存在 {data_dir}")
        return 1

    collected = set()
    for pattern in FILE_PATTERNS:
        for p in glob.glob(os.path.join(data_dir, pattern)):
            collected.add(p)

    # 也直接扫描目录下所有 csv（避免漏掉 search_comments_all.csv 等）
    for f in os.listdir(data_dir):
        if f.endswith(".csv") and not f.endswith(".csv.bak"):
            collected.add(os.path.join(data_dir, f))

    if not collected:
        print(f"在 {data_dir} 下未找到 CSV 文件")
        return 0

    print(f"数据目录: {data_dir}")
    print(f"找到 {len(collected)} 个 CSV 文件，开始去重（备份: {not args.no_backup}）\n")

    total_removed = 0
    for file_path in sorted(collected):
        name = os.path.basename(file_path)
        key_col = KEY_COLUMNS.get(
            "search_comments_all" if "search_comments_all" in name else
            "comments" if "comment" in name.lower() else
            "videos" if "video" in name.lower() else
            "creators" if "creator" in name.lower() else None
        )
        try:
            n_before, n_after = dedup_csv(file_path, key_column=key_col, backup=not args.no_backup)
            removed = n_before - n_after
            total_removed += removed
            if removed > 0:
                print(f"  {name}: {n_before} -> {n_after} 行，删除重复 {removed} 条")
            else:
                print(f"  {name}: {n_after} 行，无重复")
        except Exception as e:
            print(f"  {name}: 处理失败 - {e}")

    print(f"\n合计去除重复: {total_removed} 条")
    return 0


if __name__ == "__main__":
    exit(main())
