# -*- coding: utf-8 -*-
"""
tsar 数据处理脚本
功能：
  1. 时间戳解析 —— 毫秒级大整数 -> 人类可读的 年-月-日 时:分
  2. 按小时汇总指标 —— 每小时的平均值、最大值、分钟级采样数
"""

import csv
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics
import os

# ============================================================
# 全局配置
# ============================================================
DATA_DIR = "C:/xxq"
OUTPUT_DIR = "C:/xxq/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 时区：UTC+8（东八区）
TZ = timezone(timedelta(hours=8))

# Tab 分隔符
DELIMITER = "\t"


# ============================================================
# 工具函数
# ============================================================
def parse_ts(ts_ms):
    """将毫秒级时间戳转为人类可读的 datetime 对象（东八区）"""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=TZ)


def format_ts(ts_ms):
    """毫秒时间戳 -> 'YYYY-MM-DD HH:MM' 字符串"""
    dt = parse_ts(ts_ms)
    return dt.strftime("%Y-%m-%d %H:%M")


def format_hour(ts_ms):
    """毫秒时间戳 -> 'YYYY-MM-DD HH:00' 字符串（截断到小时）"""
    dt = parse_ts(ts_ms)
    return dt.strftime("%Y-%m-%d %H:00")


def read_tsv(filepath):
    """读取 Tab 分隔的文件，返回 dict 列表"""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        return list(reader)


def safe_float(val):
    """安全转 float"""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ============================================================
# 1. 时间戳解析
# ============================================================
def task_timestamp_parsing(disk_rows, pref_rows):
    print("=" * 70)
    print("【任务一】时间戳解析")
    print("=" * 70)

    # --- 概念讲解 ---
    print("""
■ 什么是毫秒级时间戳？
  时间戳是从 1970-01-01 00:00:00 UTC 起经过的秒数（或毫秒数）。
  本数据中的 ts 字段是 13 位整数，表示毫秒级时间戳。

■ 解析方法：
  Python: datetime.fromtimestamp(ts_ms / 1000, tz=timezone(timedelta(hours=8)))
    - ts_ms / 1000  →  毫秒转秒
    - tz=UTC+8      →  东八区（北京时间）
""")

    # --- 收集所有时间戳并去重排序 ---
    all_ts = set()
    for r in disk_rows:
        all_ts.add(int(r["ts"]))
    for r in pref_rows:
        all_ts.add(int(r["ts"]))

    sorted_ts = sorted(all_ts)
    ts_min = sorted_ts[0]
    ts_max = sorted_ts[-1]

    print(f"■ 数据时间范围：")
    print(f"  最早时间戳: {ts_min}  ->  {format_ts(ts_min)}")
    print(f"  最晚时间戳: {ts_max}  ->  {format_ts(ts_max)}")
    print(f"  去重时间点总数: {len(sorted_ts)}")

    # --- 磁盘采样间隔验证 ---
    disk_ts_sorted = sorted(set(int(r["ts"]) for r in disk_rows))
    if len(disk_ts_sorted) >= 2:
        interval_ms = disk_ts_sorted[1] - disk_ts_sorted[0]
        print(f"\n■ 磁盘采样间隔验证：")
        print(f"  第1个采样点: {disk_ts_sorted[0]} -> {format_ts(disk_ts_sorted[0])}")
        print(f"  第2个采样点: {disk_ts_sorted[1]} -> {format_ts(disk_ts_sorted[1])}")
        print(f"  间隔: {interval_ms} ms = {interval_ms / 1000} 秒 = {interval_ms / 60000} 分钟")

    # --- 性能采样间隔验证 ---
    pref_ts_sorted = sorted(set(int(r["ts"]) for r in pref_rows))
    if len(pref_ts_sorted) >= 2:
        interval_ms = pref_ts_sorted[1] - pref_ts_sorted[0]
        print(f"\n■ 性能采样间隔验证：")
        print(f"  第1个采样点: {pref_ts_sorted[0]} -> {format_ts(pref_ts_sorted[0])}")
        print(f"  第2个采样点: {pref_ts_sorted[1]} -> {format_ts(pref_ts_sorted[1])}")
        print(f"  间隔: {interval_ms} ms = {interval_ms / 1000} 秒 = {interval_ms / 60000} 分钟")

    # --- 展示前10个时间戳解析示例 ---
    print(f"\n■ 前 10 个时间戳解析示例（去重后）：")
    print(f"  {'原始时间戳(ms)':<18} {'可读时间':<20} {'小时桶'}")
    print(f"  {'-'*18} {'-'*20} {'-'*16}")
    for ts in sorted_ts[:10]:
        dt = parse_ts(ts)
        print(f"  {ts:<18} {dt.strftime('%Y-%m-%d %H:%M'):<20} {format_hour(ts)}")

    # --- 保存完整时间戳解析表 ---
    ts_output = os.path.join(OUTPUT_DIR, "timestamp_parsed.tsv")
    with open(ts_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(["ts_ms", "readable_time", "hour_bucket", "weekday"])
        for ts in sorted_ts:
            dt = parse_ts(ts)
            weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            writer.writerow([
                ts,
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                format_hour(ts),
                weekdays[dt.weekday()]
            ])
    print(f"\n  [已保存] 完整时间戳解析表 -> {ts_output}（共 {len(sorted_ts)} 行）")

    return sorted_ts


# ============================================================
# 2. 按小时汇总指标
# ============================================================
def task_hourly_aggregation(disk_rows, pref_rows, mod_lookup):
    print("\n" + "=" * 70)
    print("【任务二】按小时汇总指标")
    print("=" * 70)

    print("""
■ 汇总逻辑：
  按 (hostid, mod, hour_bucket) 三元组分组，对每组计算：
    - avg_value:  该小时内所有采样值的平均值
    - max_value:  该小时内所有采样值的最大值
    - sample_count: 该小时内的采样次数（分钟级采样数）
    - min_ts:     该小时内最早采样时间戳
    - max_ts:     该小时内最晚采样时间戳
""")

    # --- 合并 disk + pref ---
    all_rows = disk_rows + pref_rows
    print(f"  磁盘记录数: {len(disk_rows)}")
    print(f"  性能记录数: {len(pref_rows)}")
    print(f"  合并总记录数: {len(all_rows)}")

    # --- 分组 ---
    # key = (hostid, mod, hour_bucket)
    groups = defaultdict(list)
    for r in all_rows:
        ts = int(r["ts"])
        hour = format_hour(ts)
        val = safe_float(r["value"])
        if val is not None:
            key = (r["hostid"], r["mod"], hour)
            groups[key].append((ts, val))

    print(f"  分组数（hostid × mod × hour）: {len(groups)}")

    # --- 汇总 ---
    results = []
    for (hostid, mod, hour), samples in groups.items():
        values = [v for _, v in samples]
        ts_list = [t for t, _ in samples]
        mod_info = mod_lookup.get(mod, {})
        results.append({
            "hour": hour,
            "hostid": hostid,
            "mod": mod,
            "type": mod_info.get("type", ""),
            "desc": mod_info.get("desc", ""),
            "unit": mod_info.get("unit", ""),
            "tag": mod_info.get("tag", ""),
            "avg_value": round(statistics.mean(values), 2),
            "max_value": round(max(values), 2),
            "min_value": round(min(values), 2),
            "sample_count": len(values),
            "min_ts": min(ts_list),
            "max_ts": max(ts_list),
            "min_ts_readable": format_ts(min(ts_list)),
            "max_ts_readable": format_ts(max(ts_list)),
        })

    # 排序：hour -> hostid -> mod
    results.sort(key=lambda x: (x["hour"], x["hostid"], x["mod"]))

    # --- 保存全量汇总结果 ---
    agg_output = os.path.join(OUTPUT_DIR, "hourly_aggregation.tsv")
    headers = [
        "hour", "hostid", "mod", "type", "desc", "unit", "tag",
        "avg_value", "max_value", "min_value", "sample_count",
        "min_ts", "max_ts", "min_ts_readable", "max_ts_readable"
    ]
    with open(agg_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  [已保存] 全量小时汇总 -> {agg_output}（共 {len(results)} 行）")

    # --- 分 disk / pref 分别保存 ---
    disk_results = [r for r in results if r["type"] == "disk"]
    pref_results = [r for r in results if r["type"] == "pref"]

    disk_out = os.path.join(OUTPUT_DIR, "hourly_aggregation_disk.tsv")
    with open(disk_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(disk_results)
    print(f"  [已保存] 磁盘小时汇总 -> {disk_out}（共 {len(disk_results)} 行）")

    pref_out = os.path.join(OUTPUT_DIR, "hourly_aggregation_pref.tsv")
    with open(pref_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(pref_results)
    print(f"  [已保存] 性能小时汇总 -> {pref_out}（共 {len(pref_results)} 行）")

    # --- 展示示例数据 ---
    print(f"\n■ 性能指标汇总示例（host001 前 5 行）：")
    print(f"  {'小时':<16} {'主机':<10} {'指标':<12} {'说明':<16} {'单位':<8} {'平均值':>10} {'最大值':>10} {'采样数':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*16} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    shown = 0
    for r in pref_results:
        if r["hostid"] == "host001" and shown < 5:
            print(f"  {r['hour']:<16} {r['hostid']:<10} {r['mod']:<12} {r['desc']:<16} {r['unit']:<8} {r['avg_value']:>10} {r['max_value']:>10} {r['sample_count']:>6}")
            shown += 1

    print(f"\n■ 磁盘指标汇总示例（host003, sda_write 前 5 行）：")
    print(f"  {'小时':<16} {'主机':<10} {'指标':<12} {'说明':<20} {'单位':<12} {'平均值':>12} {'最大值':>12} {'采样数':>6}")
    print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*6}")
    shown = 0
    for r in disk_results:
        if r["hostid"] == "host003" and r["mod"] == "sda_write" and shown < 5:
            print(f"  {r['hour']:<16} {r['hostid']:<10} {r['mod']:<12} {r['desc']:<20} {r['unit']:<12} {r['avg_value']:>12} {r['max_value']:>12} {r['sample_count']:>6}")
            shown += 1

    # --- 采样数统计 ---
    print(f"\n■ 采样数分布统计：")
    disk_counts = [r["sample_count"] for r in disk_results]
    pref_counts = [r["sample_count"] for r in pref_results]

    if disk_counts:
        print(f"  磁盘汇总 - 每组采样数: 最小={min(disk_counts)}, 最大={max(disk_counts)}, 平均={round(statistics.mean(disk_counts), 1)}")
    if pref_counts:
        print(f"  性能汇总 - 每组采样数: 最小={min(pref_counts)}, 最大={max(pref_counts)}, 平均={round(statistics.mean(pref_counts), 1)}")

    # --- 按主机汇总概览 ---
    print(f"\n■ 各主机指标数量概览（性能类）：")
    host_mod_count = defaultdict(set)
    for r in pref_results:
        host_mod_count[r["hostid"]].add(r["mod"])
    for hid in sorted(host_mod_count.keys())[:5]:
        print(f"  {hid}: {len(host_mod_count[hid])} 个指标, 涉及 {len(set(r['hour'] for r in pref_results if r['hostid']==hid))} 个小时")
    print(f"  ... (共 {len(host_mod_count)} 台主机)")

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("  tsar 数据处理脚本")
    print("  数据目录:", DATA_DIR)
    print("  输出目录:", OUTPUT_DIR)
    print("=" * 70)

    # --- 读取数据 ---
    host_rows = read_tsv(os.path.join(DATA_DIR, "host_detail.dat"))
    mod_rows = read_tsv(os.path.join(DATA_DIR, "mod_detail.dat"))
    disk_rows = read_tsv(os.path.join(DATA_DIR, "disk_tsar.dat"))
    pref_rows = read_tsv(os.path.join(DATA_DIR, "pref_tsar.dat"))

    print(f"  host_detail: {len(host_rows)} 行")
    print(f"  mod_detail:  {len(mod_rows)} 行")
    print(f"  disk_tsar:   {len(disk_rows)} 行")
    print(f"  pref_tsar:   {len(pref_rows)} 行")

    # --- 构建 mod 字典 ---
    mod_lookup = {}
    for r in mod_rows:
        mod_lookup[r["mod"]] = r

    # --- 任务一：时间戳解析 ---
    sorted_ts = task_timestamp_parsing(disk_rows, pref_rows)

    # --- 任务二：按小时汇总 ---
    results = task_hourly_aggregation(disk_rows, pref_rows, mod_lookup)

    print("\n" + "=" * 70)
    print("  全部处理完成！")
    print(f"  输出文件目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
