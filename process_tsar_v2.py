# -*- coding: utf-8 -*-
"""
tsar 数据处理脚本 v2
功能：
  1. ER 关系图 → PNG 图片 (用 Pillow/PIL 绘制)
  2. 时间戳解析 → TSV（对最小粒度=小时的数据，额外按天汇总）
  3. 按小时汇总指标 → TSV
所有输出放在同一个文件夹 C:/xxq/output/
"""

import csv
import os
import statistics
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 全局配置
# ============================================================
DATA_DIR = "C:/xxq"
OUTPUT_DIR = "C:/xxq/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TZ = timezone(timedelta(hours=8))
DELIMITER = "\t"


# ============================================================
# 工具函数
# ============================================================
def parse_ts(ts_ms):
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=TZ)

def format_ts(ts_ms):
    return parse_ts(ts_ms).strftime("%Y-%m-%d %H:%M")

def format_hour(ts_ms):
    return parse_ts(ts_ms).strftime("%Y-%m-%d %H:00")

def format_date(ts_ms):
    return parse_ts(ts_ms).strftime("%Y-%m-%d")

def read_tsv(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        return list(reader)

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def load_font(path, size, bold_path=None):
    """尝试加载 TrueType 字体，失败则用默认"""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size)
        except Exception:
            return ImageFont.load_default(size)


# ============================================================
# 1. ER 关系图 → PNG
# ============================================================
def draw_er_diagram(output_path):
    print("=" * 70)
    print("【任务一】绘制 ER 关系图 → PNG")
    print("=" * 70)

    W, H = 3000, 1800
    img = Image.new("RGB", (W, H), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # 字体加载
    FNT_TITLE = load_font("C:/Windows/Fonts/msyhbd.ttc", 52)
    FNT_HEADER = load_font("C:/Windows/Fonts/msyhbd.ttc", 28)
    FNT_HEADER2 = load_font("C:/Windows/Fonts/msyh.ttc", 24)
    FNT_PK = load_font("C:/Windows/Fonts/msyhbd.ttc", 22)
    FNT_FIELD = load_font("C:/Windows/Fonts/msyh.ttc", 22)
    FNT_DESC = load_font("C:/Windows/Fonts/msyh.ttc", 18)
    FNT_REL = load_font("C:/Windows/Fonts/msyhbd.ttc", 32)
    FNT_REL_FIELD = load_font("C:/Windows/Fonts/msyhbd.ttc", 22)
    FNT_NOTE = load_font("C:/Windows/Fonts/msyh.ttc", 20)
    FNT_LEGEND = load_font("C:/Windows/Fonts/msyh.ttc", 18)

    # 颜色
    C_BLUE = "#185FA5"
    C_BLUE_BG = "#E6F1FB"
    C_PURPLE = "#534AB7"
    C_PURPLE_BG = "#EEEDFE"
    C_TEAL = "#0F6E56"
    C_TEAL_BG = "#E1F5EE"
    C_PK = "#A32D2D"
    C_FK = "#378ADD"
    C_TEXT = "#333333"
    C_GRAY = "#888888"

    # ---- 标题 ----
    draw.text((W // 2, 60), "tsar 数据 ER 关系图", font=FNT_TITLE, fill=C_TEXT, anchor="mm")

    # ---- 图例 ----
    draw.text((W // 2, 120), "PK = 主键    FK = 外键    1 : N = 一对多关系", font=FNT_LEGEND, fill=C_GRAY, anchor="mm")

    # =============== 实体盒子参数 ===============
    HDR_H = 100     # 头部高度
    FIELD_GAP = 75   # 字段间距
    FIELD_PAD_TOP = 50  # 第一个字段距头部底边的距离

    # ---- host_detail (左) ----
    h_fields = [
        ("PK", "hostid",   "string", "主机ID"),
        ("",   "hostname", "string", "FQDN名"),
        ("",   "owner",    "string", "负责人"),
        ("",   "model",    "string", "硬件型号"),
        ("",   "location1","string", "机房位置"),
        ("",   "location2","string", "机柜编号"),
    ]
    hx = 100
    hy = 200
    hw = 700
    hh = HDR_H + FIELD_PAD_TOP + len(h_fields) * FIELD_GAP + 40

    # 头部
    draw.rounded_rectangle([(hx, hy), (hx + hw, hy + HDR_H)], radius=10, fill=C_BLUE, outline=C_BLUE, width=2)
    draw.text((hx + hw // 2, hy + HDR_H // 2 - 12), "host_detail", font=FNT_HEADER, fill="white", anchor="mm")
    draw.text((hx + hw // 2, hy + HDR_H // 2 + 18), "(主机信息表)", font=FNT_HEADER2, fill="white", anchor="mm")

    # 身体
    draw.rounded_rectangle([(hx, hy + HDR_H), (hx + hw, hy + hh)], radius=10, fill=C_BLUE_BG, outline=C_BLUE, width=2)

    # 字段
    for i, (marker, name, ftype, desc) in enumerate(h_fields):
        fy = hy + HDR_H + FIELD_PAD_TOP + i * FIELD_GAP
        # 分隔线
        if i > 0:
            draw.line([(hx + 15, fy - FIELD_GAP // 2 + 15), (hx + hw - 15, fy - FIELD_GAP // 2 + 15)], fill="#B5D4F4", width=1)
        # 标记 + 字段名
        if marker == "PK":
            draw.text((hx + 25, fy), "PK ", font=FNT_PK, fill=C_PK, anchor="lt")
            draw.text((hx + 75, fy), name, font=FNT_PK, fill=C_TEXT, anchor="lt")
        else:
            draw.text((hx + 25, fy), name, font=FNT_FIELD, fill=C_TEXT, anchor="lt")
        # 类型
        draw.text((hx + 330, fy), ftype, font=FNT_DESC, fill=C_GRAY, anchor="lt")
        # 说明
        draw.text((hx + 430, fy), desc, font=FNT_DESC, fill=C_GRAY, anchor="lt")

    # ---- tsar_detail (中) ----
    t_fields = [
        ("",   "ts",      "long",   "采集时间戳(ms)"),
        ("FK", "hostid",  "string", "主机ID -> host_detail"),
        ("",   "type",    "string", "disk / pref"),
        ("FK", "mod",     "string", "指标代码 -> mod_detail"),
        ("",   "value",   "string", "采集值"),
        ("",   "tag",     "string", "分类标签"),
    ]
    tx = 1150
    ty = 150
    tw = 700
    th = HDR_H + FIELD_PAD_TOP + len(t_fields) * FIELD_GAP + 40

    draw.rounded_rectangle([(tx, ty), (tx + tw, ty + HDR_H)], radius=10, fill=C_PURPLE, outline=C_PURPLE, width=2)
    draw.text((tx + tw // 2, ty + HDR_H // 2 - 12), "tsar_detail", font=FNT_HEADER, fill="white", anchor="mm")
    draw.text((tx + tw // 2, ty + HDR_H // 2 + 18), "(采集明细表)", font=FNT_HEADER2, fill="white", anchor="mm")

    draw.rounded_rectangle([(tx, ty + HDR_H), (tx + tw, ty + th)], radius=10, fill=C_PURPLE_BG, outline=C_PURPLE, width=2)

    for i, (marker, name, ftype, desc) in enumerate(t_fields):
        fy = ty + HDR_H + FIELD_PAD_TOP + i * FIELD_GAP
        if i > 0:
            draw.line([(tx + 15, fy - FIELD_GAP // 2 + 15), (tx + tw - 15, fy - FIELD_GAP // 2 + 15)], fill="#CECBF6", width=1)
        if marker == "FK":
            draw.text((tx + 25, fy), "FK ", font=FNT_PK, fill=C_FK, anchor="lt")
            draw.text((tx + 75, fy), name, font=FNT_PK, fill=C_TEXT, anchor="lt")
        else:
            draw.text((tx + 25, fy), name, font=FNT_FIELD, fill=C_TEXT, anchor="lt")
        draw.text((tx + 330, fy), ftype, font=FNT_DESC, fill=C_GRAY, anchor="lt")
        draw.text((tx + 430, fy), desc, font=FNT_DESC, fill=C_GRAY, anchor="lt")

    # ---- mod_detail (右) ----
    m_fields = [
        ("PK", "mod",   "string", "指标代码"),
        ("",   "type",  "string", "disk / pref"),
        ("",   "desc",  "string", "中文说明"),
        ("",   "unit",  "string", "单位"),
        ("",   "tag",   "string", "分类标签"),
    ]
    mx = 2200
    my = 250
    mw = 700
    mh = HDR_H + FIELD_PAD_TOP + len(m_fields) * FIELD_GAP + 40

    draw.rounded_rectangle([(mx, my), (mx + mw, my + HDR_H)], radius=10, fill=C_TEAL, outline=C_TEAL, width=2)
    draw.text((mx + mw // 2, my + HDR_H // 2 - 12), "mod_detail", font=FNT_HEADER, fill="white", anchor="mm")
    draw.text((mx + mw // 2, my + HDR_H // 2 + 18), "(指标字典表)", font=FNT_HEADER2, fill="white", anchor="mm")

    draw.rounded_rectangle([(mx, my + HDR_H), (mx + mw, my + mh)], radius=10, fill=C_TEAL_BG, outline=C_TEAL, width=2)

    for i, (marker, name, ftype, desc) in enumerate(m_fields):
        fy = my + HDR_H + FIELD_PAD_TOP + i * FIELD_GAP
        if i > 0:
            draw.line([(mx + 15, fy - FIELD_GAP // 2 + 15), (mx + mw - 15, fy - FIELD_GAP // 2 + 15)], fill="#9FE1CB", width=1)
        if marker == "PK":
            draw.text((mx + 25, fy), "PK ", font=FNT_PK, fill=C_PK, anchor="lt")
            draw.text((mx + 75, fy), name, font=FNT_PK, fill=C_TEXT, anchor="lt")
        else:
            draw.text((mx + 25, fy), name, font=FNT_FIELD, fill=C_TEXT, anchor="lt")
        draw.text((mx + 330, fy), ftype, font=FNT_DESC, fill=C_GRAY, anchor="lt")
        draw.text((mx + 430, fy), desc, font=FNT_DESC, fill=C_GRAY, anchor="lt")

    # =============== 关系线 ===============
    # host_detail 1:N tsar_detail (via hostid)
    line_y1 = hy + HDR_H + FIELD_PAD_TOP + 1 * FIELD_GAP  # 对齐 hostid 字段行
    # 从 host_detail 右边到 tsar_detail 左边
    x1_start = hx + hw
    x1_end = tx
    draw.line([(x1_start, line_y1), (x1_end, line_y1)], fill=C_BLUE, width=3)

    # 箭头（指向 N 端，即 tsar_detail）
    arrow_size = 20
    draw.polygon([
        (x1_end - 2, line_y1),
        (x1_end - arrow_size, line_y1 - arrow_size // 2),
        (x1_end - arrow_size, line_y1 + arrow_size // 2),
    ], fill=C_BLUE)

    # "1" 标签
    draw.text((x1_start + 15, line_y1 - 35), "1", font=FNT_REL, fill=C_BLUE, anchor="lt")
    # "N" 标签
    draw.text((x1_end - 45, line_y1 - 35), "N", font=FNT_REL, fill=C_BLUE, anchor="lt")
    # 关联字段名
    mid_x1 = (x1_start + x1_end) // 2
    draw.rounded_rectangle([(mid_x1 - 60, line_y1 + 8), (mid_x1 + 60, line_y1 + 38)], radius=4, fill="white", outline=C_BLUE, width=1)
    draw.text((mid_x1, line_y1 + 23), "hostid", font=FNT_REL_FIELD, fill=C_BLUE, anchor="mm")

    # mod_detail 1:N tsar_detail (via mod)
    line_y2 = ty + HDR_H + FIELD_PAD_TOP + 3 * FIELD_GAP  # 对齐 mod 字段行
    x2_start = mx
    x2_end = tx + tw
    draw.line([(x2_start, line_y2), (x2_end, line_y2)], fill=C_TEAL, width=3)

    # 箭头（指向 N 端，即 tsar_detail）
    draw.polygon([
        (x2_end + 2, line_y2),
        (x2_end + arrow_size, line_y2 - arrow_size // 2),
        (x2_end + arrow_size, line_y2 + arrow_size // 2),
    ], fill=C_TEAL)

    draw.text((x2_start - 40, line_y2 - 35), "1", font=FNT_REL, fill=C_TEAL, anchor="lt")
    draw.text((x2_end + 15, line_y2 - 35), "N", font=FNT_REL, fill=C_TEAL, anchor="lt")
    mid_x2 = (x2_start + x2_end) // 2
    draw.rounded_rectangle([(mid_x2 - 45, line_y2 + 8), (mid_x2 + 45, line_y2 + 38)], radius=4, fill="white", outline=C_TEAL, width=1)
    draw.text((mid_x2, line_y2 + 23), "mod", font=FNT_REL_FIELD, fill=C_TEAL, anchor="mm")

    # =============== 底部注释 ===============
    note_y = H - 80
    note_text = "tsar_detail = disk_tsar.dat (type=disk)  +  pref_tsar.dat (type=pref)  |  分隔符: Tab  |  时间戳: 毫秒级  |  时区: UTC+8"
    draw.rounded_rectangle([(100, note_y - 15), (W - 100, note_y + 35)], radius=8, fill="#F8F8F8", outline="#CCCCCC", width=1)
    draw.text((W // 2, note_y + 10), note_text, font=FNT_NOTE, fill=C_GRAY, anchor="mm")

    # 保存
    img.save(output_path, "PNG", dpi=(150, 150))
    print(f"  [已保存] ER 关系图 -> {output_path}")
    print(f"  图片尺寸: {W}x{H} 像素, DPI=150")
    return output_path


# ============================================================
# 2. 时间戳解析 + 按天汇总
# ============================================================
def task_timestamp_and_daily(disk_rows, pref_rows, mod_lookup):
    print("\n" + "=" * 70)
    print("【任务二】时间戳解析 + 按天汇总（仅性能数据）")
    print("=" * 70)

    print("""
  毫秒级时间戳解析方法：
    datetime.fromtimestamp(ts_ms / 1000, tz=timezone(timedelta(hours=8)))
      - ts_ms / 1000 -> 毫秒转秒
      - tz=UTC+8     -> 东八区（北京时间）

  判断最小粒度后决定汇总级别：
    - 磁盘数据 (disk_tsar): 每5分钟采样 -> 最小粒度为【分钟级】-> 按小时汇总
    - 性能数据 (pref_tsar): 每小时采样   -> 最小粒度为【小时级】-> 按天汇总
""")

    # 收集时间戳
    disk_ts_set = set(int(r["ts"]) for r in disk_rows)
    pref_ts_set = set(int(r["ts"]) for r in pref_rows)
    all_ts = sorted(disk_ts_set | pref_ts_set)

    ts_min, ts_max = all_ts[0], all_ts[-1]
    print(f"  数据时间范围: {format_ts(ts_min)} ~ {format_ts(ts_max)}")

    # 判断最小粒度
    disk_sorted = sorted(disk_ts_set)
    pref_sorted = sorted(pref_ts_set)

    if len(disk_sorted) >= 2:
        disk_interval = disk_sorted[1] - disk_sorted[0]
        print(f"  磁盘采样间隔: {disk_interval/60000:.1f} 分钟 -> 最小粒度: 分钟级 -> 按小时汇总即可")

    if len(pref_sorted) >= 2:
        pref_interval = pref_sorted[1] - pref_sorted[0]
        print(f"  性能采样间隔: {pref_interval/3600000:.1f} 小时 -> 最小粒度: 小时级 -> 额外按天汇总")

    # ---- 保存所有时间戳解析 ----
    ts_output = os.path.join(OUTPUT_DIR, "timestamp_parsed.tsv")
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    with open(ts_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(["ts_ms", "readable_time", "hour_bucket", "date", "weekday", "source"])
        for ts in all_ts:
            dt = parse_ts(ts)
            source = "disk+pref" if ts in disk_ts_set and ts in pref_ts_set else \
                     "disk" if ts in disk_ts_set else "pref"
            writer.writerow([
                ts, dt.strftime("%Y-%m-%d %H:%M:%S"),
                format_hour(ts), format_date(ts),
                weekdays[dt.weekday()], source
            ])
    print(f"  [已保存] 时间戳解析表 -> {ts_output} ({len(all_ts)} 行)")

    # ---- 性能数据按天汇总（时间戳维度） ----
    pref_date_hours = defaultdict(set)
    pref_date_ts = defaultdict(set)
    for ts in pref_ts_set:
        date = format_date(ts)
        pref_date_hours[date].add(format_hour(ts))
        pref_date_ts[date].add(ts)

    daily_ts_output = os.path.join(OUTPUT_DIR, "daily_timestamp_summary.tsv")
    with open(daily_ts_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=DELIMITER)
        writer.writerow(["date", "weekday", "hour_count", "first_hour", "last_hour", "ts_count"])
        for date in sorted(pref_date_hours.keys()):
            hours = sorted(pref_date_hours[date])
            dt = datetime.strptime(date, "%Y-%m-%d")
            writer.writerow([
                date, weekdays[dt.weekday()],
                len(hours), hours[0], hours[-1],
                len(pref_date_ts[date])
            ])
    print(f"  [已保存] 按天时间戳汇总 -> {daily_ts_output}")

    # ---- 性能指标按天汇总（指标值维度） ----
    pref_by_day = defaultdict(list)
    for r in pref_rows:
        ts = int(r["ts"])
        date = format_date(ts)
        val = safe_float(r["value"])
        if val is not None:
            key = (r["hostid"], r["mod"], date)
            pref_by_day[key].append((ts, val))

    daily_headers = [
        "date", "hostid", "mod", "type", "desc", "unit", "tag",
        "daily_avg", "daily_max", "daily_min", "hour_count",
        "first_hour", "last_hour", "first_hour_readable", "last_hour_readable"
    ]
    daily_results = []
    for (hostid, mod, date), samples in pref_by_day.items():
        values = [v for _, v in samples]
        ts_list = [t for t, _ in samples]
        mod_info = mod_lookup.get(mod, {})
        daily_results.append({
            "date": date, "hostid": hostid, "mod": mod,
            "type": mod_info.get("type", "pref"),
            "desc": mod_info.get("desc", ""),
            "unit": mod_info.get("unit", ""),
            "tag": mod_info.get("tag", ""),
            "daily_avg": round(statistics.mean(values), 2),
            "daily_max": round(max(values), 2),
            "daily_min": round(min(values), 2),
            "hour_count": len(values),
            "first_hour": format_hour(min(ts_list)),
            "last_hour": format_hour(max(ts_list)),
            "first_hour_readable": format_ts(min(ts_list)),
            "last_hour_readable": format_ts(max(ts_list)),
        })

    daily_results.sort(key=lambda x: (x["date"], x["hostid"], x["mod"]))

    daily_agg_output = os.path.join(OUTPUT_DIR, "daily_aggregation_pref.tsv")
    with open(daily_agg_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=daily_headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(daily_results)
    print(f"  [已保存] 性能指标按天汇总 -> {daily_agg_output} ({len(daily_results)} 行)")

    # 示例展示
    print(f"\n  ■ 按天汇总示例（host001, cpu_usage）：")
    print(f"    {'日期':<14} {'日均值':>10} {'日最大':>10} {'日最小':>10} {'小时数':>8}")
    shown = 0
    for r in daily_results:
        if r["hostid"] == "host001" and r["mod"] == "cpu_usage" and shown < 7:
            print(f"    {r['date']:<14} {r['daily_avg']:>10} {r['daily_max']:>10} {r['daily_min']:>10} {r['hour_count']:>8}")
            shown += 1


# ============================================================
# 3. 按小时汇总指标
# ============================================================
def task_hourly_aggregation(disk_rows, pref_rows, mod_lookup):
    print("\n" + "=" * 70)
    print("【任务三】按小时汇总指标")
    print("=" * 70)

    print("""
  按 (hostid, mod, hour_bucket) 三元组分组，每组计算：
    - avg_value:  该小时内所有采样值的平均值
    - max_value:  该小时内所有采样值的最大值
    - sample_count: 该小时内的采样次数（分钟级采样数）
""")

    all_rows = disk_rows + pref_rows
    print(f"  合并总记录数: {len(all_rows)}")

    # 分组
    groups = defaultdict(list)
    for r in all_rows:
        ts = int(r["ts"])
        hour = format_hour(ts)
        val = safe_float(r["value"])
        if val is not None:
            key = (r["hostid"], r["mod"], hour)
            groups[key].append((ts, val))

    # 汇总
    hourly_headers = [
        "hour", "hostid", "mod", "type", "desc", "unit", "tag",
        "avg_value", "max_value", "min_value", "sample_count",
        "min_ts", "max_ts", "min_ts_readable", "max_ts_readable"
    ]
    results = []
    for (hostid, mod, hour), samples in groups.items():
        values = [v for _, v in samples]
        ts_list = [t for t, _ in samples]
        mod_info = mod_lookup.get(mod, {})
        results.append({
            "hour": hour, "hostid": hostid, "mod": mod,
            "type": mod_info.get("type", ""),
            "desc": mod_info.get("desc", ""),
            "unit": mod_info.get("unit", ""),
            "tag": mod_info.get("tag", ""),
            "avg_value": round(statistics.mean(values), 2),
            "max_value": round(max(values), 2),
            "min_value": round(min(values), 2),
            "sample_count": len(values),
            "min_ts": min(ts_list), "max_ts": max(ts_list),
            "min_ts_readable": format_ts(min(ts_list)),
            "max_ts_readable": format_ts(max(ts_list)),
        })

    results.sort(key=lambda x: (x["hour"], x["hostid"], x["mod"]))

    # 全量
    agg_output = os.path.join(OUTPUT_DIR, "hourly_aggregation.tsv")
    with open(agg_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hourly_headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(results)
    print(f"  [已保存] 全量小时汇总 -> {agg_output} ({len(results)} 行)")

    # 分 disk / pref
    disk_results = [r for r in results if r["type"] == "disk"]
    pref_results = [r for r in results if r["type"] == "pref"]

    disk_out = os.path.join(OUTPUT_DIR, "hourly_aggregation_disk.tsv")
    with open(disk_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hourly_headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(disk_results)
    print(f"  [已保存] 磁盘小时汇总 -> {disk_out} ({len(disk_results)} 行)")

    pref_out = os.path.join(OUTPUT_DIR, "hourly_aggregation_pref.tsv")
    with open(pref_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hourly_headers, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(pref_results)
    print(f"  [已保存] 性能小时汇总 -> {pref_out} ({len(pref_results)} 行)")

    # 示例
    print(f"\n  ■ 性能小时汇总示例（host001, cpu_usage 前5行）：")
    print(f"    {'小时':<18} {'avg':>10} {'max':>10} {'采样数':>8}")
    shown = 0
    for r in pref_results:
        if r["hostid"] == "host001" and r["mod"] == "cpu_usage" and shown < 5:
            print(f"    {r['hour']:<18} {r['avg_value']:>10} {r['max_value']:>10} {r['sample_count']:>8}")
            shown += 1


# ============================================================
# 输出清单 README
# ============================================================
def write_readme():
    readme_path = os.path.join(OUTPUT_DIR, "README.txt")
    files_info = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if fname == "README.txt":
            continue
        fsize = os.path.getsize(fpath)
        ext = os.path.splitext(fname)[1]
        size_str = f"{fsize/1024:.1f}KB" if fsize < 1024*1024 else f"{fsize/1024/1024:.1f}MB"
        files_info.append((fname, ext, size_str))

    content = """tsar 数据处理结果
==================

数据来源: C:/xxq/ (host_detail.dat, mod_detail.dat, disk_tsar.dat, pref_tsar.dat)
处理脚本: C:/xxq/process_tsar_v2.py
输出时间: 2026-07-08

输出文件清单：
"""
    for fname, ext, size_str in files_info:
        desc_map = {
            ".png": "ER 关系图（图片）",
            ".tsv": "数据表（Tab 分隔）",
        }
        desc = desc_map.get(ext, "")
        content += f"  {fname}  ({size_str})  {desc}\n"

    content += """
文件说明：
  er_diagram.png             - 四表 ER 实体关系图（PNG 图片）
  timestamp_parsed.tsv       - 所有时间戳解析结果（毫秒 → 可读时间 + 小时桶 + 日期 + 星期）
  daily_timestamp_summary.tsv - 性能数据按天的时间戳汇总（最小粒度=小时，故按天汇总）
  daily_aggregation_pref.tsv - 性能指标按天汇总（avg/max/min/hour_count）
  hourly_aggregation.tsv     - 全量按小时汇总（disk + pref）
  hourly_aggregation_disk.tsv - 磁盘指标按小时汇总
  hourly_aggregation_pref.tsv - 性能指标按小时汇总

ER 关系：
  host_detail  1:N  tsar_detail (via hostid)
  mod_detail   1:N  tsar_detail (via mod)
  tsar_detail = disk_tsar.dat (type=disk) + pref_tsar.dat (type=pref)

时间戳解析规则：
  磁盘数据 (disk_tsar): 采样间隔 5 分钟 → 最小粒度为分钟级 → 按小时汇总
  性能数据 (pref_tsar): 采样间隔 1 小时 → 最小粒度为小时级 → 按天汇总

汇总字段说明：
  avg_value / daily_avg  : 采样值平均值
  max_value / daily_max  : 采样值最大值
  min_value / daily_min  : 采样值最小值
  sample_count           : 分钟级采样数（按小时汇总时）
  hour_count             : 小时级采样数（按天汇总时）
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  [已保存] README -> {readme_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("  tsar 数据处理脚本 v2")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    # 读取数据
    host_rows = read_tsv(os.path.join(DATA_DIR, "host_detail.dat"))
    mod_rows = read_tsv(os.path.join(DATA_DIR, "mod_detail.dat"))
    disk_rows = read_tsv(os.path.join(DATA_DIR, "disk_tsar.dat"))
    pref_rows = read_tsv(os.path.join(DATA_DIR, "pref_tsar.dat"))

    mod_lookup = {}
    for r in mod_rows:
        mod_lookup[r["mod"]] = r

    print(f"  host_detail: {len(host_rows)} | mod_detail: {len(mod_rows)} | "
          f"disk_tsar: {len(disk_rows)} | pref_tsar: {len(pref_rows)}")

    # 1. ER 关系图 PNG
    draw_er_diagram(os.path.join(OUTPUT_DIR, "er_diagram.png"))

    # 2. 时间戳解析 + 按天汇总
    task_timestamp_and_daily(disk_rows, pref_rows, mod_lookup)

    # 3. 按小时汇总
    task_hourly_aggregation(disk_rows, pref_rows, mod_lookup)

    # 输出清单
    write_readme()

    print("\n" + "=" * 70)
    print("  全部完成！输出文件：")
    print("=" * 70)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        fsize = os.path.getsize(fpath)
        ext = os.path.splitext(fname)[1]
        size_str = f"{fsize/1024:.1f}KB" if fsize < 1024*1024 else f"{fsize/1024/1024:.1f}MB"
        print(f"  {fname:<45} {size_str:>10}  {ext}")


if __name__ == "__main__":
    main()
