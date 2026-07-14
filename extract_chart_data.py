# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
import statistics
import json

with open('output/hourly_aggregation_pref.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

cpu_data = []
mem_data = []
for r in rows:
    if r['hostid'] == 'host001' and r['mod'] == 'cpu_usage':
        cpu_data.append((r['hour'], float(r['avg_value'])))
    if r['hostid'] == 'host001' and r['mod'] == 'mem_used':
        mem_data.append((r['hour'], float(r['avg_value'])))

cpu_data.sort(key=lambda x: x[0])
mem_data.sort(key=lambda x: x[0])

cpu_24 = cpu_data[:24]
mem_24 = mem_data[:24]

print('CPU_LABELS:', json.dumps([d[0][-5:] for d in cpu_24]))
print('CPU_DATA:', json.dumps([d[1] for d in cpu_24]))
print('MEM_LABELS:', json.dumps([d[0][-5:] for d in mem_24]))
print('MEM_DATA:', json.dumps([d[1] for d in mem_24]))

disk_util = defaultdict(list)
with open('output/hourly_aggregation_disk.tsv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for r in reader:
        if r['mod'] == 'sda_util':
            disk_util[r['hour']].append(float(r['avg_value']))

sorted_hours = sorted(disk_util.keys())[:24]
disk_avg = [round(statistics.mean(disk_util[h]), 2) for h in sorted_hours]
print('DISK_LABELS:', json.dumps([h[-5:] for h in sorted_hours]))
print('DISK_DATA:', json.dumps(disk_avg))
