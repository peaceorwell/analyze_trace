#!/usr/bin/env python3
"""
Get Basic Info from cnperf database
获取基本信息：Host/Device 时间范围、设备信息、卡使用情况
"""

import sqlite3
import json
import sys

def get_basic_info(db_path):
    """获取基本信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT MIN(start), MAX(end) FROM function_data")
    host_min, host_max = cursor.fetchone()

    cursor.execute("SELECT MIN(start), MAX(end) FROM device_task_kernel_data")
    dev_min, dev_max = cursor.fetchone()

    cursor.execute("SELECT * FROM device_information")
    devices = cursor.fetchall()

    cursor.execute("""
        SELECT deviceId, COUNT(*) as cnt, SUM(end-start) as total
        FROM device_task_kernel_data
        GROUP BY deviceId
    """)
    usage = cursor.fetchall()

    cursor.execute("SELECT * FROM meta_information")
    meta = {m[0]: m[1] for m in cursor.fetchall()}

    conn.close()

    return {
        'host_time': (host_min, host_max),
        'device_time': (dev_min, dev_max),
        'devices': devices,
        'usage': usage,
        'meta': meta
    }

def print_basic_info(db_path):
    """打印基本信息"""
    info = get_basic_info(db_path)

    print("=" * 80)
    print("Basic Information")
    print("=" * 80)

    host_min, host_max = info['host_time']
    dev_min, dev_max = info['device_time']

    print(f"\nHost Time: {(host_max-host_min)/1e6:.2f} ms ({host_min/1e6:.2f} ~ {host_max/1e6:.2f} ms)")
    print(f"Device Time: {(dev_max-dev_min)/1e6:.2f} ms ({dev_min/1e6:.2f} ~ {dev_max/1e6:.2f} ms)")

    try:
        device_info = json.loads(info['meta'].get('deviceInfo', '{}'))
        if device_info:
            dev_name = device_info.get('m_dev_basic_info', {}).get('dev_name', 'Unknown')
            dev_count = device_info.get('m_dev_count', 0)
            print(f"\nDevice Name: {dev_name}")
            print(f"Device Count: {dev_count}")
    except:
        pass

    print(f"\n{'DeviceId':<12} {'Event Count':<15} {'Total Time (ms)':<20}")
    print("-" * 50)
    for did, cnt, total in info['usage']:
        print(f"{did:<12} {cnt:<15} {total/1e6:>15,.2f}")

    return info

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "cnperf_data.db"
    print_basic_info(db_path)
