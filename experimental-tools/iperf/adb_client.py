#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./auto_monitor.py

from adbutils import adb
import os
import sys
import time
import subprocess
import datetime as dt
import threading
from device_to_serial import device_to_serial, serial_to_device

# device_to_serial = dict((v, k) for k, v in serial_to_device.items())

### Check Mobile
mobile_info = []
for i, info in enumerate(adb.list()):
    try:
        if info.state == "device":
            # <serial> <device|offline> <device name>
            mobile_info.append((info.serial, info.state, serial_to_device[info.serial]))
        else:
            print("Unauthorized device {}: {} {}".format(serial_to_device[info.serial], info.serial, info.state))
    except:
        print("Unknown device: {} {}".format(info.serial, info.state))

mobile_info = sorted(mobile_info, key=lambda v:v[2])

adb_handles = []
# print mobile info
for i, info in enumerate(mobile_info):
    adb_handles.append(adb.device(info[0]))
    print("{} - {} {} {}".format(i+1, info[0], info[1], info[2]))
# check unauthorized mobile
for info in adb.list():
    if info.state == "unauthorized":
        sys.exit(1)
# getprop
for device, info in zip(adb_handles, mobile_info):
    print(info[2], device.shell("su -c 'getprop sys.usb.config'"))
print("----------------------------------------------")


# TODO
def run_iperf(dev, device):
    if len(sys.argv) < 2:
        print(dev, device.shell(f"""su -c '
                                    cd /sdcard/wmnl-handoff-research/experimental-tools-beta/iperf\n
                                    python3 client.py -d {dev} -u'
                                    """))
    elif sys.argv[1] == "--tsync" or sys.argv[1] == "-T":
        print(dev, device.shell(f"""su -c '
                                    cd /sdcard/wmnl-handoff-research/experimental-tools-beta/iperf\n
                                    python3 client.py -d {dev} -T'
                                    """))

threads = []
for i, (device, info) in enumerate(zip(adb_handles, mobile_info)):
    threads.append(threading.Thread(target = run_iperf, args = (info[2], device)))

for l in threads:
    l.start()

while True:
    try:
        # print(time.time())
        time.sleep(1)
    except KeyboardInterrupt:
        print("************************************************")
        for i, (device, info) in enumerate(zip(adb_handles, mobile_info)):
            print(info[2], device.shell(f"""
                                        su -c '
                                        pkill iperf3\n
                                        pkill tcpdump\n
                                        pkill python3'
                                        """))
        print("Process end!")
        break
