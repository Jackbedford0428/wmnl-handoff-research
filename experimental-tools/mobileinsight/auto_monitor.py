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
from device_to_serial import device_to_serial, serial_to_device

# device_to_serial = dict((v, k) for k, v in serial_to_device.items())

def makedir(dirpath, mode=0):  # mode=1: show message, mode=0: hide message
    if os.path.isdir(dirpath):
        if mode:
            print("mkdir: cannot create directory '{}': directory has already existed.".format(dirpath))
        return
    ### recursively make directory
    _temp = []
    while not os.path.isdir(dirpath):
        _temp.append(dirpath)
        dirpath = os.path.dirname(dirpath)
    while _temp:
        dirpath = _temp.pop()
        print("mkdir", dirpath)
        os.mkdir(dirpath)

now = dt.datetime.today()
date = [str(x) for x in [now.year, now.month, now.day]]
date = [x.zfill(2) for x in date]
date = '-'.join(date)
makedir("./log/{}".format(date))

mi2log_path = "./log/{}/{}".format(date, "mi2log")  # mobileinsight log
makedir(mi2log_path)

os.system("echo wmnlab | sudo -S su")

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
### Check All Devices (including mobiles and modems)
devices = sorted(os.listdir("/sys/class/net/"))
devices = [dev for dev in devices if dev.startswith('qc')]
for i, info in enumerate(mobile_info):
    devices.append(info[2])

print("Device List:", devices)
print("----------------------------------------------")
time.sleep(3)

### Run MobileInsight
run_list = []
for i, dev in enumerate(devices):
    run_store = subprocess.Popen("sudo python3 monitor.py -d {} -b 9600".format(dev), shell=True, preexec_fn=os.setpgrp)
    run_list.append(run_store)

for item in run_list:
    print(item.pid)

print("Start logging...")
    
### Kill Python3 sessions when capturing KeyboardInterrupt
while True:
    try:
        time.sleep(1)  # detect every second
    except KeyboardInterrupt:
        os.system("ps -ef | grep python3 > ps_temp.txt")
        with open('ps_temp.txt', 'r') as fp:
            lines = fp.readlines()
        infos = [[] for i in range(len(lines))]
        for i, line in enumerate(lines):
            infos[i] = [s for s in line[:52].split(' ') if s]
            infos[i].append(line[52:-1])
        kill_list = []
        for info in infos:
            if info[7].startswith("python3 monitor.py"):
                kill_list.append(info[1])
        for info in infos:
            if info[7].startswith("python3 ./auto_monitor.py"):
                kill_list.append(info[1])
        for pid in kill_list[:-1]:
            os.system("sudo kill -9 {}".format(pid))
        os.system("sudo rm ps_temp.txt")
        os.system("sudo kill -9 {}".format(kill_list[-1]))
        break
    except Exception as e:
        print("error", e)
