#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./auto_monitor_mobile_all.py

import os
import sys
import time
import subprocess

os.system("echo wmnlab | sudo -S su")

devices = sorted(os.listdir("/sys/class/net/"))
devices = [dev for dev in devices if dev.startswith('qc')]
print(devices)

LTE_BAND = "1:2:3:4:5:7:8:12:13:14:17:18:19:20:25:26:28:29:30:32:34:38:39:40:41:42:43:46:48:66:71"
NSA_BAND = "1:2:3:5:7:8:12:20:25:28:38:40:41:48:66:71:77:78:79:257:258:260:261"
for dev in devices:
    print("sudo ./band-setting.sh -i {} -l {} -e {}".format(dev, LTE_BAND, NSA_BAND))
    os.system("sudo ./band-setting.sh -i {} -l {} -e {}".format(dev, LTE_BAND, NSA_BAND))
    time.sleep(1)
