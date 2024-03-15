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

for dev in devices:
    print("sudo ./modem-info.sh -i {} -t 1 &".format(dev))
    os.system("sudo ./modem-info.sh -i {} -t 1 &".format(dev))
    time.sleep(0.001)

while True:
    try:
        time.sleep(1)  # detect every second
    except KeyboardInterrupt:
        os.system("sudo rm looping")
        break
    except Exception as e:
        print("error", e)
