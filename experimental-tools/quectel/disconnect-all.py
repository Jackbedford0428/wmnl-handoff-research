#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./auto_monitor_mobile_all.py

import os
import sys
import time
import subprocess

os.system("echo wmnlab | sudo -S su")
os.system("./get-all-modem.py")

devices = sorted(os.listdir("/sys/class/net/"))
devices = [dev for dev in devices if dev.startswith('qc')]
print(devices)

for dev in devices:
    os.system("sudo ./disconnect-qmi.sh -i {}".format(dev))
    time.sleep(1)

os.system("sudo rm -rf temp")
