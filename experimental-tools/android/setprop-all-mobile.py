#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./auto_mi_iperf.py

from adbutils import adb
import os
import sys
from device_to_serial import device_to_serial, serial_to_device



os.system("echo wmnlab | sudo -S su")

devices_info = []
for i, info in enumerate(adb.list()):
    try:
        if info.state == "device":
            # <serial> <device|offline> <device name>
            devices_info.append((info.serial, info.state, serial_to_device[info.serial]))
        else:
            print("Unauthorized device {}: {} {}".format(serial_to_device[info.serial], info.serial, info.state))
    except:
        print("Unknown device: {} {}".format(info.serial, info.state))

devices_info = sorted(devices_info, key=lambda v:v[2])

devices = []
for i, info in enumerate(devices_info):
    devices.append(adb.device(info[0]))
    print("{} - {} {} {}".format(i+1, info[0], info[1], info[2]))
print("-----------------------------------")

for info in adb.list():
    if info.state == "unauthorized":
        sys.exit(1)

# setprop
for device, info in zip(devices, devices_info):
    print(info[2], device.shell("su -c 'getprop sys.usb.config'"))
    print(info[2], device.shell("su -c 'setprop sys.usb.config diag,serial_cdev,rmnet,adb'"))
