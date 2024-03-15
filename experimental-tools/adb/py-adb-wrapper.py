#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://gist.github.com/TheHippo/9329179

# Command Usage:
# ./py-adb-wrapper.py ADB_COMMAND
# ./py-adb-wrapper.py shell
# 1 - R5CRA1ET22M [device usb:1-4.4 product:a42xqxx model:SM_A426B device:a42xq transport_id:13] sm07
# 2 - R5CRA2EGJ5X [device usb:1-4.3 product:a42xqxx model:SM_A426B device:a42xq transport_id:14] sm08

import subprocess
import sys
import os
from device_to_serial import device_to_serial, serial_to_device



def parse_device_list(str):
    raw_list = [x.decode("utf-8") for x in str.splitlines() if x != b''][1:]
    devices = []
    for raw_device in raw_list:
        parts = raw_device.split()
        try:
            devices.append((parts[0], " ".join(parts[1:]), serial_to_device[parts[0]]))
        except:
            devices.append((parts[0], " ".join(parts[1:]), "-"))
    if not devices:
        devices = None
    return devices

def get_device_list():
    p = subprocess.Popen(['adb', 'devices', '-l'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if len(err) == 0:
        return parse_device_list(out)
    else:
        return

def print_device_list(devices):
    for i in range(len(devices)):
        id, name, device = devices[i]
        # print("%d - %s [%s] %s" % (i+1, id, name, device))
        print("%d - %s %s [%s]" % (i+1, id, device, name))

def num(s):
    try:
        return int(s)
    except:
        return s


devices = get_device_list()

if devices == None:
    print("Could not find any device")
    sys.exit(1)

devices = sorted(devices, key=lambda v:v[2])
print_device_list(devices)

if len(sys.argv) == 1:
    print("You could specify other adb command as parameters. (default: adb shell)")
    sys.exit(1)

while True:
    if len(devices) == 1:
        choosen = 1
        break
    choosen = num(sys.stdin.readline())
    if type(choosen) is not int:
        print("Please specify a number.")
        print_device_list(devices)
        continue
    if (choosen != None and choosen > 0 and choosen <= len(devices)):
        break

os.environ['ANDROID_SERIAL'] = devices[choosen - 1][0]

print("Executing the following command:")
if len(sys.argv) == 1:
    print("    adb -s {} shell".format(devices[choosen - 1][0]))
    p = subprocess.Popen("adb shell", shell=True)
else:
    print("    adb -s {} {}".format(devices[choosen - 1][0], ' '.join(sys.argv[1:])))
    p = subprocess.Popen(["adb"] + sys.argv[1:])

try:
	p.communicate()
except KeyboardInterrupt:
	pass
