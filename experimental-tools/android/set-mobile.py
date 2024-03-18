#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./set-tools-mobile.py -d DEVICE_NAME
# ./set-tools-mobile.py -d sm01

from adbutils import adb
import argparse
import sys
import json


with open('../device_to_serial.json', 'r') as f:
    json_data = json.load(f)
    device_to_serial = json_data["device_to_serial"]
    serial_to_device = json_data["serial_to_device"]
    
with open('../password.txt', 'r', encoding='utf-8') as f:
    password = f.readline().strip()
    
with open('../savedir.txt', 'r', encoding='utf-8') as f:
    savedir = f.readline().strip()

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str,
                    help="device name (allows only 1 device)", required=True)
args = parser.parse_args()


if adb.list()[0].state != "unauthorized":
    if args.device:
        device = adb.device(device_to_serial[args.device])
    else:
        device = adb.device()
else:
    print(args.device, "adbutils.errors.AdbError: device unauthorized.")
    sys.exit(1)

print(args.device, device)
print("-----------------------------------")

tools = ["git", "iperf3m", "iperf3", "python3", "tcpdump", "tmux", "vim"]  
print(device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git pull'"))
for tool in tools:
    if args.device[:2] == "sm":
        device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools/android/sm-script/termux-tools/{} /bin/'".format(tool))
        device.shell("su -c 'chmod +x /bin/{}'".format(tool))
    elif args.device[:2] == "xm":
        device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools/android/xm-script/termux-tools/{} /sbin/'".format(tool))
        device.shell("su -c 'chmod +x /sbin/{}'".format(tool))
print("-----------------------------------")

# test tools
print(device.shell("su -c 'iperf3m --version'"))
print("-----------------------------------")
# print(device.shell("su -c 'iperf3 --version'"))
# print("-----------------------------------")
# print(device.shell("su -c 'tcpdump --version'"))
# print("-----------------------------------")
# print(device.shell("su -c 'git --version'"))
# print("-----------------------------------")
# print(device.shell("su -c 'python3 --version'"))
# print("-----------------------------------")
# print(device.shell("su -c 'tmux -V'"))
# print("-----------------------------------")
# print(device.shell("su -c 'vim --version'"))
# print("-----------------------------------")
# print(device.shell("su -c 'iperf3m -c 140.112.20.183 -p 3270 -l 250 -b 200k -V -u'"))
# print("-----------------------------------")
# print(device.shell("su -c 'iperf3m -s'"))
# print("-----------------------------------")
