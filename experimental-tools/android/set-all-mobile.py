#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./set-tools-mobile-all.py

from adbutils import adb
import time
import json
import argparse


with open('../device_to_serial.json', 'r') as f:
    json_data = json.load(f)
    device_to_serial = json_data["device_to_serial"]
    serial_to_device = json_data["serial_to_device"]
    
with open('../password.txt', 'r', encoding='utf-8') as f:
    password = f.readline().strip()
    
with open('../savedir.txt', 'r', encoding='utf-8') as f:
    savedir = f.readline().strip()
    
parser = argparse.ArgumentParser()
parser.add_argument("-rf", "--clear", action="store_true", help="clear data")
args = parser.parse_args()

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

tools = ["git", "iperf3", "python3", "tcpdump", "tmux", "vim"]
for device, info in zip(devices, devices_info):
    # print(info[2], device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git fetch && /data/git checkout -b experiment origin/experiment'"))
    print(info[2], device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git fetch && /data/git checkout experiment && /data/git pull'"))
    print("-----------------------------------")
    if info[2][:2] == "sm":
        device.shell("su -c 'mount -o remount,rw /system/bin'")
        for tool in tools:
            device.shell("su -c 'cp /data/{} /bin'".format(tool))
            device.shell("su -c 'chmod +x /bin/{}'".format(tool))
            # device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools/android/sm-script/termux-tools/{} /bin'".format(tool))
            # device.shell("su -c 'chmod +x /bin/{}'".format(tool))
    elif info[2][2] == "xm":
        # device.shell("su -c 'mount -o remount,rw /system/sbin'")
        for tool in tools:
            device.shell("su -c 'cp /data/{} /sbin'".format(tool))
            device.shell("su -c 'chmod +x /sbin/{}'".format(tool))
            # device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools/android/xm-script/termux-tools/{} /sbin'".format(tool))
            # device.shell("su -c 'chmod +x /sbin/{}'".format(tool))
    
    # test tools
    print(info[2], 'git:', device.shell("su -c 'git --version'"))
    print(device.shell("su -c 'cd /sdcard/wmnl-handoff-research && git branch'"))
    print("-----------------------------------")
    print(info[2], 'iperf3:', device.shell("su -c 'iperf3 --version'"))
    print("-----------------------------------")
    print(info[2], 'tcpdump:', device.shell("su -c 'tcpdump --version'"))
    print("-----------------------------------")
    print(info[2], 'python3:', device.shell("su -c 'python3 --version'"))
    print("-----------------------------------")
    
    # UDP_Phone
    su_cmd = 'rm -rf /sdcard/UDP_Phone && cp -r /sdcard/wmnl-handoff-research/experimental-tools/udp-socket-programming/v3/UDP_Phone /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(su_cmd)
    print(info[2], 'Update UDP_Phone! v3')
    print("-----------------------------------")
    
    # TCP_Phone
    su_cmd = 'rm -rf /sdcard/TCP_Phone && cp -r /sdcard/wmnl-handoff-research/experimental-tools/tcp-socket-programming/v3/TCP_Phone /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(su_cmd)
    print(info[2], 'Update TCP_Phone! v3')
    print("-----------------------------------")
    
    # Clear data
    if args.clear:
        su_cmd = 'rm -rf /sdcard/pcapdir && rm -rf /sdcard/experiment_log'
        adb_cmd = f"su -c '{su_cmd}'"
        device.shell(su_cmd)
        print(info[2], 'Clear data: /sdcard/pcapdir && /sdcard/experiment_log')
        print("-----------------------------------")
    
    time.sleep(2.5)

print('---End Of File---')
