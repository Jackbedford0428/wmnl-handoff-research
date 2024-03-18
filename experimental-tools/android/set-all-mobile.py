#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./set-tools-mobile-all.py

from adbutils import adb
from device_to_serial import device_to_serial, serial_to_device

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

# tools = ["git", "iperf3m", "iperf3", "python3", "tcpdump", "tmux", "vim"]
tools = ["git", "iperf3m", "iperf3", "tcpdump", "tmux", "vim"]
for device, info in zip(devices, devices_info):
    print(info[2], device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git pull'"))
    print("-----------------------------------")
    if info[2][:2] == "sm":
        device.shell("su -c 'mount -o remount,rw /system/bin'")
        for tool in tools:
            device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools-beta/android/sm-script/termux-tools/{} /bin'".format(tool))
            device.shell("su -c 'chmod +x /bin/{}'".format(tool))
        device.shell("su -c 'cp /data/python3 /bin'")
        device.shell("su -c 'chmod +x /bin/python3'")
    elif info[2][2] == "xm":
        # device.shell("su -c 'mount -o remount,rw /system/sbin'")
        for tool in tools:
            device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimental-tools-beta/android/xm-script/termux-tools/{} /sbin'".format(tool))
            device.shell("su -c 'chmod +x /sbin/{}'".format(tool))
        device.shell("su -c 'cp /data/python3 /sbin'")
        device.shell("su -c 'chmod +x /sbin/python3'")
    
    # test tools
    print(info[2], 'iperf3m:', device.shell("su -c 'iperf3m --version'"))
    print("-----------------------------------")
    
    # UDP_Phone
    su_cmd = 'rm -rf /sdcard/UDP_Phone && cp -r /sdcard/wmnl-handoff-research/experimental-tools-beta/udp-socket-programming/v3/UDP_Phone /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(su_cmd)
    print(info[2], 'Update UDP_Phone! v3')
    print("-----------------------------------")
    
    # TCP_Phone
    su_cmd = 'rm -rf /sdcard/TCP_Phone && cp -r /sdcard/wmnl-handoff-research/experimental-tools-beta/tcp-socket-programming/v3/TCP_Phone /sdcard'
    adb_cmd = f"su -c '{su_cmd}'"
    device.shell(su_cmd)
    print(info[2], 'Update TCP_Phone! v3')
    print("-----------------------------------")

print('---End Of File---')