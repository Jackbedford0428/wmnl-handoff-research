#!/usr/bin/env python3

# Command Usage:
# pip3 install adbutils
# ./set-tools-mobile-all.py

from adbutils import adb

serial_to_device = {
    "R5CRA1ET5KB":"sm00",
    "R5CRA1D2MRJ":"sm01",
    "R5CRA1GCHFV":"sm02",
    "R5CRA1JYYQJ":"sm03",
    "R5CRA1EV0XH":"sm04",
    "R5CRA1GBLAZ":"sm05",
    "R5CRA1ESYWM":"sm06",
    "R5CRA1ET22M":"sm07",
    "R5CRA2EGJ5X":"sm08",
    "73e11a9f":"xm00",
    "491d5141":"xm01",
    "790fc81d":"xm02",
    "e2df293a":"xm03",
    "28636990":"xm04",
    "f8fe6582":"xm05",
    "d74749ee":"xm06",
    "10599c8d":"xm07",
    "57f67f91":"xm08",
    "232145e8":"xm09",
    "70e87dd6":"xm10",
    "df7aeaf8":"xm11",
    "e8c1eff5":"xm12",
    "ec32dc1e":"xm13",
    "2aad1ac6":"xm14",
    "64545f94":"xm15",
    "613a273a":"xm16",
    "fe3df56f":"xm17",
}

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

tools = ["git", "iperf3m", "iperf3", "python3", "tcpdump", "tmux", "vim"]
for device, info in zip(devices, devices_info):
    print(info[2], device.shell("su -c 'cd /sdcard/wmnl-handoff-research && /data/git pull'"))
    for tool in tools:
        if info[2][:2] == "sm":
            device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimentation-tools/android/sm-script/termux-tools/{} /bin/'".format(tool))
            device.shell("su -c 'chmod +x /bin/{}'".format(tool))
        elif info[2][2] == "xm":
            device.shell("su -c 'cp /sdcard/wmnl-handoff-research/experimentation-tools/android/xm-script/termux-tools/{} /sbin/'".format(tool))
            device.shell("su -c 'chmod +x /sbin/{}'".format(tool))
    
    # test tools
    print("-----------------------------------")
    print(info[2], device.shell("su -c 'iperf3m --version'"))
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
