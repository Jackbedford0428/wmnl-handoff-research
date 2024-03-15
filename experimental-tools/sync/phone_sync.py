import os, sys
import subprocess
from device_to_port import device_to_port
from device_to_serial import device_to_serial
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")   # Lab249 外網
parser.add_argument("-d", "--devices", type=str, nargs='+',   # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-p", "--ports", type=str, nargs='+',     # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-b", "--bitrate", type=str,
                    help="target bitrate in bits/sec (0 for unlimited)", default="1M")
parser.add_argument("-l", "--length", type=str,
                    help="length of buffer to read or write in bytes (packet size)", default="250")
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
args = parser.parse_args()

devices = []
for dev in args.devices:
    if '-' in dev:
        pmodel = dev[:2]
        start = int(dev[2:4])
        stop = int(dev[5:7]) + 1
        for i in range(start, stop):
            _dev = "{}{:02d}".format(pmodel, i)
            devices.append(_dev)
        continue
    devices.append(dev)
serials = [device_to_serial[dev] for dev in devices]

print(devices)
print(serials)

time.sleep(3)

for device, serial in zip(devices, serials):
    su_cmd = 'cd sdcard/wmnl-handoff-research/experimental-tools-beta/sync && python3 time_sync.py -c'
    adb_cmd = f"su -c '{su_cmd}'"
    subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True)
    