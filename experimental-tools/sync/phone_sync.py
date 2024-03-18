import os, sys
import subprocess
import argparse
import time
import json
from tqdm import tqdm


if __name__ == "__main__":
    with open('../device_to_serial.json', 'r') as f:
        json_data = json.load(f)
        device_to_serial = json_data["device_to_serial"]
        serial_to_device = json_data["serial_to_device"]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--devices", type=str, nargs='+', help="list of devices", default=["unam"])  # input list of devices sep by 'space'
    parser.add_argument("-pd", "--period", type=int, help="period of time synchronization (second)", default=300)
    parser.add_argument("-H", "--host", type=str, help="host: e.g., 140.112.xx.xxx", default="140.112.20.183")
    parser.add_argument("-p", "--port", type=int, help="port", default=3298)
    parser.add_argument("-n", "--number", type=int, help="number of packet per round", default=500)
    args = parser.parse_args()
    
    count_down = args.period - 10

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


    while True:
        subprocess.Popen([f'python3 time_sync.py -c -H {args.host} -p {args.port} -n {args.number}'], shell=True)
        for device, serial in zip(devices, serials):
            su_cmd = f'cd /sdcard/wmnl-handoff-research/experimental-tool/sync && python3 time_sync.py -c -H {args.host} -p {args.port} -n {args.number}'
            adb_cmd = f"su -c '{su_cmd}'"
            subprocess.Popen([f'adb -s {serial} shell "{adb_cmd}"'], shell=True)
        
        for i in tqdm(range(args.period)):
            time.sleep(1)
        # time.sleep(count_down)
        # for i in range(10, 0, -1):
        #     print(i); time.sleep(1)
