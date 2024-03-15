# Command usage:
# (1) python3 iperf_server_single.py -d LIST_DEVICES
#     python3 iperf_server_single.py -d sm01 sm08
# (2) python3 iperf_server_single.py -d LIST_DEVICES -p LIST_PORTS -S STREAMING_DIRECTION
#     python3 iperf_server_single.py -d sm01 sm08 -p 3270 3271 3272 3273 -S bl
import os
import sys
import datetime as dt
import time
import subprocess
import argparse
import signal
import threading
import re
from device_to_port import device_to_port, port_to_device

# ------------------------------ Add Arguments & Global Variables ------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--devices", type=str, nargs='+',   # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-p", "--ports", type=str, nargs='+',     # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-k", "--keywords", type=str, nargs='+',  # input list of keywords sep by 'space'
                    help="keywords for socket statistics", default=["bytes_sent", "cwnd"])
parser.add_argument("-S", "--stream", type=str,
                    help="streaming direction: uplink (ul), downlink (dl), bi-link (bl, 2 ports/device)", default="bl")
parser.add_argument("-L", "--logfile", action="store_true",
                    help="save iperf output to logfile")
parser.add_argument("-T", "--tsync", action="store_true",     # needs no value, True if set "-S" or "--sync"
                    help="time sync mode")                    # default "experiment" mode
args = parser.parse_args()

# ----------------------------------------- Parameters ------------------------------------------ #
# thread_stop = False
# exit_program = False

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

ports = []
if not args.ports:
    for device in devices:
        ports.append((device_to_port[device][0]))  # default uplink port for each device
        ports.append((device_to_port[device][1]))  # default downlink port for each device
else:
    for port in args.ports:
        if '-' in port:
            start = int(port[:port.find('-')])
            stop = int(port[port.find('-') + 1:]) + 1
            for i in range(start, stop):
                ports.append(i)
            continue
        ports.append(int(port))

print(devices)
print(ports)

# ----------------------------------------- Save Path ------------------------------------------- #
def makedir(dirpath, mode=0):  # mode=1: show message, mode=0: hide message
    if os.path.isdir(dirpath):
        if mode:
            print("mkdir: cannot create directory '{}': directory has already existed.".format(dirpath))
        return
    ### recursively make directory
    _temp = []
    while not os.path.isdir(dirpath):
        _temp.append(dirpath)
        dirpath = os.path.dirname(dirpath)
    while _temp:
        dirpath = _temp.pop()
        print("mkdir", dirpath)
        os.mkdir(dirpath)

now = dt.datetime.today()
date = [str(x) for x in [now.year, now.month, now.day]]
date = [x.zfill(2) for x in date]
date = '-'.join(date)
makedir("./log/{}".format(date))

pcap_path = "./log/{}/{}".format(date, "server_pcap")  # wireshark capture
makedir(pcap_path)
ilog_path = "./log/{}/{}".format(date, "server_ilog")  # iperf log
makedir(ilog_path)
ss_path = "./log/{}/{}".format(date, "server_ss")      # socket statistics (Linux: ss)
makedir(ss_path)

# ----------------------------------- Define Utils Function ------------------------------------- #
def get_ss(device, port, mode, tsync=False):
    global thread_stop
    global n
    global args

    if tsync:
        fp = open(os.path.join(ss_path, "server_ss_{}_{}_{}_{}_tsync.csv".format(mode.upper(), device, port, n)), 'a+')
    else:
        fp = open(os.path.join(ss_path, "server_ss_{}_{}_{}_{}.csv".format(mode.upper(), device, port, n)), 'a+')
    print(fp)
    while not thread_stop:
        # ss --help (Linux/Android)
        # ss -ai src :PORT (server-side command)
        # ss -ai dst :PORT (client-side command)
        proc = subprocess.Popen(["ss -ai src :{}".format(port)], stdout=subprocess.PIPE, shell=True)
        text = proc.communicate()[0].decode()
        lines = text.split('\n')
        for line in lines:
            if any(s in line for s in args.keywords):  # change the keywords if needed
                l = line.strip()
                fp.write(",".join([str(dt.datetime.now())]+ re.split("[: \n\t]", l))+'\n')
                break
        time.sleep(1)
    fp.close()

# ---------------------------------- Transmission / Receiving ----------------------------------- #
thread_stop = False
# exit_program = False

# Avoid need to feed in password for superuser priviledge
os.system("echo wmnlab | sudo -S su")

print("---------------------------------------------------------------------------")
print("Supported port: 3200-3299, even number for Uplink, odd number for Downlink.")
print("---------------------------------------------------------------------------")

# Get time
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])

_l = []          # command list
ss_threads = []  # ss thread command list

print("Main Ports:", ports[::2])
print("Auxiliary Ports:", ports[1::2])
print("----------------------------------------------")
if True:
    ports = ports[::2]
else:
    ports = ports[1::2]

for device, port in zip(devices, ports):
    # tcpdump process
    if args.tsync:
        pcap = os.path.join(pcap_path, "server_pcap_{}_{}_{}_{}_tsync.pcap".format(args.stream.upper(), device, port, n))
    else:
        pcap = os.path.join(pcap_path, "server_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, port, n))
    _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
    # iperf process
    if args.tsync:
        log = os.path.join(ilog_path, "server_log_{}_{}_{}_{}_tsync.log".format(args.stream.upper(), device, port, n))
    else:
        log = os.path.join(ilog_path, "server_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, port, n))
    if args.logfile:
        _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile".format(port, log))
    else:
        _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port))
    # ss
    ss_threads.append(threading.Thread(target = get_ss, args = (device, port, args.stream, args.tsync)))

# Run all commands in the collection
run_list = []  # running session list
for l in ss_threads:
    l.start()
    time.sleep(0.00001)
for l in _l: 
    print(l)
    run_store = subprocess.Popen(l, shell=True, preexec_fn=os.setpgrp)
    run_list.append(run_store)

# Kill iperf3 & tcpdump sessions with PID when detecting KeyboardInterrupt (Ctrl-C,Z)
while True:
    try:
        time.sleep(1)  # detect every second
    except KeyboardInterrupt:
        # subprocess.Popen(["killall -9 iperf3"], shell=True, preexec_fn=os.setsid)
        for run_item in run_list:
            print(run_item, ", PID: ", run_item.pid)
            os.killpg(os.getpgid(run_item.pid), signal.SIGTERM)
            # command = "sudo kill -9 -{}".format(run_item.pid)
            # subprocess.check_output(command.split(" "))
        break
    except Exception as e:
        print("error", e)

thread_stop = True
