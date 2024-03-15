# Command usage (need su priviledge): 
# (1) python3 iperf_client_single.py -d DEVICE [-u]
#     python3 iperf_client_single.py -d sm01 -u
# (2) python3 iperf_client_single.py -d DEVICE -H SERVER_IP -p LIST_PORTS -S STREAMING_DIRECTION
#     python3 iperf_client_single.py -d sm01 -H 140.112.17.209 -p 3270 3271 -S bl
# (3) python3 iperf_client_single.py -d DEVICE -b BITRATE -l PACKET_SIZE -t EXP_TIME
#     python3 iperf_client_single.py -d sm01 -b 2M -l 2500 -t 300
import os
import sys
import time
import socket
import threading
import datetime as dt
import argparse
import subprocess
import re
import signal
from device_to_port import device_to_port, port_to_device

# ------------------------------ Add Arguments & Global Variables ------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")   # Lab249 外網
                    # help="server ip address", default="192.168.1.251")  # Lab249 內網
                    # help="server ip address", default="210.65.88.213")  # CHT 外網
parser.add_argument("-d", "--devices", type=str, nargs='+',   # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-p", "--ports", type=str, nargs='+',     # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-u", "--udp", action="store_true",       # needs no value, True if set "-u"
                    help="use UDP rather than TCP")           # default TCP
parser.add_argument("-b", "--bitrate", type=str,
                    help="target bitrate in bits/sec (0 for unlimited)", default='1M')
parser.add_argument("-l", "--length", type=str,
                    help="length of buffer to read or write in bytes (packet size)", default='250')
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
parser.add_argument("-k", "--keywords", type=str, nargs='+',  # input list of keywords sep by 'space'
                    help="keywords for socket statistics", default=["bytes_sent", "cwnd"])
parser.add_argument("-S", "--stream", type=str,
                    help="streaming direction: uplink (ul), downlink (dl), bi-link (bl, 2 ports/device)", default="bl")
parser.add_argument("-L", "--logfile", action="store_true",
                    help="save iperf output to logfile")
parser.add_argument("-T", "--tsync", action="store_true",     # needs no value, True if set "-S" or "--sync"
                    help="time sync mode")                    # default "experiment" mode
parser.add_argument("-V", "--private", action="store_true",   # needs no value, True if set "-V" or "--private"
                    help="private mode")                      # default "public" mode
parser.add_argument("-F", "--force", action="store_true",     # needs no value, True if set "-F" or "--force"
                    help="public mode")                       # default "non-force" mode
args = parser.parse_args()

if not args.force:
    if args.private:
        args.host = "192.168.1.251"
    if args.tsync:
        args.host = "192.168.1.251"
        args.udp = True
        # args.bitrate = "4k"
        # args.time = 43200
        args.bitrate = "1M"
        args.time = 60

# ----------------------------------------- Parameters ------------------------------------------ #
# thread_stop = False
# exit_program = False

serverip = args.host

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

is_udp = "-u" if args.udp else ""

bitrate = args.bitrate     # unit: bps
packet_size = args.length  # unit: bytes

if args.stream == "dl":
    is_reverse = "-R"
elif args.stream == "bl":
    is_reverse = "--bidir"
else:  # args.stream == "ul"
    is_reverse = ""

# check whether the device has iperf3m
if os.path.exists("/bin/iperf3m") or os.path.exists("/sbin/iperf3m"):
    iperf = "iperf3m"
else:
    iperf = "iperf3"

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

pcap_path = "./log/{}/{}".format(date, "client_pcap")  # wireshark capture
makedir(pcap_path)
ilog_path = "./log/{}/{}".format(date, "client_ilog")  # iperf log
makedir(ilog_path)
ss_path = "./log/{}/{}".format(date, "client_ss")      # socket statistics (Linux: ss)
makedir(ss_path)

# ----------------------------------- Define Utils Function ------------------------------------- #
def get_network_interface_list():
    pipe = subprocess.Popen('ifconfig', stdout=subprocess.PIPE, shell=True)
    text = pipe.communicate()[0].decode()
    lines = text.split('\n')
    network_interface_list = []
    do = 0
    interface = "unknown"
    for line in lines:
        if "flags=" in line or "Link encap:" in line:
            do = 1
            # if "enp5s0" in line:    # ethernet for laptop
            #     interface = "enp5s0"
            # elif "wlp2s0" in line:  # wi-fi for laptop
            #     interface = "wlp2s0"
            # elif "wlan0" in line:   # wi-fi for samsung
            #     interface = "wlan0"
            # elif "rmnet_data0" in line:  # 4G/5G for samsung
            #     interface = "rmnet_data0"
            if "flags=" in line:
                interface = line[:line.find(':')]
            elif "Link encap:" in line:
                interface = line[:line.find(' ')]
        if do and r"RUNNING" in line:
            network_interface_list.append(interface)
            do = 0
    return sorted(network_interface_list)

def get_ss(device, port, mode, tsync=False):
    global thread_stop
    global n
    global args

    # fp = None
    if tsync:
        fp = open(os.path.join(ss_path, "client_ss_{}_{}_{}_{}_tsync.csv".format(mode.upper(), device, port, n)), 'a+')
    else:
        fp = open(os.path.join(ss_path, "client_ss_{}_{}_{}_{}.csv".format(mode.upper(), device, port, n)), 'a+')
    print(fp)
    while not thread_stop:
        # ss --help (Linux/Android)
        # ss -ai src :PORT (server-side command)
        # ss -ai dst :PORT (client-side command)
        proc = subprocess.Popen(["ss -ai dst :{}".format(port)], stdout=subprocess.PIPE, shell=True)
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
# os.system("echo wmnlab | sudo -S su")

# Get time
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])

_l = []          # command list
ss_threads = []  # ss thread command list

network_interface_list = get_network_interface_list()
print("Available Interfaces:", network_interface_list)
print("----------------------------------------------")
time.sleep(1)

interfaces = devices.copy()
for i, item in enumerate(interfaces):
    if item.startswith('sm') and 'wlan0' in network_interface_list:
        if not args.tsync:
            print("Warning: WiFi is on!!!!!")
            print("If you want to do experiment with 4G/5G, please turn off wifi and continue...")
            # print("Turn off WiFi to continue the experiment.")
            time.sleep(1)
            # sys.exit(1)
        interfaces[i] = 'wlan0'
    elif item.startswith('sm') and 'rmnet_data0' in network_interface_list:
        if args.tsync:
            print("Warning: WiFi is off!!!!!")
            print("If you want to do experiment with wi-fi, please turn on wifi and continue...")
            # print("Turn on WiFi to continue time sync process.")
            time.sleep(1)
            # sys.exit(1)
        interfaces[i] = 'rmnet_data0'
    elif item.startswith('xm') and 'rmnet_data2' in network_interface_list:
        if args.tsync:
            print("Warning: WiFi is off!!!!!")
            print("If you want to do experiment with wi-fi, please turn on wifi and continue...")
            # print("Turn on WiFi to continue time sync process.")
            time.sleep(1)
            # sys.exit(1)
        interfaces[i] = 'rmnet_data2'
    elif item.startswith('qc') and 'enp5s0' in network_interface_list and args.tsync:
        interfaces[i] = 'enp5s0'
    elif item.startswith('qc') and 'wlp2s0' in network_interface_list and args.tsync:
        interfaces[i] = 'wlp2s0'
    elif item.startswith('qc') and 'wlp0s20f3' in network_interface_list and args.tsync:
        interfaces[i] = 'wlp0s20f3'

print("Devices:", devices)
print("Selected Interface:", interfaces)
print("Main Ports:", ports[::2])
print("Auxiliary Ports:", ports[1::2])
print("----------------------------------------------")
if True:
    ports = ports[::2]
else:
    ports = ports[1::2]

for device, port, intf in zip(devices, ports, interfaces):
    # tcpdump process
    if args.tsync:
        pcap = os.path.join(pcap_path, "client_pcap_{}_{}_{}_{}_tsync.pcap".format(args.stream.upper(), device, port, n))
    else:
        pcap = os.path.join(pcap_path, "client_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, port, n))
    _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
    ## iperf process
    if args.tsync:
        log = os.path.join(ilog_path, "client_log_{}_{}_{}_{}_tsync.log".format(args.stream.upper(), device, port, n))
    else:
        log = os.path.join(ilog_path, "client_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, port, n))
    if args.logfile:
        if device == 'unam':
            _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, is_reverse))
        else:
            _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} --bind-dev {} {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, intf, is_reverse))
    else:
        if device == 'unam':
            _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, is_reverse))
        else:
            _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --bind-dev {} {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, intf, is_reverse))
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
            # command = "kill -9 -{}".format(run_item.pid)
            # subprocess.check_output(command.split(" "))
        break
    except Exception as e:
        print("error", e)

thread_stop = True
