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

# ------------------------------ Add Arguments & Global Variables ------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--devices", type=str, nargs='+',  # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-p", "--ports", type=int, nargs='+',    # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-S", "--stream", type=str,
                    help="streaming direction: uplink (ul), downlink (dl), bi-link (bl, 2 ports/device)", default="bl")
parser.add_argument("-L", "--logfile", action="store_true",
                    help="save iperf output to logfile")
parser.add_argument("-K", "--keywords", type=str,
                    help="keywords for socket statistics", default=["bytes_sent", "cwnd"])
parser.add_argument("--tsync", action="store_true",          # needs no value, True if set "--timesync"
                    help="time sync mode")                   # default "experiment" mode
# parser.add_argument("-R", "--reverse", action="store_true",  # needs no value, True if set "-R"
#                     help="downlink or not")                  # default using uplink
# parser.add_argument("--bidir", action="store_true",          # needs no value, True if set "--bidir"
#                     help="bi-link or not")                   # default using uplink
args = parser.parse_args()

device_to_port = {
    "xm00": (3230, 3231),
    "xm01": (3232, 3233),
    "xm02": (3234, 3235),
    "xm03": (3236, 3237),
    "xm04": (3238, 3239),
    "xm05": (3240, 3241),
    "xm06": (3242, 3243),
    "xm07": (3244, 3245),
    "xm08": (3246, 3247),
    "xm09": (3248, 3249),
    "xm10": (3250, 3251),
    "xm11": (3252, 3253),
    "xm12": (3254, 3255),
    "xm13": (3256, 3257),
    "xm14": (3258, 3259),
    "xm15": (3260, 3261),
    "xm16": (3262, 3263),
    "xm17": (3264, 3265),
    "sm00": (3200, 3201),
    "sm01": (3202, 3203),
    "sm02": (3204, 3205),
    "sm03": (3206, 3207),
    "sm04": (3208, 3209),
    "sm05": (3210, 3211),
    "sm06": (3212, 3213),
    "sm07": (3214, 3215),
    "sm08": (3216, 3217),
    "qc00": (3270, 3271),
    "qc01": (3272, 3273),
    "qc02": (3274, 3275),
    "qc03": (3276, 3277),
    "unam": (3280, 3281),
}

# ----------------------------------------- Parameters ------------------------------------------ #
thread_stop = False
# exit_program = False

devices = args.devices
# if args.ports:
#     ports = args.ports
#     if args.stream == "bl" and len(ports) != 2*len(devices):
#         raise Exception("must specify at least and only 2 ports for each device to transmit bi-link.")
#     elif (args.stream == "ul" or args.stream == "dl") and len(ports) != len(devices):
#         raise Exception("must specify at least and only 1 port for each device to transmit uplink or downlink.")
# else:
ports = []
for device in devices:
    ports.append((device_to_port[device][0]))  # default uplink port for each device
    ports.append((device_to_port[device][1]))  # default downlink port for each device

# ----------------------------------------- Save Path ------------------------------------------- #
pcap_path = "./server_pcap"  # packet capture
if not os.path.exists(pcap_path):
    os.mkdir(pcap_path)

log_path = "./server_log"   # iperf log
if not os.path.exists(log_path):
    os.mkdir(log_path)

ss_path = "./server_stats"  # socket statistics (Linux: ss)
if not os.path.exists(ss_path):
    os.mkdir(ss_path)

# ----------------------------------- Define Utils Function ------------------------------------- #
def get_ss(device, port, mode, tsync=False):
    global thread_stop
    global n
    global args

    if tsync:
        fp = open(os.path.join(ss_path, "server_stats_{}_{}_{}_{}_tsync.csv".format(mode.upper(), device, port, n)), 'a+')
    else:
        fp = open(os.path.join(ss_path, "server_stats_{}_{}_{}_{}.csv".format(mode.upper(), device, port, n)), 'a+')
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
if args.stream == "bl":  # bi-link
    print("Uplink   Ports:", ports[::2])
    print("Downlink Ports:", ports[1::2])
    for device, port1, port2 in zip(devices, ports[::2], ports[1::2]):
        # tcpdump
        pcap_ul = os.path.join(pcap_path, "server_pcap_UL_{}_{}_{}.pcap".format(device, port1, n))
        pcap_dl = os.path.join(pcap_path, "server_pcap_DL_{}_{}_{}.pcap".format(device, port2, n))
        _l.append("tcpdump -i any port {} -w {} &".format(port1, pcap_ul))
        _l.append("tcpdump -i any port {} -w {} &".format(port2, pcap_dl))
        # iperf
        log_ul = os.path.join(log_path, "server_log_UL_{}_{}_{}.log".format(device, port1, n))
        log_dl = os.path.join(log_path, "server_log_DL_{}_{}_{}.log".format(device, port2, n))
        if args.logfile:
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile {}".format(port1, log_ul))
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile {}".format(port2, log_dl))
        else:
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port1))
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port2))
        # ss
        ss_threads.append(threading.Thread(target = get_ss, args = (device, port1, 'ul')))
        ss_threads.append(threading.Thread(target = get_ss, args = (device, port2, 'dl')))
elif args.stream == "ul" or args.stream == "dl":  # uplink or downlink
    if args.stream == "ul":
        print("Uplink   Ports:", ports)
    else:
        print("Downlink Ports:", ports)
    for device, port in zip(devices, ports):
        # tcpdump
        pcap = os.path.join(pcap_path, "server_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, port, n))
        _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
        # iperf
        log = os.path.join(log_path, "server_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, port, n))
        if args.logfile:
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile".format(port, log))
        else:
            _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port))
        # ss
        ss_threads.append(threading.Thread(target = get_ss, args = (device, port, args.stream)))
else:
    raise Exception("must specify from {ul, dl, bl}.")

# print("Main Ports:", ports[::2])
# print("Auxiliary Ports:", ports[1::2])
# if True:
#     ports = ports[::2]
# else:
#     ports = ports[1::2]
# for device, port in zip(devices, ports):
#     # if args.stream == 'bl':
#     #     # tcpdump
#     #     pcap = os.path.join(pcap_path, "server_pcap_BL_{}_{}_{}.pcap".format(device, port, n))
#     #     _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
#     #     # iperf
#     #     log = os.path.join(log_path, "server_log_BL_{}_{}_{}.log".format(device, port, n))
#     #     if args.logfile:
#     #         _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile {}".format(port, log))
#     #     else:
#     #         _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port))
#     #     # ss
#     #     ss_threads.append(threading.Thread(target = get_ss, args = (device, port, 'bl')))
#     # else:
#     # tcpdump
#     if args.tsync:
#         pcap = os.path.join(pcap_path, "server_pcap_{}_{}_{}_{}_tsync.pcap".format(args.stream.upper(), device, port, n))
#     else:
#         pcap = os.path.join(pcap_path, "server_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, port, n))
#     _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
#     # iperf
#     if args.tsync:
#         log = os.path.join(log_path, "server_log_{}_{}_{}_{}_tsync.log".format(args.stream.upper(), device, port, n))
#     else:
#         log = os.path.join(log_path, "server_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, port, n))
#     if args.logfile:
#         _l.append("iperf3 -s -B 0.0.0.0 -p {} -V --logfile".format(port, log))
#     else:
#         _l.append("iperf3 -s -B 0.0.0.0 -p {} -V".format(port))
#     # ss
#     ss_threads.append(threading.Thread(target = get_ss, args = (device, port, args.stream, args.tsync)))

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
