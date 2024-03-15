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

# ------------------------------ Add Arguments & Global Variables ------------------------------- #
parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--device", type=str,
#                     help="device name (allows only 1 device)", default="unam")
parser.add_argument("-d", "--devices", type=str, nargs='+',  # input list of devices sep by 'space'
                    help="list of devices", default=["unam"])
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")   # Lab249 外網
                    # help="server ip address", default="192.168.1.251")  # Lab249 內網
                    # help="server ip address", default="140.112.17.209") # Lab355 外網
                    # help="server ip address", default="210.65.88.213")  # CHT 外網
parser.add_argument("-p", "--ports", type=int, nargs='+',     # input list of port numbers sep by 'space'
                    help="ports to bind")
parser.add_argument("-u", "--udp", action="store_true",       # needs no value, True if set "-u"
                    help="use UDP rather than TCP")           # default TCP
parser.add_argument("-b", "--bitrate", type=str,
                    # help="target bitrate in bits/sec (0 for unlimited)", default=["1M", "1M"])  # [UDP, TCP]
                    help="target bitrate in bits/sec (0 for unlimited)", default='1M')
parser.add_argument("-l", "--length", type=str,
                    # help="length of buffer to read or write in bytes (packet size)", default=["250", "250"])  # [UDP, TCP]
                    help="length of buffer to read or write in bytes (packet size)", default='250')
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
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

if args.tsync:
    if args.host == "140.112.20.183":
        args.host = "192.168.1.251"
    args.udp = True
    args.time = 60

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

serverip = args.host
devices = args.devices
# if args.ports:
#     ports = args.ports
#     if args.stream == "bl" and len(ports) != 2:
#         raise Exception("must specify at least and only 2 ports for the device to transmit bi-link.")
#     elif (args.stream == "ul" or args.stream == "dl") and len(ports) != 1:
#         raise Exception("must specify at least and only 1 port for the device to transmit uplink or downlink.")
# else:
#     ports = [device_to_port[device][0], device_to_port[device][1]]  # default port setting for each device
ports = []
for device in devices:
    ports.append((device_to_port[device][0]))  # default uplink port for each device
    ports.append((device_to_port[device][1]))  # default downlink port for each device

is_udp = "-u" if args.udp else ""

# if type(args.bitrate) is list:  # unit: bps
#     bitrate = args.bitrate[0] if args.udp else args.bitrate[1]
# else:
#     bitrate = args.bitrate
# if type(args.length) is list:  # unit: bytes
#     packet_size = args.length[0] if args.udp else args.length[1]
# else:
#     packet_size = args.length

bitrate = args.bitrate     # unit: bps
packet_size = args.length  # unit: bytes

if args.stream == "ul":
    is_reverse = ""
elif args.stream == "dl":
    is_reverse = "-R"

# check whether it has iperf3m
if os.path.exists("/bin/iperf3m") or os.path.exists("/sbin/iperf3m"):
    iperf = "iperf3m"
else:
    iperf = "iperf3"

# ----------------------------------------- Save Path ------------------------------------------- #
pcap_path = "./client_pcap"  # packet capture
if not os.path.exists(pcap_path):
    os.mkdir(pcap_path)

log_path = "./client_log"   # iperf log
if not os.path.exists(log_path):
    os.mkdir(log_path)

ss_path = "./client_stats"  # socket statistics (Linux: ss)
if not os.path.exists(ss_path):
    os.mkdir(ss_path)

# ----------------------------------- Define Utils Function ------------------------------------- #
def get_network_interface_list():
    pipe = subprocess.Popen('ifconfig', stdout=subprocess.PIPE, shell=True)
    text = pipe.communicate()[0].decode()
    lines = text.split('\n')
    network_interface_list = []
    flag = 0
    for line in lines:
        if not flag and (r"RUNNING" in line and 'lo' not in line and 'Metric' not in line) or 'wlan0' in line or 'rmnet_data0' in line:
            interface = line[:line.find(':')]
            flag = 1
            if 'wlan0' in line:
                interface = 'wlan0'
            elif 'rmnet_data0' in line:
                interface = 'rmnet_data0'
        elif flag:
            if interface == 'wlan0':
                if 'inet' not in line:
                    flag = 0
                    continue
                ip = line[line.find('inet')+5:line.find('Bcast')-2]
            elif interface == 'rmnet_data0':
                ip = line[line.find('inet')+5:line.find('Mask')-2]
            else:
                ip = line[line.find('inet')+5:line.find('netmask')-2]
            network_interface_list.append((interface, ip))
            flag = 0
    return sorted(network_interface_list)

def get_ss(device, port, mode, tsync=False):
    global thread_stop
    global n
    global args

    # fp = None
    if tsync:
        fp = open(os.path.join(ss_path, "client_stats_{}_{}_{}_{}_tsync.csv".format(mode.upper(), device, port, n)), 'a+')
    else:
        fp = open(os.path.join(ss_path, "client_stats_{}_{}_{}_{}.csv".format(mode.upper(), device, port, n)), 'a+')
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
# Get time
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])

_l = []          # command list
ss_threads = []  # ss thread command list
if args.stream == "bl":  # bi-link
    # tcpdump
    pcap_bl = os.path.join(pcap_path, "client_pcap_BL_{}_{}_{}_{}.pcap".format(device, ports[0], ports[1], n))
    tcpproc = "tcpdump -i any net {} -w {} &".format(serverip, pcap_bl)
    # iperf
    log1 = os.path.join(log_path, "client_log_UL_{}_{}_{}.log".format(device, ports[0], n))
    log2 = os.path.join(log_path, "client_log_UL_{}_{}_{}.log".format(device, ports[1], n))
    if args.logfile:
        socket_proc1 = "{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {}".format(iperf, serverip, ports[0], bitrate, packet_size, is_udp, args.time, log1)
        socket_proc2 = "{} -c {} -p {} -b {} -l {} {} -R -t {} -V --logfile {}".format(iperf, serverip, ports[1], bitrate, packet_size, is_udp, args.time, log2)
    else:
        socket_proc1 = "{} -c {} -p {} -b {} -l {} {} -t {} -V".format(iperf, serverip, ports[0], bitrate, packet_size, is_udp, args.time)
        socket_proc2 = "{} -c {} -p {} -b {} -l {} {} -R -t {} -V".format(iperf, serverip, ports[1], bitrate, packet_size, is_udp, args.time)
    _l = [tcpproc, socket_proc1, socket_proc2]
    # ss
    ss_threads.append(threading.Thread(target = get_ss, args = (device, ports[0], 'ul')))
    ss_threads.append(threading.Thread(target = get_ss, args = (device, ports[1], 'dl')))
elif args.stream == "ul" or args.stream == "dl":  # uplink or downlink
    # tcpdump
    pcap = os.path.join(pcap_path, "client_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, ports[0], n))
    tcpproc = "tcpdump -i any net {} -w {} &".format(serverip, pcap)
    # iperf
    log = os.path.join(log_path, "client_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, ports[0], n))
    if args.logfile:
        socket_proc = "{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {}".format(iperf, serverip, ports[0], bitrate, packet_size, is_udp, args.time, log)
    else:
        socket_proc = "{} -c {} -p {} -b {} -l {} {} -t {} -V".format(iperf, serverip, ports[0], bitrate, packet_size, is_udp, args.time)
    _l = [tcpproc, socket_proc]
    # ss
    ss_threads.append(threading.Thread(target = get_ss, args = (device, ports[0], args.stream)))
else:
    raise Exception("must specify from {ul, dl, bl}.")

# # interface_to_ip = {item[0] : item[1] for item in get_network_interface_list() if item[0].startswith(('qc', 'sm', 'xm', 'wlan0', 'rmnet_data0', 'wlp2s0', 'enp5s0'))}
# interface_to_ip = {item[0] : item[1] for item in get_network_interface_list()}
# # samsung: 4G/5G - rmnet_data0; wi-fi - wlan0; wi-fi > 4G/5G in priority
# # laptop (Ubuntu): wired - enp5s0; wi-fi - wlp2s0

# # interfaces = devices[:]
# interfaces = devices.copy()
# for i, item in enumerate(interfaces):
#     if item.startswith('sm') and 'wlan0' in interface_to_ip.keys():
#         if not args.tsync:
#             print("Warning: Wi-Fi is on!!!!!")
#             print("Halting the process.")
#             print("Turn off Wi-Fi to continue the experiment.")
#             sys.exit(1)
#         interfaces[i] = 'wlan0'
#     elif item.startswith('sm') and 'rmnet_data0' in interface_to_ip.keys():
#         if args.tsync:
#             print("Warning: Wi-Fi is off!!!!!")
#             print("Halting the process.")
#             print("Turn on Wi-Fi to continue time sync process.")
#             sys.exit(1)
#         interfaces[i] = 'rmnet_data0'
#     elif item.startswith('qc') and 'enp5s0' in interface_to_ip.keys() and args.tsync:
#         interfaces[i] = 'enp5s0'
#     elif item.startswith('qc') and 'wlp2s0' in interface_to_ip.keys() and args.tsync:
#         interfaces[i] = 'wlp2s0'
# print(interface_to_ip)
# # if args.tsync:
# #     for i, item in enumerate(interfaces):
# #         if item.startswith('sm'):
# #             if 'wlan0' in interface_to_ip.keys():
# #                 interfaces[i] = 'wlan0'
# #             else:
# #                 interfaces[i] = 'rmnet_data0'
# #         elif item.startswith('qc'):
# #             if 'enp5s0' in interface_to_ip.keys():
# #                 interfaces[i] = 'enp5s0'
# #             else:
# #                 interfaces[i] = 'wlp2s0'
# print("Selected Interface:", interfaces)

# print("Main Ports:", ports[::2])
# print("Auxiliary Ports:", ports[1::2])
# if True:
#     ports = ports[::2]
# else:
#     ports = ports[1::2]
# for device, port, intf in zip(devices, ports, interfaces):
#     bind_ip = '0.0.0.0'
#     if device.startswith(('qc', 'sm')):
#         # bind_ip = interface_to_ip[device]
#         bind_ip = interface_to_ip[intf]
#     # if args.bidir:
#     #     # tcpdump
#     #     pcap = os.path.join(pcap_path, "client_pcap_BL_{}_{}_{}.pcap".format(device, port, n))
#     #     _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
#     #     # iperf
#     #     log = os.path.join(log_path, "client_log_BL_{}_{}_{}.log".format(device, port, n))
#     #     if args.logfile:
#     #         _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} --bidir -B {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, bind_ip))
#     #     else:
#     #         _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --bidir -B {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, bind_ip))
#     #     # ss
#     #     ss_threads.append(threading.Thread(target = get_ss, args = (device, port, 'bl')))
#     # else:
#     # tcpdump
#     if args.tsync:
#         pcap = os.path.join(pcap_path, "client_pcap_{}_{}_{}_{}_tsync.pcap".format(args.stream.upper(), device, port, n))
#     else:
#         pcap = os.path.join(pcap_path, "client_pcap_{}_{}_{}_{}.pcap".format(args.stream.upper(), device, port, n))
#     _l.append("tcpdump -i any port {} -w {} &".format(port, pcap))
#     # iperf
#     if args.tsync:
#         log = os.path.join(log_path, "client_log_{}_{}_{}_{}_tsync.log".format(args.stream.upper(), device, port, n))
#     else:
#         log = os.path.join(log_path, "client_log_{}_{}_{}_{}.log".format(args.stream.upper(), device, port, n))
#     if args.logfile:
#         if args.stream == 'bl':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} --bidir".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} -B {} --bind-dev {} --bidir".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, bind_ip, intf))
#         elif args.stream == 'dl':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} -R".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} -B {} --bind-dev {} -R".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, bind_ip, intf))
#         elif args.stream == 'ul':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --logfile {} -B {} --bind-dev {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, log, bind_ip, intf))
#     else:
#         if args.stream == 'bl':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V --bidir".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V -B {} --bind-dev {} --bidir".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, bind_ip, intf))
#         elif args.stream == 'dl':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V -R".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V -B {} --bind-dev {} -R".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, bind_ip, intf))
#         elif args.stream == 'ul':
#             if device == 'unam':
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time))
#             else:
#                 _l.append("{} -c {} -p {} -b {} -l {} {} -t {} -V -B {} --bind-dev {}".format(iperf, serverip, port, bitrate, packet_size, is_udp, args.time, bind_ip, intf))
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
            # command = "kill -9 -{}".format(run_item.pid)
            # subprocess.check_output(command.split(" "))
        break
    except Exception as e:
        print("error", e)

thread_stop = True
