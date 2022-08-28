import socket
import time
import threading
import os
import datetime as dt
import argparse
import subprocess
import re
import signal


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int,
                    help="port to bind", default=3271)
parser.add_argument("-H", "--HOST", type=str,
                    help="server ip address", default="140.112.20.183")
                    # help="server ip address", default="210.65.88.213")
args = parser.parse_args()

IP_MTU_DISCOVER   = 10
IP_PMTUDISC_DONT  =  0  # Never send DF frames.
IP_PMTUDISC_WANT  =  1  # Use per route hints.
IP_PMTUDISC_DO    =  2  # Always DF.
IP_PMTUDISC_PROBE =  3  # Ignore dst pmtu.
TCP_CONGESTION = 13

PORT = int(args.port)           # UL
serverip = args.HOST
thread_stop = False
exit_program = False
length_packet = 250 # unit: byte
bandwidth = 200 # unit: kbps
total_time = 3600 # unit: second
pcap_path = os.path.join(".", "client_pcap")
# try: 
#     os.mkdir(pcap_path)
# except: 
#     pass
if not os.path.exists(pcap_path):
    os.mkdir(pcap_path)

# ss_dir = "ss"
# hostname = str(PORT) + ":"

# cong = 'reno'.encode()

# tcpproc1 =  subprocess.Popen(["tcpdump -i any port %s -w %s &"%(PORT,  pcapfile1)], shell=True, preexec_fn=os.setsid)
now = dt.datetime.today()

n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
for i in range(len(n)-3, len(n)):
    if len(n[i]) < 2:
        n[i] = '0' + n[i]
n = '-'.join(n)

# pcap_ul = "%s/CLIENT_UL_%d_%s.pcap"%(pcap_path, PORT, n)
# pcap_dl = "%s/CLIENT_DL_%d_%s.pcap"%(pcap_path, PORT+1, n)
pcap_bl = "%s/CLIENT_BL_%d_%d_%s.pcap"%(pcap_path, PORT, PORT+1, n)
socket_proc1 =  subprocess.Popen(["iperf3 -c %s -p %d -b %dk -t 3600"%(serverip, PORT, bandwidth)], shell=True, preexec_fn=os.setsid)
socket_proc2 =  subprocess.Popen(["iperf3 -c %s -p %d -b %dk -R -t 3600"%(serverip, PORT+1, bandwidth)], shell=True, preexec_fn=os.setsid)
tcpproc =  subprocess.Popen(["tcpdump -i any net %s -w %s &"%(serverip, pcap_bl)], shell=True, preexec_fn=os.setsid)


while True:

    try:
        time.sleep(1)
    except KeyboardInterrupt:
        subprocess.Popen(["killall -9 iperf3"], shell=True, preexec_fn=os.setsid)
        os.killpg(os.getpgid(socket_proc1.pid), signal.SIGTERM)
        os.killpg(os.getpgid(socket_proc2.pid), signal.SIGTERM)
        os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
        break
    except Exception as e:
        print("error", e)

os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)