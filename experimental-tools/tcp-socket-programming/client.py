#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
    Author: Jing-You, Yan

    This script will send duplicated packet to all avaiable subflows.
    You could change the PARAMETERS below.

    Run:
        $ python3 client.py -p Port -H server_ip_address
    ex:
        $ python3 client.py -p 3270 -H 140.112.20.183

"""


import socket
import time
import threading
import datetime as dt
import select
import sys
import os
import queue
import argparse
import subprocess
import re
import numpy as np
import signal
from device_to_port import device_to_port, port_to_device


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int,
                    help="port to bind", default=3270)
parser.add_argument("-H", "--HOST", type=str,
                    help="server ip address", default="140.112.20.183")
                    # help="server ip address", default="210.65.88.213")
args = parser.parse_args()

HOST = args.HOST
port = args.port


def get_network_interface_list():
    pipe = subprocess.Popen('ifconfig', stdout=subprocess.PIPE, shell=True)
    text = pipe.communicate()[0].decode()
    l = text.split('\n')
    network_interface_list = []
    for x in l:
        if r"RUNNING" in x and 'lo' not in x:
            network_interface_list.append(x[:x.find(':')])
    network_interface_list = sorted(network_interface_list)
    return network_interface_list
network_interface_list = get_network_interface_list()
print(network_interface_list)

num_ports = len(network_interface_list)
UL_ports = np.arange(port, port+2*num_ports, 2)
DL_ports = np.arange(port+1, port+1+2*num_ports, 2)


thread_stop = False
exit_program = False
length_packet = 400
bandwidth = 5000*1024 # units kbps
total_time = 3600
expected_packet_per_sec = bandwidth / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec
prev_sleeptime = sleeptime
# pcap_path = "pcapdir"
exitprogram = False
TCP_CONGESTION = 13   # defined in /usr/include/netinet/tcp.h
cong = 'cubic'.encode()
# ss_dir = "ss"

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
ss_path = "./log/{}/{}".format(date, "client_ss")      # socket statistics (Linux: ss)
makedir(ss_path)

def get_ss(port, type):
    now = dt.datetime.today()
    # n = '-'.join([str(x) for x in[ now.year, now.month, now.day, now.hour, now.minute, now.second]])
    n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    n = [x.zfill(2) for x in n]  # zero-padding to two digit
    n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
    f = ""
    if type == 't':
        # f = open(os.path.join(ss_path, f"client_ss_UL_{device}_{port}_{n}.csv"), 'a+')
        f = open(os.path.join(ss_path, f"client_ss_UL_{port}_{n}.csv"), 'a+')
    elif type == 'r':
        # f = open(os.path.join(ss_path, f"client_ss_DL_{device}_{port}_{n}.csv"), 'a+')
        f = open(os.path.join(ss_path, f"client_ss_DL_{port}_{n}.csv"), 'a+')
    print(f)
    global thread_stop

    while not thread_stop:
        proc = subprocess.Popen(["ss -ai dst :%d"%(port)], stdout=subprocess.PIPE, shell=True)

        text = proc.communicate()[0].decode()
        lines = text.split('\n')

        for line in lines:
            if "cwnd" in line:
                l = line.strip()
                f.write(",".join([str(dt.datetime.now())]+ re.split("[: \n\t]", l))+'\n')
                break
        time.sleep(1)
    f.close()

def connection_setup(host, port, interface, result):
    s_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_tcp.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
    s_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, ((interface)+'\0').encode())
    s_tcp.settimeout(10)
    s_tcp.connect((host, port))

    while True:
        print("%d wait for starting..."%(port))
        try:
            indata = s_tcp.recv(65535)
            if indata == b'START':
                print("START")
                break
            else:
                print("WTF", indata)
                break
        except Exception as inst:
            print("Error: ", inst)

    result[0] = s_tcp

def transmision(stcp_list):
    print("start transmision", stcp_list)
    i = 0
    prev_transmit = 0
    ok = (1).to_bytes(1, 'big')
    start_time = time.time()
    count = 1
    sleeptime = 1.0 / expected_packet_per_sec
    prev_sleeptime = sleeptime
    global thread_stop
    while time.time() - start_time < total_time and not thread_stop:
        try:
            t = time.time()
            t = int(t*1000).to_bytes(8, 'big')
            z = i.to_bytes(4, 'big')
            redundent = os.urandom(length_packet-12-1)
            outdata = t + z + ok +redundent
            for j in range(len(stcp_list)):
                stcp_list[j].sendall(outdata)
            i += 1
            time.sleep(sleeptime)
            if time.time()-start_time > count:
                transmit_bytes = (i-prev_transmit) * length_packet
                if transmit_bytes <= 1024*1024:
                    print("[%d-%d]"%(count-1, count), "%g kbps"%(transmit_bytes/1024*8))
                else:
                    print("[%d-%d]"%(count-1, count), "%g Mbps"%(transmit_bytes/1024/1024*8))
                count += 1
                sleeptime = (prev_sleeptime / expected_packet_per_sec * (i-prev_transmit) + sleeptime) / 2
                prev_transmit = i
                prev_sleeptime = sleeptime
        except:
            break
    thread_stop = True
    print("---transmision timeout---")
    print("transmit", i, "packets")


def receive(s_tcp, port):
    s_tcp.settimeout(10)
    print("wait for indata...")
    start_time = time.time()
    count = 1
    capture_bytes = 0
    global thread_stop
    global buffer
    buffer = queue.Queue()
    while not thread_stop:
        try:
            indata = s_tcp.recv(65535)
            capture_bytes += len(indata)
            if time.time()-start_time > count:
                if capture_bytes <= 1024*1024:
                    print(port, "[%d-%d]"%(count-1, count), "%g kbps"%(capture_bytes/1024*8))
                else:
                    print(port, "[%d-%d]"%(count-1, count), "%g Mbps" %(capture_bytes/1024/1024*8))
                count += 1
                capture_bytes = 0
        except Exception as inst:
            print("Error: ", inst)
            thread_stop = True
    thread_stop = True
    if capture_bytes <= 1024*1024:
        print(port, "[%d-%d]"%(count-1, count), "%g kbps"%(capture_bytes/1024*8))
    else:
        print(port, "[%d-%d]"%(count-1, count), "%g Mbps" %(capture_bytes/1024/1024*8))
    print("---Experiment Complete---")
    print("STOP receiving")


while not exitprogram:
    get_network_interface_list

    try:
        x = input("Press Enter to start\n")
        if x == "EXIT":
            break
        now = dt.datetime.today()

        # n = [str(x) for x in[ now.year, now.month, now.day, now.hour, now.minute, now.second]]
        # for i in range(len(n)-3, len(n)):
        #     if len(n[i]) < 2:
        #         n[i] = '0' + n[i]
        # n = '-'.join(n)
        n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
        n = [x.zfill(2) for x in n]  # zero-padding to two digit
        n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
        get_ss_thread = []
        UL_pcapfiles = []
        DL_pcapfiles = []
        tcpdump_UL_proc = []
        tcpdump_DL_proc = []
        for p in UL_ports:
            # pcap = os.path.join(pcap_path, f"client_pcap_UL_{device}_{p}_{n}_sock.pcap")
            pcap = os.path.join(pcap_path, f"client_pcap_UL_{p}_{n}_sock.pcap")
            UL_pcapfiles.append(pcap)
        for p in DL_ports:
            # pcap = os.path.join(pcap_path, f"client_pcap_DL_{device}_{p}_{n}_sock.pcap")
            pcap = os.path.join(pcap_path, f"client_pcap_DL_{p}_{n}_sock.pcap")
            DL_pcapfiles.append(pcap)

        for p, pcapfile in zip(UL_ports, UL_pcapfiles):
            tcpdump_UL_proc.append(subprocess.Popen(["tcpdump -i any port %s -w %s &"%(p,pcapfile)], shell=True, preexec_fn=os.setsid))
            get_ss_thread.append(threading.Thread(target = get_ss, args = (p, 't')))

        for p, pcapfile in zip(DL_ports, DL_pcapfiles):
            tcpdump_DL_proc.append(subprocess.Popen(["tcpdump -i any port %s -w %s &"%(p,pcapfile)], shell=True, preexec_fn=os.setsid))
            get_ss_thread.append(threading.Thread(target = get_ss, args = (p, 'r')))


        thread_list = []
        UL_result_list = []
        DL_result_list = []

        UL_tcp_list = [None] * num_ports
        DL_tcp_list = [None] * num_ports

        for i in range(num_ports):
            UL_result_list.append([None])
            DL_result_list.append([None])

        for i in range(len(UL_ports)):
            thread_list.append(threading.Thread(target = connection_setup, args = (HOST, UL_ports[i], network_interface_list[i], UL_result_list[i])))

        for i in range(len(DL_ports)):
            thread_list.append(threading.Thread(target = connection_setup, args = (HOST, DL_ports[i], network_interface_list[i], DL_result_list[i])))
        
        for i in range(len(thread_list)):
            thread_list[i].start()

        for i in range(len(thread_list)):
            thread_list[i].join()

        for i in range(num_ports):
            UL_tcp_list[i] = UL_result_list[i][0]
            DL_tcp_list[i] = DL_result_list[i][0]

        print("UL_tcp_list", UL_tcp_list)
        print("DL_tcp_list", DL_tcp_list)

        for i in range(num_ports):
            assert(UL_tcp_list[i] != None)
            assert(DL_tcp_list[i] != None)

    except Exception as inst:
        print("Error: ", inst)

        for i in range(len(tcpdump_UL_proc)):
            os.killpg(os.getpgid(tcpdump_UL_proc[i].pid), signal.SIGTERM)
        for i in range(len(tcpdump_DL_proc)):
            os.killpg(os.getpgid(tcpdump_DL_proc[i].pid), signal.SIGTERM)


        continue
    thread_stop = False

    thread_stop = False
    transmision_thread = threading.Thread(target = transmision, args = (UL_tcp_list, ))
    recive_thread_list = []
    for i in range(num_ports):
        recive_thread_list.append(threading.Thread(target = receive, args = (DL_tcp_list[i], DL_ports[i])))


    try:
        transmision_thread.start()
        for i in range(len(recive_thread_list)):
            recive_thread_list[i].start()

        for i in range(len(get_ss_thread)):
            get_ss_thread[i].start()

        transmision_thread.join()


        for i in range(len(recive_thread_list)):
            recive_thread_list[i].join()

        for i in range(len(get_ss_thread)):
            get_ss_thread[i].join()


        while transmision_thread.is_alive():
            x = input("Enter STOP to Stop\n")
            if x == "STOP":
                thread_stop = True
                break
            elif x == "EXIT":
                thread_stop = True
                exitprogram = True
        thread_stop = True


    except Exception as inst:
        print("Error: ", inst)
    except KeyboardInterrupt:
        print("finish")

    finally:
        thread_stop = True
        for i in range(num_ports):
            UL_tcp_list[i].close()
            DL_tcp_list[i].close()

        for i in range(len(tcpdump_UL_proc)):
            os.killpg(os.getpgid(tcpdump_UL_proc[i].pid), signal.SIGTERM)
        for i in range(len(tcpdump_DL_proc)):
            os.killpg(os.getpgid(tcpdump_DL_proc[i].pid), signal.SIGTERM)
