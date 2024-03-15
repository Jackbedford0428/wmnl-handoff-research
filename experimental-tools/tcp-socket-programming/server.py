#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    
    Author: Jing-You, Yan

    This script will send duplicated packet to all susflows. UL and DL at the same time.
    You could change the PARAMETERS below.

    Run:
        $ python3 server.py -p Port -d NumOfDevice
    ex:
        $ python3 server.py -p 3270 -d 2

"""

import socket
import time
import threading
import os
import datetime as dt
import argparse
import subprocess
import re
import signal
import numpy as np
from device_to_port import device_to_port, port_to_device


parser = argparse.ArgumentParser()

TCP_CONGESTION = 13

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int,
                    help="port to bind", default=3270)
parser.add_argument("-d", "--num_device", type=int,
                    help="number of devices, number of subflow", default=2)


args = parser.parse_args()

port = args.port
num_devices = args.num_device

UL_ports = np.arange(port, port+2*num_devices, 2)
DL_ports = np.arange(port+1, port+1+2*num_devices, 2)

print("UL_ports", UL_ports)
print("DL_ports", DL_ports)

HOST = '0.0.0.0'

thread_stop = False
exit_program = False


# PARAMETERS ##############################
length_packet = 400
bandwidth = 5000*1024 # units kbps
total_time = 3600
# pcap_path = "/home/wmnlab/D/pcap_data"
# pcap_path = "./pcap_data"
# ss_dir = "./ss"
cong = 'cubic'.encode()
###########################################


expected_packet_per_sec = bandwidth / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec
prev_sleeptime = sleeptime


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
ss_path = "./log/{}/{}".format(date, "server_ss")      # socket statistics (Linux: ss)
makedir(ss_path)


def connection(host, port, result):
    s_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s_tcp.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
    s_tcp.bind((host, port))
    print((host, port), "wait for connection...")
    s_tcp.listen(1)
    conn, tcp_addr = s_tcp.accept()
    print((host, port), "connection setup complete")
    result[0] = s_tcp, conn, tcp_addr

def get_ss(port, type):
    global thread_stop
    now = dt.datetime.today()
    # n = '-'.join([str(x) for x in[ now.year, now.month, now.day, now.hour, now.minute, now.second]])
    n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    n = [x.zfill(2) for x in n]  # zero-padding to two digit
    n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
    f = ""

    if type == 't':
        # f = open(os.path.join(ss_path, f"server_ss_DL_{device}_{port}_{n}.csv"), 'a+')
        f = open(os.path.join(ss_path, f"server_ss_DL_{port}_{n}.csv"), 'a+')
    elif type == 'r':
        # f = open(os.path.join(ss_path, f"server_ss_UL_{device}_{port}_{n}.csv"), 'a+')
        f = open(os.path.join(ss_path, f"server_ss_UL_{port}_{n}.csv"), 'a+')
    print(f)
    while not thread_stop:
        proc = subprocess.Popen(["ss -ai src :%d"%(port)], stdout=subprocess.PIPE, shell=True)

        text = proc.communicate()[0].decode()
        lines = text.split('\n')

        for line in lines:
            if "bytes_sent" in line:
                l = line.strip()
                f.write(",".join([str(dt.datetime.now())]+ re.split("[: \n\t]", l))+'\n')
                break
        time.sleep(1)
    f.close()

def transmission(conn_list):
    print("start transmission")
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
            for j in range(len(conn_list)):
                conn_list[j].sendall(outdata)
            i += 1
            time.sleep(sleeptime)
            if time.time()-start_time > count:
                transmit_bytes = (i-prev_transmit) * length_packet
                if transmit_bytes <= 1024*1024:
                    print("[%d-%d]"%(count-1, count), "send" ,"%g kbps"%(transmit_bytes/1024*8))
                else:
                    print("[%d-%d]"%(count-1, count), "send" ,"%g Mbps"%(transmit_bytes/1024/1024*8))
                count += 1
                sleeptime = (prev_sleeptime / expected_packet_per_sec * (i-prev_transmit) + sleeptime) / 2
                prev_transmit = i
                prev_sleeptime = sleeptime
        except:
            thread_stop = True
            break    
    print("---transmission timeout---")
    print("transmit", i, "packets")


def receive(conn, port):
    conn.settimeout(10)
    print("wait for indata...")
    i = 0
    start_time = time.time()
    count = 1
    seq = 0
    prev_capture = 0
    capture_bytes = 0
    prev_loss = 0
    global thread_stop
    global buffer
    while not thread_stop:
        try:
            indata = conn.recv(65535)
            capture_bytes += len(indata)
            if time.time()-start_time > count:
                if capture_bytes <= 1024*1024:
                    print(port, "[%d-%d]"%(count-1, count), "%g kbps"%(capture_bytes/1024*8))
                else:
                    print(port, "[%d-%d]"%(count-1, count), "%gMbps" %(capture_bytes/1024/1024*8))
                count += 1
                capture_bytes = 0
        except Exception as inst:
            print("Error: ", inst)
            thread_stop = True
    thread_stop = True
    if capture_bytes <= 1024*1024:
        print(port, "[%d-%d]"%(count-1, count), "%g kbps"%(capture_bytes/1024*8))
    else:
        print(port, "[%d-%d]"%(count-1, count), "%gMbps" %(capture_bytes/1024/1024*8))
    print("---Experiment Complete---")
    print("STOP receiving")


while not exit_program:

    now = dt.datetime.today()
    # n = [str(x) for x in[ now.year, now.month, now.day, now.hour, now.minute, now.second]]
    # for i in range(len(n)-3, len(n)):
    #     if len(n[i]) < 2:
    #         n[i] = '0' + n[i]
    # n = '-'.join(n)
    n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    n = [x.zfill(2) for x in n]  # zero-padding to two digit
    n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
    UL_pcapfiles = []
    DL_pcapfiles = []
    tcpdump_UL_proc = []
    tcpdump_DL_proc = []
    get_ss_thread = []


    for p in UL_ports:
        # pcap = os.path.join(pcap_path, f"server_pcap_UL_{device}_{p}_{n}_sock.pcap")
        pcap = os.path.join(pcap_path, f"server_pcap_UL_{p}_{n}_sock.pcap")
        UL_pcapfiles.append(pcap)
    for p in DL_ports:
        # pcap = os.path.join(pcap_path, f"server_pcap_DL_{device}_{p}_{n}_sock.pcap")
        pcap = os.path.join(pcap_path, f"server_pcap_DL_{p}_{n}_sock.pcap")
        DL_pcapfiles.append(pcap)

    for p, pcapfile in zip(UL_ports, UL_pcapfiles):
        tcpdump_UL_proc.append(subprocess.Popen(["tcpdump -i any port %s -w %s &"%(p,  pcapfile)], shell=True, preexec_fn=os.setsid))
        get_ss_thread.append(threading.Thread(target = get_ss, args = (p, 'r')))

    for p, pcapfile in zip(DL_ports, DL_pcapfiles):
        tcpdump_DL_proc.append(subprocess.Popen(["tcpdump -i any port %s -w %s &"%(p,  pcapfile)], shell=True, preexec_fn=os.setsid))
        get_ss_thread.append(threading.Thread(target = get_ss, args = (p, 't')))
    time.sleep(1)
    try:

    
        thread_list = []
        UL_result_list = []
        DL_result_list = []
        for i in range(num_devices):
            UL_result_list.append([None])
            DL_result_list.append([None])
        UL_tcp_list = [None] * num_devices
        UL_conn_list = [None] * num_devices
        DL_tcp_list = [None] * num_devices
        DL_conn_list = [None] * num_devices
        for i in range(len(UL_ports)):
            thread_list.append(threading.Thread(target = connection, args = (HOST, UL_ports[i], UL_result_list[i])))

        for i in range(len(DL_ports)):
            thread_list.append(threading.Thread(target = connection, args = (HOST, DL_ports[i], DL_result_list[i])))

        for i in range(len(thread_list)):
            thread_list[i].start()


        for i in range(len(thread_list)):
            thread_list[i].join()



        for i in range(num_devices):
            UL_tcp_list[i] = UL_result_list[i][0][0]
            UL_conn_list[i] = UL_result_list[i][0][1]
            DL_tcp_list[i] = DL_result_list[i][0][0]
            DL_conn_list[i] = DL_result_list[i][0][1]


    except KeyboardInterrupt:
        print("KeyboardInterrupt -> kill tcpdump")

        for i in range(len(tcpdump_UL_proc)):
            os.killpg(os.getpgid(tcpdump_UL_proc[i].pid), signal.SIGTERM)
        for i in range(len(tcpdump_DL_proc)):
            os.killpg(os.getpgid(tcpdump_DL_proc[i].pid), signal.SIGTERM)


        for pcapfile in UL_pcapfiles:
            subprocess.Popen(["rm %s"%(pcapfile)], shell=True)
        for pcapfile in DL_pcapfiles:
            subprocess.Popen(["rm %s"%(pcapfile)], shell=True)
        exit_program = True
        thread_stop = True
        exit()

    except Exception as inst:
        print("Connection Error:", inst)
        print("KeyboardInterrupt -> kill tcpdump")
        os.system("killall -9 tcpdump")
        for pcapfile in UL_pcapfiles:
            subprocess.Popen(["rm %s"%(pcapfile)], shell=True)
        for pcapfile in DL_pcapfiles:
            subprocess.Popen(["rm %s"%(pcapfile)], shell=True)
        exit()
        continue


    for i in range(num_devices):
        UL_conn_list[i].sendall(b"START")
        DL_conn_list[i].sendall(b"START")
    


    time.sleep(0.5)
    thread_stop = False
    transmision_thread = threading.Thread(target = transmission, args = (DL_conn_list, ))
    recive_thread_list = []
    for i in range(num_devices):
        recive_thread_list.append(threading.Thread(target = receive, args = (UL_conn_list[i], UL_ports[i])))



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


    except KeyboardInterrupt:
        print("finish")
    except Exception as inst:
        print("Error:", inst)
        print("finish")
    finally:
        thread_stop = True
        for i in range(num_devices):
            UL_conn_list[i].close()
            DL_conn_list[i].close()
            UL_tcp_list[i].close()
            DL_tcp_list[i].close()

        for i in range(len(tcpdump_UL_proc)):
            os.killpg(os.getpgid(tcpdump_UL_proc[i].pid), signal.SIGTERM)
        for i in range(len(tcpdump_DL_proc)):
            os.killpg(os.getpgid(tcpdump_DL_proc[i].pid), signal.SIGTERM)
