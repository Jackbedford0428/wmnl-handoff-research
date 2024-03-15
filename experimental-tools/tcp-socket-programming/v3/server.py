#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import time
import threading
import multiprocessing
import os
import sys
import datetime as dt
import argparse
import subprocess
import signal
from device_to_port import device_to_port
from socket import error as SocketError
import errno


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--number_client", type=int,
                        help="number of client", default=1)
    parser.add_argument("-d", "--devices", type=str, nargs='+',
                        help="list of devices", default=["unam"])
    parser.add_argument("-p", "--ports", type=str, nargs='+',
                        help="ports to bind")
    parser.add_argument("-b", "--bitrate", type=str,
                        help="target bitrate in bits/sec (0 for unlimited)", default="1M")
    parser.add_argument("-l", "--length", type=str,
                        help="length of buffer to read or write in bytes (packet size)", default="250")
    parser.add_argument("-t", "--time", type=int,
                        help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
    return parser.parse_args()

def create_device_list(devices):
    device_list = []
    for dev in devices:
        if '-' in dev:
            pmodel = dev[:2]
            start = int(dev[2:4])
            stop = int(dev[5:7]) + 1
            for i in range(start, stop):
                _dev = "{}{:02d}".format(pmodel, i)
                device_list.append(_dev)
            continue
        device_list.append(dev)
    return device_list

def create_port_list(args, devices):
    if not args.ports:
        return [(device_to_port[device][0], device_to_port[device][1]) for device in devices]
    else:
        port_list = []
        for port in args.ports:
            if '-' in port:
                start = int(port[:port.find('-')])
                stop = int(port[port.find('-') + 1:]) + 1
                port_list.extend(list(range(start, stop)))
            else:
                port_list.append(int(port))
        return port_list

# ===================== Main Process =====================

args = parse_arguments()

devices = create_device_list(args.devices)
ports = create_port_list(args, devices)

length_packet = int(args.length)

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)

total_time = args.time
number_client = args.number_client

expected_packet_per_sec = bitrate / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec

print(devices)
print(ports)
print("bitrate:", f'{args.bitrate}bps')

# ===================== Simple Socket =====================

# # 設定伺服器的主機和埠
# HOST = '127.0.0.1'
# PORT = 12345

# # 建立TCP伺服器
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_socket.bind((HOST, PORT))
# server_socket.listen(1)

# print('等待客戶端連線...')
# client_socket, client_address = server_socket.accept()
# print('已連線至:', client_address)

# while True:
#     data = client_socket.recv(1024)
#     if not data:
#         break
#     print('收到來自客戶端的訊息:', data.decode())
#     response = input('請輸入要回傳的訊息: ')
#     client_socket.sendall(response.encode())

# client_socket.close()
# server_socket.close()

# ===================== Parameters =====================
HOST = '0.0.0.0'
pcap_path = '/home/wmnlab/temp'

# ===================== Global Variables =====================
stop_threads = False

# ===================== setup socket =====================

os.system("echo wmnlab | sudo -S su")

rx_sockets = []
tx_sockets = []

# Setup connections
for dev, port in zip(devices, ports):
    s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # server_socket
    # s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s1.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
    s1.bind((HOST, port[0]))
    s1.listen(1)  # Listen for incoming connections
    rx_sockets.append(s1)
    
    s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # server_socket
    # s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # s2.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
    s2.bind((HOST, port[1]))
    s2.listen(1)  # Listen for incoming connections
    tx_sockets.append(s2)

    print(f'Create socket at {HOST}:{port[0]} for UL...')
    print(f'Create socket at {HOST}:{port[1]} for DL...')

print('Waiting for connections...')

tcp_addr = {}
rx_connections = []
tx_connections = []

def accept_connection(s1, s2, device):
    conn1, addr1 = s1.accept()  # client_socket 1, client_address 1
    print('Connection established with:', device, addr1)
    tcp_addr[s1] = addr1
    rx_connections.append(conn1)
    
    conn2, addr2 = s2.accept()  # client_socket 2, client_address 2
    print('Connection established with:', device, addr2)
    tcp_addr[s2] = addr2
    tx_connections.append(conn2)
    
# Accept incoming connections
act_threads = []
for s1, s2, dev in zip(rx_sockets, tx_sockets, devices):
    accept_connection(s1, s2, dev)

print("Successfully establish all connections!")

for conn, dev in zip(rx_connections, devices):
    while True:
        try:
            indata = conn.recv(65535)
            if indata.decode()[:5] == 'START':
                print(f"{dev} START")
                break
            else:
                print("WTF?????", indata)
                break
            
        except Exception as inst:
            print("Error:", inst)
            print("Sockets closed")
            for s1, s2 in zip(rx_connections, tx_connections):
                s1.close()
                s2.close()
            for s1, s2 in zip(rx_sockets, tx_sockets):
                s1.close()
                s2.close()
            sys.exit()

time.sleep(1)

# ===================== traffic capture =====================

tcpproc_list = []

def capture_traffic(devices, ports, pcap_path, current_datetime):
    for device, port in zip(devices, ports):
        pcap = os.path.join(pcap_path, f"server_pcap_BL_{device}_{port[0]}_{port[1]}_{current_datetime}_sock.pcap")
        tcpproc = subprocess.Popen([f"tcpdump -i any port '({port[0]} or {port[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
        # tcpproc = subprocess.Popen([f"sudo tcpdump -i any port '({port[0]} or {port[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
        tcpproc_list.append(tcpproc)
    time.sleep(1)

def kill_traffic_capture():
    print('Killing tcpdump process...')
    for tcpproc in tcpproc_list:
        os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
        # os.system(f"sudo kill -15 {tcpproc.pid}")
    time.sleep(1)

now = dt.datetime.today()
current_datetime = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
current_datetime = [x.zfill(2) for x in current_datetime]  # zero-padding to two digit
current_datetime = '-'.join(current_datetime[:3]) + '_' + '-'.join(current_datetime[3:])

capture_traffic(devices, ports, pcap_path, current_datetime)

# ===================== transmit & receive =====================

def receive(conn, dev, port):
    global stop_threads
    print(f"Waiting for indata from {dev} at 0.0.0.0:{port}...")
    
    time_slot = 1
    capture_bytes = 0
    while not stop_threads:
        try:
            
            indata = conn.recv(65535)  # Receive data from connection

            try:
                rx_start_time
            except NameError:
                rx_start_time = time.time()

            capture_bytes += len(indata)
            
            # Show information
            if time.time() - rx_start_time > time_slot:
                if capture_bytes <= 1000*1000/8:
                    print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", "%g kbps"%(capture_bytes/1000*8))
                else:
                    print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", "%g Mbps"%(capture_bytes/1000/1000*8))
                time_slot += 1
                capture_bytes = 0
            
            # try:
            #     if indata.decode()[:4] == 'STOP':
            #         stop_threads = True
            #         print(f"{dev} STOP")
            #         break
            # except:
            #     pass

        except SocketError as inst:
            print("SocketError:", inst)
            if inst.errno != errno.ECONNRESET:
                # The error is NOT a ConnectionResetError
                stop_threads = True
    
        except ValueError as inst:
            print("ValueError:", inst)
            stop_threads = True
        
        except KeyboardInterrupt as inst:
            print("KeyboardInterrupt:", inst)
            stop_threads = True
            
        except Exception as inst:
            print("Error:", inst)
            stop_threads = True

def transmit(connections):
    global stop_threads
    print("Start transmission:")
    
    seq = 1
    prev_transmit = 0
    
    start_time = time.time()
    next_transmit_time = start_time + sleeptime
    
    time_slot = 1
    
    while time.time() - start_time < total_time and not stop_threads:
        try:
            t = time.time()
            while t < next_transmit_time:
                t = time.time()
            next_transmit_time = next_transmit_time + sleeptime
            
            euler = 271828
            pi = 31415926
            datetimedec = int(t)
            microsec = int((t - int(t))*1000000)
            
            redundant = os.urandom(length_packet-4*5)
            outdata = euler.to_bytes(4, 'big') + pi.to_bytes(4, 'big') + datetimedec.to_bytes(4, 'big') + microsec.to_bytes(4, 'big') + seq.to_bytes(4, 'big') + redundant
            
            for conn in connections:
                conn.sendall(outdata)  # Send data over the connection
            seq += 1
            
            if time.time() - start_time > time_slot:
                print("[%d-%d]" % (time_slot-1, time_slot), "transmit", seq-prev_transmit)
                time_slot += 1
                prev_transmit = seq
        
        except SocketError as inst:
            print("SocketError:", inst)
            if inst.errno != errno.ECONNRESET:
                # The error is NOT a ConnectionResetError
                stop_threads = True
                
        except ValueError as inst:
            print("ValueError:", inst)
            stop_threads = True
        
        except KeyboardInterrupt as inst:
            print("KeyboardInterrupt:", inst)
            stop_threads = True
            
        except Exception as inst:
            print("Error:", inst)
            stop_threads = True
            
    stop_threads = True
    print("---transmission timeout---")
    print("transmit", seq, "packets")

# Create and start Uplink receiving multi-threading
rx_threads = []
for conn, dev, port in zip(rx_connections, devices, ports):
    t_rx = threading.Thread(target = receive, args=(conn, dev, port[0],), daemon=True)
    rx_threads.append(t_rx)
    t_rx.start()

# Create adn start Downlink transmission multi-processing
# p_tx = multiprocessing.Process(target=transmit, args=(tx_connections,), daemon=True)
# p_tx.start()
t_tx = threading.Thread(target=transmit, args=(tx_connections,), daemon=True)
t_tx.start()

# ===================== wait for experiment end =====================

def cleanup_and_exit():
    # Kill transmit process
    # p_tx.terminate()
    # time.sleep(1)

    # Close sockets
    for s1, s2 in zip(rx_connections, tx_connections):
        s1.close()
        s2.close()

    for s1, s2 in zip(rx_sockets, tx_sockets):
        s1.close()
        s2.close()

    # Kill tcpdump process
    kill_traffic_capture()

    print('Successfully closed.')
    # sys.exit()

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True
        time.sleep(1)

# End without KeyboardInterrupt (Ctrl-C, Ctrl-Z)
cleanup_and_exit()
print("---End Of File---")
