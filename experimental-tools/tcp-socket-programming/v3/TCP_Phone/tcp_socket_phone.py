# Cell phone udp socket programing

import socket
import time
import threading
import datetime as dt
import os
import sys
import argparse
import subprocess
import signal
from socket import error as SocketError
import errno


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host", type=str,
                        help="server ip address", default="140.112.20.183")   # Lab249 外網
    parser.add_argument("-d", "--device", type=str,
                        help="device", default=["unam"])
    parser.add_argument("-p", "--ports", type=int, nargs='+',    # input list of port numbers sep by 'space'
                        help="list of ul/dl port to bind")
    parser.add_argument("-b", "--bitrate", type=str,
                        help="target bitrate in bits/sec (0 for unlimited)", default="1M")
    parser.add_argument("-l", "--length", type=str,
                        help="length of buffer to read or write in bytes (packet size)", default="250")
    parser.add_argument("-t", "--time", type=int,
                        help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
    return parser.parse_args()

# ===================== Main Process =====================

args = parse_arguments()

HOST = args.host
device = args.device
ports = args.ports

if args.bitrate[-1] == 'k':
    bitrate = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bitrate = int(args.bitrate[:-1]) * 1e6
else:
    bitrate = int(args.bitrate)
    
length_packet = int(args.length)
total_time = args.time

expected_packet_per_sec = bitrate / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec

print(f'---{device}----')
# print(ports)
print("bitrate:", f'{args.bitrate}bps')

# ===================== Simple Socket =====================

# # 設定伺服器的主機和埠
# HOST = '127.0.0.1'
# PORT = 12345

# # 建立TCP客戶端
# client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect((HOST, PORT))

# while True:
#     message = input('請輸入訊息 (或輸入 "exit" 離開): ')
#     if message == 'exit':
#         break
#     client_socket.sendall(message.encode())
#     data = client_socket.recv(1024)
#     print('收到伺服器的回應:', data.decode())

# client_socket.close()

# ===================== Parameters =====================
# HOST = '140.112.20.183'  # Lab 249
pcap_path = '/sdcard/pcapdir'

# ===================== Global Variables =====================
stop_threads = False

# ===================== setup socket =====================

# Setup connections
rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# rx_socket.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
# rx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, (device+'\0').encode())  # 綁定特定網路介面
rx_socket.connect((HOST, ports[1]))  # 連線到指定的主機和埠

tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# tx_socket.setsockopt(socket.IPPROTO_TCP, TCP_CONGESTION, cong)
# tx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, (device+'\0').encode())  # 綁定特定網路介面
tx_socket.connect((HOST, ports[0]))  # 連線到指定的主機和埠

print(f'Create UL socket for {device} at {HOST}:{ports[0]}.')
print(f'Create DL socket for {device} at {HOST}:{ports[1]}.')

time.sleep(1)

# ===================== traffic capture =====================

if not os.path.isdir(pcap_path):
   os.system(f'mkdir {pcap_path}')

now = dt.datetime.today()
current_datetime = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
current_datetime = [x.zfill(2) for x in current_datetime]  # zero-padding to two digit
current_datetime = '-'.join(current_datetime[:3]) + '_' + '-'.join(current_datetime[3:])

pcap = os.path.join(pcap_path, f"client_pcap_BL_{device}_{ports[0]}_{ports[1]}_{current_datetime}_sock.pcap")
tcpproc = subprocess.Popen([f"tcpdump -i any port '({ports[0]} or {ports[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)
time.sleep(1)

# ===================== transmit & receive =====================

def receive(s, dev):
    global stop_threads
    # print(f"Waiting for indata to {dev} from server...")
    
    time_slot = 1
    capture_bytes = 0
    while not stop_threads:
        try:
            indata = s.recv(65535)

            # try:
            #     rx_start_time
            # except NameError:
            #     rx_start_time = time.time()

            # capture_bytes += len(indata)
            
            # # Show information
            # if time.time() - rx_start_time > time_slot:
            #     if capture_bytes <= 1000*1000/8:
            #         print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", "%g kbps"%(capture_bytes/1000*8))
            #     else:
            #         print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", "%g Mbps"%(capture_bytes/1000/1000*8))
            #     time_slot += 1
            #     capture_bytes = 0

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


def transmit(s):

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
            
            s.sendall(outdata)  # Send data over the connection
            seq += 1
        
            # if time.time() - start_time > time_slot:
            #     print("[%d-%d]"%(time_slot-1, time_slot), "transmit", seq-prev_transmit)
            #     time_slot += 1
            #     prev_transmit = seq

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

# Create and start Downlink receiving thread
t_rx = threading.Thread(target=receive, args=(rx_socket, device, ), daemon=True)
t_rx.start()

# Create and start Uplink transmission thread
t_tx = threading.Thread(target=transmit, args=(tx_socket,), daemon=True)
t_tx.start()

# ===================== wait for experiment end =====================

def cleanup_and_exit():
    
    # Close sockets
    tx_socket.close()
    rx_socket.close()
    
    # Kill tcpdump process
    os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
    
    print(f'{device} successfully closed.')
    # sys.exit()

time.sleep(3)
while not stop_threads:
    try:
        time.sleep(3)
        
    except KeyboardInterrupt:
        stop_threads = True
        
        # for i in range(10):
        #     # rx_socket.sendall('STOP'.encode())
        #     tx_socket.sendall('STOP'.encode())

# End without KeyboardInterrupt (Ctrl-C, Ctrl-Z)
cleanup_and_exit()
print(f"---End Of File ({device})---")
