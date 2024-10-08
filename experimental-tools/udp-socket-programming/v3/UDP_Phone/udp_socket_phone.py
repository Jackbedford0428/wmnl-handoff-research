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

#=================argument parsing======================
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
                    help="time in seconds to transmit for (default 1.5 hour = 5400 secs)", default=5400)
args = parser.parse_args()



length_packet = int(args.length)

if args.bitrate[-1] == 'k':
    bandwidth = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bandwidth = int(args.bitrate[:-1]) * 1e6
else:
    bandwidth = int(args.bitrate)

print("bitrate:", bandwidth)

total_time = args.time

expected_packet_per_sec = bandwidth / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec


#=================other variables========================
HOST = args.host # Lab 249
dev = args.device
ports = args.ports

length_packet = int(args.length)

if args.bitrate[-1] == 'k':
    bandwidth = int(args.bitrate[:-1]) * 1e3
elif args.bitrate[-1] == 'M':
    bandwidth = int(args.bitrate[:-1]) * 1e6
else:
    bandwidth = int(args.bitrate)

# print("bitrate:", bandwidth)

total_time = args.time

expected_packet_per_sec = bandwidth / (length_packet << 3)
sleeptime = 1.0 / expected_packet_per_sec

print(ports)

#=================gloabal variables======================
stop_threads = False

# Function define
def give_server_DL_addr():
    
    outdata = 'hello'
    rx_socket.sendto(outdata.encode(), (HOST, ports[1]))

def receive(s, dev): # s should be rx_socket

    global stop_threads
    # print(f"wait for indata to {dev} from server...")

    seq = 1
    prev_receive = 1
    time_slot = 1

    while not stop_threads:
        try:

            indata, _ = s.recvfrom(1500)

            try: start_time
            except NameError:
                start_time = time.time()

            if len(indata) != length_packet:
                print("packet with strange length: ", len(indata))

            seq = int(indata.hex()[32:40], 16)
            ts = int(int(indata.hex()[16:24], 16)) + float("0." + str(int(indata.hex()[24:32], 16)))

            # # Show information
            # if time.time()-start_time > time_slot:
            #     print(f"{dev} [{time_slot-1}-{time_slot}]", "receive", seq-prev_receive)
            #     time_slot += 1
            #     prev_receive = seq
        except KeyboardInterrupt:
            print('Manually interrupted.')
            stop_threads = True

        except Exception as inst:
            print("Error: ", inst)
            stop_threads = True

def transmit(s):

    global stop_threads
    print("start transmission: ")
    
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
               
            s.sendto(outdata, (HOST, ports[0]))
            seq += 1
        
            # if time.time()-start_time > time_slot:
            #     print("[%d-%d]"%(time_slot-1, time_slot), "transmit", seq-prev_transmit)
            #     time_slot += 1
            #     prev_transmit = seq

        except Exception as e:
            print(e)
            stop_threads = True
    stop_threads = True
    print("---transmission timeout---")
    print("transmit", seq, "packets")

# Create DL receive and UL transmit multi-client sockets
rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f'Create DL socket.')
print(f'Create UL socket.')

# Transmit data from receive socket to server DL port to let server know addr first

while True:
    give_server_DL_addr()
    # x = input('Continue? (y/n) ')
    # if x == 'n':
    break

# Start subprocess of tcpdump
now = dt.datetime.today()
n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
n = [x.zfill(2) for x in n]  # zero-padding to two digit
n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])

pcap_path = '/sdcard/experiment_log/{}/client_pcap'.format(n[:10])
if not os.path.isdir(pcap_path):
   print("makedir: {pcap_path}")
   os.makedirs(pcap_path)


pcap = os.path.join(pcap_path, f"client_pcap_BL_{dev}_{ports[0]}_{ports[1]}_{n}_sock.pcap")
tcpproc = subprocess.Popen([f"tcpdump -i any port '({ports[0]} or {ports[1]})' -w {pcap}"], shell=True, preexec_fn=os.setpgrp)

time.sleep(1)

# Create and start DL receive multipleprocess
t_rx = threading.Thread(target=receive, args=(rx_socket, dev, ), daemon=True)
t_rx.start()

# Create and start UL transmission multiprocess
t_tx = threading.Thread(target=transmit, args=(tx_socket,), daemon=True)
t_tx.start()

# Main Process waitng...
# try:

#     while True:
#         time.sleep(10)

# except KeyboardInterrupt:

#     # Kill tcpdump process
#     print('Killing tcpdump process...')
#     os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)

#     time.sleep(1)
#     print(f'{dev} successfully closed.')
#     sys.exit()

def cleanup_and_exit():
    
    # Close sockets
    tx_socket.close()
    rx_socket.close()
    
    # Kill tcpdump process
    os.killpg(os.getpgid(tcpproc.pid), signal.SIGTERM)
    
    print(f'{dev} successfully closed.')
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
print(f"---End Of File ({dev})---")
