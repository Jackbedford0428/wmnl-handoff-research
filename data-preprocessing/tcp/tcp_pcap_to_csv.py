#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Filename: tcp_pcap_to_csv.py
"""
Convert udp pcap into csv format and extract the features needed.

Usages:
(1) Decode only one file. 
$ python3 tcp_pcap_to_csv.py [-i <input filepath>]
$ python3 tcp_pcap_to_csv.py -i data/client_pcap_BL_sm03_3206_3207_2024-03-19_19-55-46_sock.pcap

(2) Decode files a batch of files on specific dates.
$ python3 tcp_pcap_to_csv.py [-d <date>[ date2 date3 ...]]
$ python3 tcp_pcap_to_csv.py -d 2024-03-19 2024-03-20

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2024-03-27
"""

"""
    Future Development Plans:
        (1) Adding function of onefile parsing.
        
"""
import os
import sys
import argparse
import time
import traceback
import subprocess
from pytictoc import TicToc

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *

__all__ = [
    "pcap_to_csv",
]


# ===================== Arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--onefile", type=str, help="input filepath")
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
args = parser.parse_args()


# ===================== Utils =====================
HASH_SEED = time.time()
LOGFILE = os.path.basename(__file__).replace('.py', '') + '_' + query_datetime() + '-' + generate_hex_string(HASH_SEED, 5) + '.log'

def pop_error_message(error_message=None, locate='.', signal=None, logfile=None, stdout=False, raise_flag=False):
    if logfile is None:
        logfile = LOGFILE
    
    file_exists = os.path.isfile(logfile)

    with open(logfile, "a") as f:
        if not file_exists:
            f.write(''.join([os.path.abspath(__file__), '\n']))
            f.write(''.join(['Start Logging: ', time.strftime('%Y-%m-%d %H:%M:%S'), '\n']))
            f.write('--------------------------------------------------------\n')
        
        if signal is None:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
            f.write(str(locate) + "\n")
            f.write(traceback.format_exc())
            f.write('--------------------------------------------------------\n')
        else:
            f.write(''.join([f'{signal}: ', time.strftime('%Y-%m-%d %H:%M:%S'), '\n']))
            f.write('--------------------------------------------------------\n')
    
    if raise_flag: raise
    
    if stdout:
        if signal is None:
            sys.stderr.write(traceback.format_exc())
            print('--------------------------------------------------------')
        else:
            print(signal)
            print('--------------------------------------------------------')


# ===================== Features =====================
def pcap_to_csv(fin, fout):
    
    # frame.number: the number of frame captured by tcpdump, start from 1
    # frame.time: frame arrival time 'Sep 29, 2022 16:24:58.254416000' (utc-8, CST)
    # frame.time_epoch: unix time (epoch) of frame arrival '1664439898.254416000' (utc-0)
    # frame.len: length of Ethernet frame
    # _ws.col.Protocol: TCP/UDP
    # ip.proto: 6(TCP)/17(UDP)
    # sll.pkttype: Sent by us (4): TX, Unicast to us (0): RX
    # ip.len: length of IP packet
    # ip.src: source IP
    # ip.dst: destination IP
    # udp.length/tcp.len: length of UDP segment/length of TCP segment
    # udp.srcport/tcp.srcport: source port
    # udp.dstport/tcp.srcport: destination port
    # data.len: length of data payload
    # data.data: content of data payload
    # udp.payload/tcp.payload: content of UDP/TCP data payload
    # _ws.col.Info: 48530 → 3211 [PSH, ACK] Seq=1 Ack=1 Win=128 Len=37 TSval=2269012485 TSecr=969346414
    
    s = f"tshark -r {fin} -T fields \
          -e frame.number -e frame.time -e frame.time_epoch -e frame.len \
          -e sll.pkttype -e _ws.col.Protocol \
          -e ip.proto -e ip.len -e ip.src -e ip.dst \
          -e tcp.len -e tcp.srcport -e tcp.dstport \
          -e data.len -e tcp.payload -e _ws.col.Info \
          -e tcp.seq_raw -e tcp.seq -e tcp.nxtseq -e tcp.ack_raw -e tcp.ack \
          -e tcp.analysis.acks_frame -e tcp.analysis.ack_rtt \
          -e tcp.analysis.initial_rtt -e tcp.analysis.bytes_in_flight -e tcp.analysis.push_bytes_sent \
          -e tcp.analysis.retransmission -e tcp.analysis.fast_retransmission \
          -e tcp.analysis.out_of_order \
          -E header=y -E separator=@ > {fout}"
    
    p = subprocess.Popen(s, shell=True, preexec_fn=os.setpgrp)
    while p.poll() is None:
        # print(p.pid, p.poll())
        time.sleep(1)


# ===================== Main Process =====================        
if __name__ == "__main__":
    if args.onefile is None:
        
        if args.dates is not None:
            dates = sorted(args.dates)
        else:
            raise TypeError("Please specify the date you want to process.")
        
        metadatas = metadata_loader(dates)
        
        print('\n================================ Start Processing ================================')
        pop_error_message(signal='Converting pcap to csv', stdout=True)
        
        for metadata in metadatas:
            try:
                print(metadata)
                print('--------------------------------------------------------')
                raw_dir = os.path.join(metadata[0], 'raw')
                middle_dir = os.path.join(metadata[0], 'middle')
                makedir(middle_dir)
                
                filenames = [s for s in os.listdir(raw_dir) if s.startswith(('server_pcap', 'client_pcap')) and s.endswith('.pcap')]
                for j, filename in enumerate(filenames):
                    t = TicToc(); t.tic()
                    fin = os.path.join(raw_dir, filename)
                    fout = os.path.join(middle_dir, filename.replace('.pcap', '.csv'))
                    print(f">>>>> {fin} -> {fout}")
                    # ******************************************************************
                    pcap_to_csv(fin, fout)
                    # ******************************************************************
                    t.toc(); print()
                
                print()
                    
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish converting pcap to csv', stdout=True)
        
    else:
        print(args.onefile)
