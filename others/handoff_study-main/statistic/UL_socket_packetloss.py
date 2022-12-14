import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tqdm
import os

import dpkt
import sys

import socket

def get_loss_latency(pcap_filename):

    f = open(pcap_filename, "rb")
    pcap = dpkt.pcap.Reader(f)

    timestamp_list = []

    #This for loop parse the payload of the iperf3 UDP packets and store the timestamps and the sequence numbers in timestamp_list; 
    #The timestamp is stored in the first 8 bytes, and the sequence number is stored in the 9~12 bytes
    #-----------------------------------------------------------------------------------------------

    seq_set = set()
    for ts, buf in pcap:

        #Extract payload of the UDP packet
        #---------------------------------
        eth = dpkt.sll.SLL(buf)  
        
        if (len(eth.data) - (20+8)) % 250 == 0:    # We set the payload length to be 250 in iperf, so here we set the length checking to be 250 + (4+20+8)
            
            ip = eth.data
            udp = ip.data
            
            dst_ip_addr_str = socket.inet_ntoa(ip.dst)
            if dst_ip_addr_str == "192.168.1.251":
                continue
            #print("haha")
            
            #------------only DL data left--------------
            #bug fix: duplicate packets | credit: JingYou
            duplicate_num = (len(eth.data) - (20+8)) // 250
            
            datetimedec = int(udp.data[0:8].hex(), 16)
            microsec = int(udp.data[8:16].hex(), 16)

            seq = int(udp.data[16:24].hex(), 16)
            print(duplicate_num, seq)
            
            
            if seq == 1:
                timestamp_list = []
            for i in range(duplicate_num):
                if seq+i not in seq_set:
                    seq_set.add(seq+i)
                    timestamp_list.append((ts, datetimedec, microsec, seq+i))
    timestamp_list = sorted(timestamp_list, key = lambda v : v[3])  #We consider out of order UDP packets

    pointer = 1
    timestamp_store = None
    loss_timestamp = []

    #Checking packet loss...
    #----------------------------------------------
    for timestamp in timestamp_list:
        if timestamp[3] == pointer:
            timestamp_store = timestamp
        else:
            if timestamp_store == None:
                continue
            loss_linspace = np.linspace(timestamp_store, timestamp, timestamp[3]-pointer+2)
            
            for i in loss_linspace:
                loss_time = dt.datetime.utcfromtimestamp(i[0]) + dt.timedelta(hours=8)
                loss_timestamp.append(loss_time)
                
        pointer = timestamp[3] + 1
        
    #x and y stands for the timestamp (x) and the one-way latency (y) on the timestamp, respectively
    #----------------------------------------------
    x = []
    y = []
    
    for i in range(len(timestamp_list)):
        arrival_time = dt.datetime.utcfromtimestamp(timestamp_list[i][0]) + dt.timedelta(seconds=3600*8) #for pcap packets, the timestamps are needed to add 8 hours (timezone) 
        x.append(arrival_time)
        
        y.append( ( timestamp_list[i][0]+3600*8 - (timestamp_list[i][1] + timestamp_list[i][2]/1000000. + 3600*8) ) * 1000 )
    
    latency = [x, y]
    
    return loss_timestamp, latency
    
server_UL_file = sys.argv[1]

loss_timestamp, latency = get_loss_latency(server_UL_file)

print("number of packet", len(latency[0]))
print("number of lost packet", len(loss_timestamp))
print("packet loss rate", len(loss_timestamp) / len(latency[0]))