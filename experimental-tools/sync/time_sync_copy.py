#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import socket
import time
import json
import datetime as dt

HOST = '140.112.20.183'
PORT = 3298
dirpath = './log'

def mean(numbers):
    if len(numbers) == 0:
        return 0
    _sum = 0
    for item in numbers:
        _sum += item
    return _sum / len(numbers)

def quantile(data, q):
    q /= 100
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = q * (n - 1)
    if index.is_integer():
        return sorted_data[int(index)]
    else:
        lower = sorted_data[int(index)]
        upper = sorted_data[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

def qset_bdd(client_rtt):
    # 找 outliers 的判斷範圍
    sorted_rtt = sorted([s[3] for s in client_rtt])
    
    upper_q = quantile(sorted_rtt, 75)
    lower_q = quantile(sorted_rtt, 25)
    iqr = (upper_q - lower_q) * 1.5
    
    qset = (lower_q - iqr, upper_q + iqr)
    return qset

def clock_diff(timestamp_server, timestamp_client):
    client = [[*s, float(s[2]) - float(s[1])] for s in timestamp_client]
    server = timestamp_server[:]
    
    diff = 0.0
    qset = qset_bdd(client)
    deltas = []
    for i in range(len(client)):
        RTT = client[i][3]
        if (RTT < qset[0]) or (RTT > qset[1]):
            continue
        cen_client = (float(client[i][2]) + float(client[i][1]))/2
        cen_server = (float(server[i][2]) + float(server[i][1]))/2
        diff = cen_server - cen_client
        deltas.append(diff)
    
    sorted_deltas = sorted(deltas)
    upper_q = quantile(sorted_deltas, 75)
    lower_q = quantile(sorted_deltas, 25)
    iqr = (upper_q - lower_q) * 1.5
    qset = (lower_q - iqr, upper_q + iqr)
    
    deltas = [s for s in deltas if s >= qset[0] and s <= qset[1]]
    diff = mean(deltas)
    print(len(deltas), qset)

    # diff > 0: client is behind server by abs(diff) seconds
    # diff < 0: client is ahead of server by abs(diff) seconds
    return diff

# client
if sys.argv[1] == '-c':
    
    now = dt.datetime.today()
    date = [str(x) for x in [now.year, now.month, now.day]]
    date = [x.zfill(2) for x in date]
    date = '-'.join(date)
    
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    
    server_addr = (HOST, PORT)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(3)
    num_packet_per_round = 10
    packet_interval = 0

    timestamp_client = []
    timestamp_server = []

    i = 0
    ctmo_cnt = 0
    while i <= num_packet_per_round:
        time0 = time.time()
        outdata = str(i).zfill(3)
        s.sendto(outdata.encode(), server_addr)
        s.settimeout(3)
        try:
            indata, addr = s.recvfrom(1024)
            time1 = time.time()
            indata = indata.decode()
            ctmo_cnt = 0
        except:
            print("timeout", outdata)
            ctmo_cnt += 1
            if ctmo_cnt == 3:
                break
            continue
        print(outdata, time0, time1, "RTT =", (time1-time0)*1000, "ms")
        timestamp_client.append([outdata, time0, time1])
        timestamp_server.append(indata.split(' '))
        time.sleep(packet_interval)
        i += 1
    outdata = 'end'
    s.sendto(outdata.encode(), server_addr)
    
    current_time = dt.datetime.now()
    diff = clock_diff(timestamp_server, timestamp_client)
    print(current_time, diff, "seconds")
    
    json_file = os.path.join(dirpath, f'time_sync_{date}.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as f:
            json_object = json.load(f)
    else:
        json_object = {}
    json_object[str(current_time)] = diff
    with open(json_file, 'w') as f:
        json.dump(json_object, f)

# server
elif sys.argv[1] == '-s':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', PORT))

    while True:
        print('server start at: %s:%s' % ('0.0.0.0', PORT))
        print('wait for connection...')

        while True:
            indata, addr = s.recvfrom(1024)
            time0 = time.time()
            indata = indata.decode()
            if indata == 'end':
                break
            print('recvfrom ' + str(addr) + ': ' + indata)
            
            time1 = time.time()
            outdata = f'{indata} {time0} {time1}'
            s.sendto(outdata.encode(), addr)
            print(indata, time0, time1)
