#!/usr/bin/python3
### Filename: udp_analysis.py

"""
Analyze packet capture of udp experiments.
Need to convert .pcap into .csv by tshark with script udp_pcap_to_csv.py

若 pcap, ho.csv 檔案為空:
    start_time = '-'
    end_time = '-'
    exp_time = 0

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2022/10/09
"""

"""
    Future Development Plan
        (1) Neglect filename start with ".~lock". (e.g., ".~lock.packet_info.csv#", ".~lock.client_pcap_BL_sm05_3210_3211_2022-09-29_16-24-57.csv#")
        (2) Output packet loss statistics and lanency.
        (3) Time synchronization. (preprocessing)
        (4) functionalize
        (5) 對一下 packet loss rate / packet loss number 是否與截圖 summary 相符
        (6) 若一開始就掉封包，建議改成外插法預測！
        (7) focus 在 UE 端狀況，因此 timestamp 應取 expected arrival time (DL) 和 transmit time (UL)，目前都是取 expected arrival time。
        (8) readline of csv 逐行（不要一次 handle 整個 dataframe，因為 payload 太長，RAM 會爆掉），目前先把 rxdf, txdf 分別處理，不要兩個都讀進來才處理（30min-500pps-pcap.csv 還夠用）
    
"""
import os
import sys
import argparse
import csv
import pandas as pd
import datetime as dt
import numpy as np
from pprint import pprint
from tqdm import tqdm
from pytictoc import TicToc
import traceback
from statistics import median
from statistics import mean
from statistics import mode
from statistics import stdev
from scipy import stats
from scipy import signal
import portion as P
import swifter

__all__ = [
    'parse_packet_info',
]

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
# database = "/Users/jackbedford/Desktop/MOXA/Code/data/"
# date = "2023-01-12"
dates = [
    "2023-03-26",
]
devices = sorted([
    # "sm00",
    # "sm01",
    # "sm02",
    # "sm03",
    # "sm04",
    # "sm05",
    # "sm06",
    # "sm07",
    # "sm08",
    "qc00",
    # "qc01",
    "qc02",
    "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (16, []),
    "_Bandlock_Udp_B3_B7_B8_RM500Q": (6, []),
    "_Bandlock_Udp_All_RM500Q": (4, []),
}

class Payl:
    LENGTH = 250              # (Bytes)
    TAG = "000425d401df5e76"  # 2 71828 3 1415926 (hex)            : 8-bytes
    OFS_TIME = (16, 24)       # epoch time of 'yyyy/mm/dd hh:mm:ss': 4-bytes
    OFS_USEC = (24, 32)       # microsecond (usec)                 : 4-bytes
    OFS_SEQN = (32, 40)       # sequence number (start from 1)     : 4-bytes
class ServerIP:
    PUBLIC = "140.112.20.183"  # 2F    
    PRIVATE = "192.168.1.251"  # 2F
    # PRIVATE = "192.168.1.248"  # 2F previous
    # PUBLIC = "140.112.17.209"  # 3F
    # PRIVATE = "192.168.1.108"  # 3F

class GParam:
    TRANS = 4
    RECV = 0
    TCP = 6
    UDP = 17

# DATA_RATE = 1000e3  # bits-per-second
# PKT_RATE = DATA_RATE / Payload.LENGTH / 8  # packets-per-second
# print("packet_rate (pps):", PKT_RATE, "\n")
# *****************************************************************************

# ****************************** Utils Functions ******************************
def makedir(dirpath, mode=0):  # mode=1: show message; mode=0: hide message
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

def epoch_to_utc8(ts):
    """
    Convert an epoch time into a readable format.
    Switch from utc-0 into utc-8.
    
    Args:
        ts (float): timestamp composed of datetimedec + microsecond (e.g., 1644051509.989306)
    Returns:
        (datetime.datetime): a readable timestamp (utc-8)
    """
    return (dt.datetime.utcfromtimestamp(ts) + dt.timedelta(hours=8))

def str_to_datetime(ts):
    """
    Convert a timestamp string in microseconds or milliseconds into datetime.datetime

    Args:
        ts (str): timestamp string (e.g., 2022-09-29 16:24:58.252615)
    Returns:
        (datetime.datetime)
    """
    try:
        ts_datetime = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
    except:
        ts_datetime = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return ts_datetime

def datetime_to_str(ts):
    """
    Convert a datetime timestamp in microseconds into str

    Args:
        ts (datetime.datetime): datetime timestamp (e.g., datetime.datetime(2022, 9, 29, 16, 24, 58, 252615))
    Returns:
        (str): timestamp string (e.g., 2022-09-29 16:24:58.252615)
    """
    try:
        ts_string = dt.datetime.strftime(ts, "%Y-%m-%d %H:%M:%S.%f")
    except:
        ts_string = dt.datetime.strftime(ts, "%Y-%m-%d %H:%M:%S")
    return ts_string
# *****************************************************************************

# **************************** Auxiliary Functions ****************************

def parse_packet_info(fin, fout, term, direct, proto):   
    new_header = ["seq", "rpkg", "frame_id", "frame_time", "frame_time_epoch", "pyl_time", "pyl_time_epoch"]
    timestamp_list = []
    seq_set = set()
    # Open the CSV file for reading
    with open(fin, 'r') as csvfile:
        # Create a CSV reader
        csvreader = csv.reader(csvfile, delimiter='@')

        # Read the header row
        header = next(csvreader)
        # print(header)
        
        # Iterate through the rows in the CSV file
        for content in tqdm(csvreader):
            # print(content)
            row = {k: v for k, v in zip(header, content)}
            
            # server/client; uplink/downlink
            if (term == "server" and direct == "ul") and int(row['sll.pkttype']) != GParam.RECV:
                continue
            elif (term == "client" and direct == "ul") and int(row['sll.pkttype']) != GParam.TRANS:
                continue
            elif (term == "server" and direct == "dl") and int(row['sll.pkttype']) != GParam.TRANS:
                continue
            elif (term == "client" and direct == "dl") and int(row['sll.pkttype']) != GParam.RECV:
                continue
            
            # udp/tcp
            if proto == "udp" and int(row['ip.proto']) != GParam.UDP:
                continue
            elif proto == "tcp" and int(row['ip.proto']) != GParam.TCP:
                continue
            
            # check customized packet
            if proto == "udp" and not ((int(row['udp.length']) > Payl.LENGTH) and (int(row['udp.length']) % Payl.LENGTH == 8)):
                continue
            elif proto == "tcp" and Payl.TAG not in row['tcp.payload']:
                continue
            
            # print(content)
            
            payload = row['udp.payload']
            rpkg_num = int(row['udp.length']) // Payl.LENGTH
            offset = [s * Payl.LENGTH * 2 for s in list(range(rpkg_num))]  # 1-Byte == 2-hex-digits
            
            sequence_list = []
            payload_time_list = []
            payload_time_epoch_list = []
            for ofs in offset:
                try:
                    datetimedec = int(payload[ofs + Payl.OFS_TIME[0] : ofs + Payl.OFS_TIME[1]], 16)
                    microsec = int(payload[ofs + Payl.OFS_USEC[0] : ofs + Payl.OFS_USEC[1]], 16)
                    seq = int(payload[ofs + Payl.OFS_SEQN[0] : ofs + Payl.OFS_SEQN[1]], 16)
                except:
                    print(traceback.format_exc())
                    print(row['frame.time'])
                    print(payload)
                    sys.exit(1)
                
                payload_time = epoch_to_utc8(datetimedec + microsec * 1e-6)
                
                sequence_list.append(seq)
                payload_time_list.append(payload_time)
                payload_time_epoch_list.append(datetimedec + microsec * 1e-6)
            
            # print("rpkg", rpkg_num)
            # print("frame_id", int(row['frame.number']))
            # print("frame_time", pd.to_datetime(row['frame.time']).tz_localize(None))
            # print("frame_time_epoch", float(row['frame.time_epoch']))
            # print("seq", sequence_list)
            # print("pyl_time", payload_time_list)
            # print("pyl_time_epoch", payload_time_epoch_list)
            
            for (seq, pyl_time, pyl_time_epoch) in zip(sequence_list, payload_time_list, payload_time_epoch_list):
                if (seq, pyl_time_epoch) not in seq_set:
                    timestamp_list.append([seq, rpkg_num, int(row['frame.number']), pd.to_datetime(row['frame.time']).tz_localize(None), float(row['frame.time_epoch']), pyl_time, pyl_time_epoch])
                    seq_set.add((seq, pyl_time_epoch))

    # print(seq_set)
    # print(timestamp_list)

    timestamp_list = sorted(timestamp_list, key = lambda v : v[0])

    print("output >>>", fout)
    with open(fout, 'w', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(new_header)
        writer.writerows(timestamp_list)
    
    return
# *****************************************************************************


if __name__ == "__main__":
    def fgetter(direction, terminal):
        if direction == "ul" and terminal == "client":
            tags = ("client_pcap_BL", "client_pcap_UL")
        elif direction == "dl" and terminal == "client":
            tags = ("client_pcap_BL", "client_pcap_DL")
        elif direction == "ul" and terminal == "server":
            tags = ("server_pcap_BL", "server_pcap_UL")
        elif direction == "dl" and terminal == "server":
            tags = ("server_pcap_BL", "server_pcap_DL")
        for filename in filenames:
            if filename.startswith(tags) and filename.endswith(".csv"):
                print(">>>>>", os.path.join(source_dir, filename))
                return os.path.join(source_dir, filename)
        print("No candidate file.")
        return None

    def main():
        ### detailed information for each udp packet's frame
        # udp_uplk_client_pkt_info   # udp ultx detailed info
        # udp_uplk_server_pkt_info   # udp ulrx detailed info
        # udp_dnlk_server_pkt_info   # udp dltx detailed info
        # udp_dnlk_client_pkt_info   # udp dlrx detailed info

        ### brief information for each udp packet
        # udp_uplk_client_pkt_brief  # udp ultx brief info
        # udp_uplk_server_pkt_brief  # udp ulrx brief info
        # udp_dnlk_server_pkt_brief  # udp dltx brief info
        # udp_dnlk_client_pkt_brief  # udp dlrx brief info

        ### dirpath
        # source_dir
        # target_dir

        ### parse detailed information & brief information for each packet
        
        ## client_pcap_BL or client_pcap_UL
        filepath = fgetter("uplink", "client")
        if filepath:
            t1 = TicToc()  # create instance of class
            t1.tic()       # Start timer
            parse_packet_info(filepath, os.path.join(target_dir, "udp_uplk_client_pkt_brief.csv"), "client", "ul", "udp")
            t1.toc()  # Time elapsed since t1.tic()
        
        ## client_pcap_BL or client_pcap_DL
        filepath = fgetter("downlink", "client")
        if filepath:
            t1 = TicToc()  # create instance of class
            t1.tic()       # Start timer
            parse_packet_info(filepath, os.path.join(target_dir, "udp_dnlk_client_pkt_brief.csv"), "client", "dl", "udp")
            t1.toc()  # Time elapsed since t1.tic()

        ## server_pcap_BL or server_pcap_UL
        filepath = fgetter("uplink", "server")
        if filepath:
            t1 = TicToc()  # create instance of class
            t1.tic()       # Start timer
            parse_packet_info(filepath, os.path.join(target_dir, "udp_uplk_server_pkt_brief.csv"), "server", "ul", "udp")
            t1.toc()  # Time elapsed since t1.tic()
        
        ## server_pcap_BL or server_pcap_DL
        filepath = fgetter("downlink", "server")
        if filepath:
            t1 = TicToc()  # create instance of class
            t1.tic()       # Start timer
            parse_packet_info(filepath, os.path.join(target_dir, "udp_dnlk_server_pkt_brief.csv"), "server", "dl", "udp")
            t1.toc()  # Time elapsed since t1.tic()
        print()
        
        
        files = ["udp_uplk_server_pkt_brief.csv", "udp_uplk_client_pkt_brief.csv", "udp_dnlk_server_pkt_brief.csv", "udp_dnlk_client_pkt_brief.csv"]
        files = [os.path.join(target_dir, s) for s in files]

        st_t = []
        ed_t = []
        for file in files:
            df = pd.read_csv(file)
            df['frame_time'] = pd.to_datetime(df['frame_time'])
            st_t.append(df.iloc[0]['frame_time'] - pd.Timedelta(seconds=5))
            ed_t.append(df.iloc[-1]['frame_time'] + pd.Timedelta(seconds=5))
            del df

        st_t = max(st_t)
        ed_t = min(ed_t)

        for file in files:
            df = pd.read_csv(file)
            df['frame_time'] = pd.to_datetime(df['frame_time'])
            df = df[(df['frame_time'] > st_t) & (df['frame_time'] < ed_t)]
            df.to_csv(file, index=False)
            
        return

    # ******************************* Check Files *********************************
    for date in dates:
        for expr, (times, traces) in exps.items():
            print(os.path.join(database, date, expr))
            for dev in devices:
                if not os.path.isdir(os.path.join(database, date, expr, dev)):
                    print("|___ {} does not exist.".format(os.path.join(database, date, expr, dev)))
                    continue
                
                print("|___", os.path.join(database, date, expr, dev))
                if traces == None:
                    # print(os.path.join(database, date, expr, dev))
                    continue
                elif len(traces) == 0:
                    traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
                
                print("|    ", times)
                traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
                if len(traces) != times:
                    print("***************************************************************************************")
                    print("Warning: the number of traces does not match the specified number of experiment times.")
                    print("***************************************************************************************")
                for trace in traces:
                    print("|    |___", os.path.join(database, date, expr, dev, trace))
            print()
    # *****************************************************************************

    # ******************************** Processing *********************************
    t = TicToc()  # create instance of class
    t.tic()       # Start timer
    for date in dates:
        for expr, (times, traces) in exps.items():
            for dev in devices:
                if not os.path.isdir(os.path.join(database, date, expr, dev)):
                    print("{} does not exist.\n".format(os.path.join(database, date, expr, dev)))
                    continue

                if traces == None:
                    print("------------------------------------------")
                    print(date, expr, dev)
                    print("------------------------------------------")
                    source_dir = os.path.join(database, date, expr, dev)
                    target_dir = os.path.join(database, date, expr, dev)
                    makedir(target_dir)
                    traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
                    # filenames = os.listdir(source_dir)
                    # main()
                    # continue
                elif len(traces) == 0:
                    traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
                
                traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
                for trace in traces:
                    print("------------------------------------------")
                    print(date, expr, dev, trace)
                    print("------------------------------------------")
                    source_dir = os.path.join(database, date, expr, dev, trace, "middle")
                    target_dir = os.path.join(database, date, expr, dev, trace, "middle")
                    if expr == "tsync":
                        source_dir = os.path.join(database, date, expr, dev, trace)
                        target_dir = os.path.join(database, date, expr, dev, trace)
                    makedir(target_dir)
                    filenames = os.listdir(source_dir)
                    main()
    t.toc()  # Time elapsed since t.tic()
    # *****************************************************************************
    