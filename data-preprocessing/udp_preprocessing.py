import os, sys
from bs4 import BeautifulSoup
from itertools import chain
from pytictoc import TicToc
import argparse
import shutil
import re
from pprint import pprint
import pandas as pd
import numpy as np
import datetime as dt
import itertools as it
import time
import csv
import json
from statistics import median
from statistics import mean
from statistics import mode
from statistics import stdev

pdir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # for jupyter-notebook
sys.path.insert(1, pdir)
from myutils import makedir

database = "/home/wmnlab/D/database/"
# database = "/Users/jackbedford/Desktop/MOXA/Code/data/"
dates = [
    "2023-06-24",
]
json_files = [
    "time_sync_lpt3.json",
]
json_files = [os.path.join(database, date, json_file) for date, json_file in zip(dates, json_files)]

exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    "Control_Group": (6, ["#{:02d}".format(i+1) for i in range(6)]),
    # "Modem_Action_Test": (5, ["#{:02d}".format(i+1) for i in range(5)]),
}
_devices = [
    # ["qc00", "qc01", "qc02", "qc03"],
    ["qc00", "qc03"],
    # ["qc00", "qc03"],
]
_schemes = [
    # ["All", "LTE", "B7", "B8"],
    ["All_0", "All_1"],
    # ["test1", "test2"],
]

class Payload:
    LENGTH = 250              # (Bytes)
    TAG = "000425d401df5e76"  # 2 71828 3 1415926 (hex)            : 8-bytes
    OFS_TIME = (16, 24)       # epoch time of 'yyyy/mm/dd hh:mm:ss': 4-bytes
    OFS_USEC = (24, 32)       # microsecond (usec)                 : 4-bytes
    OFS_SEQN = (32, 40)       # sequence number (start from 1)     : 4-bytes

DATA_RATE = 1000e3  # bits-per-second
PKT_RATE = DATA_RATE / Payload.LENGTH / 8  # packets-per-second
print("packet_rate (pps):", PKT_RATE, "\n")

from mi2log.mi_offline_analysis import mi_decode, error_handling

def fgetter():
    files_collection = []
    tags = "diag_log"
    for filename in filenames:
        if filename.startswith(tags) and filename.endswith(".mi2log"):
            files_collection.append(filename)
    return files_collection

def main():
    files_collection = fgetter()
    if len(files_collection) == 0:
        print("No candidate file.")
    for filename in files_collection:
        fin = os.path.join(source_dir, filename)
        fout = os.path.join(target_dir, "{}.txt".format(filename[:-7]))
        print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
        err_handle = mi_decode(fin, fout)
        err_handles.append(err_handle)
    print()

# ******************************* Check Files *********************************
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
err_handles = []
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
                filenames = os.listdir(source_dir)
                main()
                continue
            elif len(traces) == 0:
                traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
            
            traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
            for trace in traces:
                print("------------------------------------------")
                print(date, expr, dev, trace)
                print("------------------------------------------")
                source_dir = os.path.join(database, date, expr, dev, trace, "raw")
                target_dir = os.path.join(database, date, expr, dev, trace, "middle")
                makedir(target_dir)
                filenames = os.listdir(source_dir)
                main()
### Check errors
flag = False
for err_handle in err_handles:
    flag = error_handling(err_handle)
if not flag and err_handles:
    print("**************************************************")
    print("No error occurs!!")
    print("**************************************************")
t.toc()  # Time elapsed since t.tic()
# *****************************************************************************

from mi2log.xml_mi import xml_to_csv_rrc, xml_to_csv_ml1, xml_to_csv_nr_ml1

def fgetter():
    files_collection = []
    tags = "diag_log"
    for filename in filenames:
        if filename.startswith(tags) and filename.endswith(".txt"):
            files_collection.append(filename)
    return files_collection

def main():
    files_collection = fgetter()
    if len(files_collection) == 0:
        print("No candidate file.")
    for filename in files_collection:
        fin = os.path.join(source_dir, filename)
        fout1 = os.path.join(target_dir, "{}_rrc.csv".format(filename[:-4]))
        fout2 = os.path.join(target_dir, "{}_ml1.csv".format(filename[:-4]))
        fout3 = os.path.join(target_dir, "{}_nr_ml1.csv".format(filename[:-4]))
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout1))
        xml_to_csv_rrc(fin, fout1)
        # savemove(os.path.join(source_dir, "{}_rrc.csv".format(filename[:-4])), target_dir, "{}_rrc.csv".format(filename[:-4]))
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout2))
        xml_to_csv_ml1(fin, fout2)
        # savemove(os.path.join(source_dir, "{}_ml1.csv".format(filename[:-4])), target_dir, "{}_ml1.csv".format(filename[:-4]))
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout3))
        xml_to_csv_nr_ml1(fin, fout3)
        # savemove(os.path.join(source_dir, "{}_nr_ml1.csv".format(filename[:-4])), target_dir, "{}_nr_ml1.csv".format(filename[:-4]))
    print()

# ******************************* Check Files *********************************
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
err_handles = []
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
                filenames = os.listdir(source_dir)
                main()
                continue
            elif len(traces) == 0:
                traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
            
            traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
            for trace in traces:
                print("------------------------------------------")
                print(date, expr, dev, trace)
                print("------------------------------------------")
                source_dir = os.path.join(database, date, expr, dev, trace, "middle")
                target_dir = os.path.join(database, date, expr, dev, trace, "data")
                makedir(target_dir)
                filenames = os.listdir(source_dir)
                main()
t.toc()  # Time elapsed since t.tic()
# *****************************************************************************

from udp.pcap_to_csv import pcap_to_csv

def fgetter():
    files_collection = []
    tags = ("server_pcap", "client_pcap")
    for filename in filenames:
        if filename.startswith(tags) and filename.endswith(".pcap"):
            files_collection.append(filename)
    return files_collection

def main():
    files_collection = fgetter()
    if len(files_collection) == 0:
        print("No candidate file.")
    for filename in files_collection:
        fin = os.path.join(source_dir, filename)
        fout = os.path.join(target_dir, "{}.csv".format(filename[:-5]))
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
        err_handle = pcap_to_csv(fin, fout)
        err_handles.append(err_handle)
    print()

# ******************************* Check Files *********************************
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
err_handles = []
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
                source_dir = os.path.join(database, date, expr, dev, trace, "raw")
                target_dir = os.path.join(database, date, expr, dev, trace, "middle")
                if expr == "tsync":
                    source_dir = os.path.join(database, date, expr, dev, trace)
                    target_dir = os.path.join(database, date, expr, dev, trace)
                makedir(target_dir)
                filenames = os.listdir(source_dir)
                main()
### Check errors
flag = False
for err_handle in err_handles:
    flag = error_handling(err_handle)
if not flag and err_handles:
    print("**************************************************")
    print("No error occurs!!")
    print("**************************************************")
t.toc()  # Time elapsed since t.tic()
# *****************************************************************************

from udp.parse_info import filter, parse_packet_info, parse_brief_info

def fgetter(direction, terminal):
    if direction == "uplink" and terminal == "client":
        tags = ("client_pcap_BL", "client_pcap_UL")
    elif direction == "downlink" and terminal == "client":
        tags = ("client_pcap_BL", "client_pcap_DL")
    elif direction == "uplink" and terminal == "server":
        tags = ("server_pcap_BL", "server_pcap_UL")
    elif direction == "downlink" and terminal == "server":
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
        df = pd.read_csv(filepath, sep='@')
        df_ultx = filter(df.copy(), "uplink", "client", "udp")
        df_ultx = parse_packet_info(df_ultx, os.path.join(target_dir, "udp_uplk_client_pkt_info.csv"))
        df_ultx = parse_brief_info(df_ultx, os.path.join(target_dir, "udp_uplk_client_pkt_brief.csv"))
        os.system(f'rm {os.path.join(target_dir, "udp_uplk_client_pkt_info.csv")}') # Remove
        t1.toc()  # Time elapsed since t1.tic()
    
    ## client_pcap_BL or client_pcap_DL
    filepath = fgetter("downlink", "client")
    if filepath:
        t1 = TicToc()  # create instance of class
        t1.tic()       # Start timer
        df = pd.read_csv(filepath, sep='@')
        df_dlrx = filter(df.copy(), "downlink", "client", "udp")
        df_dlrx = parse_packet_info(df_dlrx, os.path.join(target_dir, "udp_dnlk_client_pkt_info.csv"))
        df_dlrx = parse_brief_info(df_dlrx, os.path.join(target_dir, "udp_dnlk_client_pkt_brief.csv"))
        os.system(f'rm {os.path.join(target_dir, "udp_dnlk_client_pkt_info.csv")}') # Remove
        t1.toc()  # Time elapsed since t1.tic()

    ## server_pcap_BL or server_pcap_UL
    filepath = fgetter("uplink", "server")
    if filepath:
        t1 = TicToc()  # create instance of class
        t1.tic()       # Start timer
        df = pd.read_csv(filepath, sep='@')
        df_ulrx = filter(df.copy(), "uplink", "server", "udp")
        df_ulrx = parse_packet_info(df_ulrx, os.path.join(target_dir, "udp_uplk_server_pkt_info.csv"))
        df_ulrx = parse_brief_info(df_ulrx, os.path.join(target_dir, "udp_uplk_server_pkt_brief.csv"))
        os.system(f'rm {os.path.join(target_dir, "udp_uplk_server_pkt_info.csv")}') # Remove
        t1.toc()  # Time elapsed since t1.tic()
    
    ## server_pcap_BL or server_pcap_DL
    filepath = fgetter("downlink", "server")
    if filepath:
        t1 = TicToc()  # create instance of class
        t1.tic()       # Start timer
        df = pd.read_csv(filepath, sep='@')
        df_dltx = filter(df.copy(), "downlink", "server", "udp")
        df_dltx = parse_packet_info(df_dltx, os.path.join(target_dir, "udp_dnlk_server_pkt_info.csv"))
        df_dltx = parse_brief_info(df_dltx, os.path.join(target_dir, "udp_dnlk_server_pkt_brief.csv"))
        os.system(f'rm {os.path.join(target_dir, "udp_dnlk_server_pkt_info.csv")}') # Remove
        t1.toc()  # Time elapsed since t1.tic()
    print()
    return

# ******************************* Check Files *********************************
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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

from udp.parse_loss_latency import get_loss, consolidate, compensate, get_latency, get_statistics

t = TicToc()  # create instance of class
t.tic()  # Start timer
# --------------------- (3) decode a batch of files (User Settings) ---------------------
# err_handles = []
for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
        
# --------------------- Phase 2: Parse packet loss & latency --------------------- 
### Read files
for date, json_file in zip(dates, json_files):
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
                target_dir1 = os.path.join(database, date, expr, dev, trace, "data")
                target_dir2 = os.path.join(database, date, expr, dev, trace, "statistics")
                if expr == "tsync":
                    source_dir = os.path.join(database, date, expr, dev, trace)
                    target_dir = os.path.join(database, date, expr, dev, trace)
                makedir(target_dir1)
                makedir(target_dir2)
                filenames = os.listdir(source_dir)
                t1 = TicToc()  # create instance of class
                t1.tic()  # Start timer
                
                dl_txdf = pd.read_csv(os.path.join(source_dir, "udp_dnlk_server_pkt_brief.csv"))
                dl_rxdf = pd.read_csv(os.path.join(source_dir, "udp_dnlk_client_pkt_brief.csv"))
                ul_txdf = pd.read_csv(os.path.join(source_dir, "udp_uplk_client_pkt_brief.csv"))
                ul_rxdf = pd.read_csv(os.path.join(source_dir, "udp_uplk_server_pkt_brief.csv"))
                
                dl_txseq = list(dl_txdf["seq"].array)
                dl_rxseq = list(dl_rxdf["seq"].array)
                dlst = max(dl_txseq[0], dl_rxseq[0])
                dlet = min(dl_txseq[-1], dl_rxseq[-1])
                # print(dlst, dlet)

                ul_txseq = list(ul_txdf["seq"].array)
                ul_rxseq = list(ul_rxdf["seq"].array)
                ulst = max(ul_txseq[0], ul_rxseq[0])
                ulet = min(ul_txseq[-1], ul_rxseq[-1])
                # print(ulst, ulet)

                st = max(dlst, ulst)
                et = min(dlet, ulet)
                # print("----------------")
                st += PKT_RATE * 5  # 開頭切5秒
                et -= PKT_RATE * 5  # 結尾切5秒
                # print(st, et)

                dl_txdf = dl_txdf[(dl_txdf["seq"] >= st) & (dl_txdf["seq"] <= et)].copy().reset_index(drop=True)
                dl_rxdf = dl_rxdf[(dl_rxdf["seq"] >= st) & (dl_rxdf["seq"] <= et)].copy().reset_index(drop=True)
                ul_txdf = ul_txdf[(ul_txdf["seq"] >= st) & (ul_txdf["seq"] <= et)].copy().reset_index(drop=True)
                ul_rxdf = ul_rxdf[(ul_rxdf["seq"] >= st) & (ul_rxdf["seq"] <= et)].copy().reset_index(drop=True)
                
                json_object = {}
                if os.path.isfile(json_file):
                    with open(json_file, 'r') as f:
                        json_object = json.load(f)
                else:
                    print('*************', json_file, 'does not exist! *************')
                delta = pd.DataFrame.from_dict(json_object, orient='index', columns=['delta']).reset_index(names='Timestamp')
                delta['Timestamp'] = pd.to_datetime(delta['Timestamp'])
                delta['timedelta'] = pd.to_timedelta(delta['delta'], unit='seconds')

                ### Downlink
                fout1_dl = os.path.join(target_dir1, "udp_dnlk_loss_latency.csv")
                fout2_dl = os.path.join(target_dir2, "udp_dnlk_loss_statistics.csv")
                fout3_dl = os.path.join(target_dir2, "udp_dnlk_excl_statistics.csv")
                
                losdf = get_loss(dl_rxdf.copy(), dl_txdf.copy())
                latdf = consolidate(dl_rxdf.copy(), dl_txdf.copy())
                df = pd.concat([latdf, losdf], axis=0)
                df = df.sort_values(by=["seq"]).reset_index(drop=True)
                df = compensate(df.copy(), "dl", delta.copy())
                df = get_latency(df.copy(), "dl")
                get_statistics(df.copy(), fout1_dl, fout2_dl, fout3_dl)
                
                ### Uplink
                fout1_ul = os.path.join(target_dir1, "udp_uplk_loss_latency.csv")
                fout2_ul = os.path.join(target_dir2, "udp_uplk_loss_statistics.csv")
                fout3_ul = os.path.join(target_dir2, "udp_uplk_excl_statistics.csv")
                
                losdf = get_loss(ul_rxdf.copy(), ul_txdf.copy())
                latdf = consolidate(ul_rxdf.copy(), ul_txdf.copy())
                df = pd.concat([latdf, losdf], axis=0)
                df = df.sort_values(by=["seq"]).reset_index(drop=True)
                df = compensate(df.copy(), "ul", delta.copy())
                df = get_latency(df.copy(), "ul")
                get_statistics(df.copy(), fout1_ul, fout2_ul, fout3_ul)
                
                t1.toc()
# ### Check errors
# flag = False
# for err_handle in err_handles:
#     flag = error_handling(err_handle)
# if not flag:
#     print("**************************************************")
#     print("No error occurs!!")
#     print("**************************************************")
t.toc()  # Time elapsed since t.tic()

#! Extra
database = "/home/wmnlab/D/database/"
# database = "/Users/jackbedford/Desktop/MOXA/Code/data/"
dates = [
    "2023-06-24",
]
json_files = [
    "time_sync_lpt3.json",
]
json_files = [os.path.join(database, date, json_file) for date, json_file in zip(dates, json_files)]

exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    "Control_Group": (6, ["#{:02d}".format(i+1) for i in range(6)]),
    "Modem_Action_Test": (5, ["#{:02d}".format(i+1) for i in range(5)]),
}
_devices = [
    # ["qc00", "qc01", "qc02", "qc03"],
    ["qc00", "qc03"],
    ["qc00", "qc03"],
]
_schemes = [
    # ["All", "LTE", "B7", "B8"],
    ["All_0", "All_1"],
    ["test1", "test2"],
]

for date in dates:
    for (expr, (times, traces)), devices, schemes in zip(exps.items(), _devices, _schemes):
        for trace in traces:
            target_dir = os.path.join(database, date, expr, "combo", trace)
            makedir(target_dir)
            for tag in ["dnlk", "uplk"]:
            # for tag in ["dnlk",]:
            # for tag in ["uplk",]:
                print("------------------------------------------")
                print(date, expr, trace, tag)  
                print("------------------------------------------")
                t = TicToc()
                t.tic()
                dfs = []
                for i, (dev, scheme) in enumerate(zip(devices, schemes)):
                    source_dir = os.path.join(database, date, expr, dev, trace, "data")
                    dfs.append(pd.read_csv(os.path.join(source_dir, f"udp_{tag}_loss_latency.csv")))
                ### TODO 1
                st, et = [], []
                for i, (dev, scheme) in enumerate(zip(devices, schemes)):
                    st.append(dfs[i]['seq'].array[0])
                    et.append(dfs[i]['seq'].array[-1])
                st, et = max(st), min(et)
                for i, (dev, scheme) in enumerate(zip(devices, schemes)):
                    dfs[i] = dfs[i][(dfs[i]['seq'] >= st) & (dfs[i]['seq'] <= et)]
                    dfs[i].reset_index(drop=True, inplace=True)
                df = dfs[0][['seq', 'Timestamp']]
                for i, (dev, scheme) in enumerate(zip(devices, schemes)):
                    dfs[i] = dfs[i][['xmit_time','arr_time','lost','excl','latency']]
                    dfs[i].rename(
                        columns={
                            'xmit_time': f'xmit_time_{scheme}',
                            'arr_time': f'arr_time_{scheme}',
                            'lost': f'lost_{scheme}',
                            'excl': f'excl_{scheme}',
                            'latency': f'latency_{scheme}',
                        }, inplace=True
                    )
                df = pd.concat([df, *dfs], axis=1)
                ### TODO 2
                # xs = list(itertools.combinations(range(len(schemes)), 2))
                xs = list(it.combinations(range(len(schemes)), 2))
                for x in xs:
                    df[f'lost_{schemes[x[0]]}+{schemes[x[1]]}'] = df[f'lost_{schemes[x[0]]}'] & df[f'lost_{schemes[x[1]]}']
                    df[f'excl_{schemes[x[0]]}+{schemes[x[1]]}'] = df[f'excl_{schemes[x[0]]}'] & df[f'excl_{schemes[x[1]]}']
                    df[f'latency_{schemes[x[0]]}+{schemes[x[1]]}'] = df[[f'latency_{schemes[x[0]]}', f'latency_{schemes[x[1]]}']].min(axis=1)
                fout1 = os.path.join(target_dir, f"udp_{tag}_combo_loss_latency.csv")
                print("output >>>", fout1)
                df.to_csv(fout1, index=False)
                ### TODO 3
                colnames = []
                data = []
                print('', 'loss(%)', 'excl(%)', 'latency', 'max_latency', 'min_latency', 'negative_ratio(%)', sep='\t')
                for i, (dev, scheme) in enumerate(zip(devices, schemes)):
                    _df = df[df[f'lost_{scheme}'] == False]
                    loss = round(sum(df[f'lost_{scheme}']) / (len(df)+1e-9) * 100, 3)
                    excl = round(sum(_df[f'excl_{scheme}']) / (len(_df)+1e-9) * 100, 3)
                    latency = round(mean(_df[f'latency_{scheme}']), 6)
                    max_lat = round(max(_df[f'latency_{scheme}']), 6)
                    min_lat = round(min(_df[f'latency_{scheme}']), 6)
                    neg_ratio = round(sum(_df[f'latency_{scheme}'] < 0) / (len(_df)+1e-9) * 100, 3)
                    data = [*data, *[loss, excl, latency, max_lat, min_lat, neg_ratio]]
                    colnames = [*colnames, *[f'lost_{scheme}', f'excl_{scheme}', f'latency_{scheme}', f'max_lat_{scheme}', f'min_lat_{scheme}', f'negative_ratio_{scheme}']]
                    # print(scheme, round(loss, 3), round(excl, 3), latency, max_lat, sep='\t')
                    print(scheme, loss, excl, latency, max_lat, min_lat, neg_ratio, sep='\t')
                for x in xs:
                    _df = df[df[f'lost_{schemes[x[0]]}+{schemes[x[1]]}'] == False]
                    loss = round(sum(df[f'lost_{schemes[x[0]]}+{schemes[x[1]]}']) / (len(df)+1e-9) * 100, 3)
                    excl = round(sum(_df[f'excl_{schemes[x[0]]}+{schemes[x[1]]}']) / (len(_df)+1e-9) * 100, 3)
                    latency = round(mean(_df[f'latency_{schemes[x[0]]}+{schemes[x[1]]}']), 6)
                    max_lat = round(max(_df[f'latency_{schemes[x[0]]}+{schemes[x[1]]}']), 6)
                    min_lat = round(min(_df[f'latency_{schemes[x[0]]}+{schemes[x[1]]}']), 6)
                    neg_ratio = round(sum(_df[f'latency_{schemes[x[0]]}+{schemes[x[1]]}'] < 0) / (len(_df)+1e-9) * 100, 3)
                    data = [*data, *[loss, excl, latency, max_lat, min_lat, neg_ratio]]
                    colnames = [*colnames, *[f'lost_{schemes[x[0]]}+{schemes[x[1]]}', f'excl_{schemes[x[0]]}+{schemes[x[1]]}', f'latency_{schemes[x[0]]}+{schemes[x[1]]}', f'max_lat_{schemes[x[0]]}+{schemes[x[1]]}', f'min_lat_{schemes[x[0]]}+{schemes[x[1]]}', f'negative_ratio_{schemes[x[0]]}+{schemes[x[1]]}']]
                    # print(f'{schemes[x[0]]}+{schemes[x[1]]}', round(loss, 3), round(excl, 3), latency, max_lat, sep='\t')
                    print(f'{schemes[x[0]]}+{schemes[x[1]]}', loss, excl, latency, max_lat, min_lat, neg_ratio, sep='\t')
                fout2 = os.path.join(target_dir, f"udp_{tag}_combo_statistics.csv")
                print("output >>>", fout2)
                with open(fout2, "w", newline='') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(colnames)
                    writer.writerow(data)
                t.toc()
                ### TODO END

for date in dates:
    for (expr, (times, traces)), devices in zip(exps.items(), _devices):
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
                filenames = os.listdir(source_dir)
                continue
            elif len(traces) == 0:
                traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
            
            traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
            for trace in traces:
                print("------------------------------------------")
                print(date, expr, dev, trace)
                print("------------------------------------------")
                source_dir = os.path.join(database, date, expr, dev, trace, "data")
                target_dir = os.path.join(database, date, expr, dev, trace, "data")
                
                for filename in sorted(os.listdir(source_dir)):
                    if not filename.endswith('.csv'):
                        continue
                    filepath = os.path.join(source_dir, filename)
                    print(filepath, os.path.isfile(filepath))
                    print(f'{filepath[:-4]}.pkl')
                    
                    fp = pd.read_csv(filepath)
                    fp.to_pickle(f'{filepath[:-4]}.pkl')
