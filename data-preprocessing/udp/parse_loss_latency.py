#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: parse_loss_latency.py
import os
import sys
import argparse
import time
import traceback
from pytictoc import TicToc
from pprint import pprint
from tqdm import tqdm
import csv
import json
import pandas as pd
import datetime as dt
import numpy as np
import portion as P
from statistics import median
from statistics import mean
from statistics import mode
from statistics import stdev
from scipy import stats
from scipy import signal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *

__all__ = [
    'get_loss',
    'consolidate',
    'compensate',
    'get_latency',
    'get_statistics',
]


# ===================== Arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--onefile", type=str, help="input filepath")
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
args = parser.parse_args()

class Payl:
    LENGTH = 250              # (Bytes)
    TAG = "000425d401df5e76"  # 2 71828 3 1415926 (hex)            : 8-bytes
    OFS_TIME = (16, 24)       # epoch time of 'yyyy/mm/dd hh:mm:ss': 4-bytes
    OFS_USEC = (24, 32)       # microsecond (usec)                 : 4-bytes
    OFS_SEQN = (32, 40)       # sequence number (start from 1)     : 4-bytes

XMIT = 4; RECV = 0
UDP = 17; TCP = 6

DATA_RATE = 1000e3  # bits-per-second
PKT_RATE = DATA_RATE / Payl.LENGTH / 8  # packets-per-second
print("packet_rate (pps):", PKT_RATE, "\n")


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
def get_loss(rxdf, txdf):
    rxdf['frame_time'] = pd.to_datetime(rxdf['frame_time'])  # arrival.time
    rxdf['pyl_time'] = pd.to_datetime(rxdf['pyl_time'])  # payload.time

    timestamp_list = list(map(list, zip(rxdf['seq'].astype(int).array, rxdf['frame_time_epoch'].astype(float).array)))
    timestamp_store = timestamp_list[0]
    loss_timestamp_list = []
    count = 0  # to count the total number of packet losses
    _eseq = timestamp_list[0][0] # next expected sequence number: ignore the first-N lost packets if existing.
    for i in tqdm(range(len(rxdf))):
        timestamp = timestamp_list[i]
        if timestamp[0] == _eseq:
            ### received packet's sequence number as expected
            pass
        else:
            ### packet losses occur
            ### 可處理連續掉 N 個封包的狀況
            ### timestamp_store: 前一刻收到的封包
            ### timestamp: 此時此刻收到的封包
            ### _eseq 為預期收到的封包 sequence number (前一刻收到的 seq number + 1)
            ### rxdf.loc[i, 'sequence.number'] 為此時此刻收到的封包 seq
            ### rxdf.loc[i, 'sequence.number']-pointer+2 == 遺漏的封包數+2 (頭+尾)，因此要去頭去尾才是實際遺漏的封包
            n = timestamp[0] - _eseq + 2
            loss_linspace = np.linspace(timestamp_store, timestamp, n)
            loss_linspace = loss_linspace[1:-1]  # 去頭去尾
            for item in loss_linspace:
                count += 1
                loss_time = [round(item[0]), epoch_to_datetime(item[1]), item[1]]  # (expected) arrival timestamp
                loss_timestamp_list.append(loss_time)
        # Update information
        timestamp_store = timestamp
        _eseq = timestamp[0] + 1
    
    ### add payload & transmitted timestamp
    tx_ts_arr = list(zip(txdf['seq'].array, txdf['pyl_time'].array, txdf['pyl_time_epoch'].array, txdf['frame_time'].array, txdf['frame_time_epoch'].array))
    j = 0
    N = len(loss_timestamp_list)
    for i in tqdm(range(len(tx_ts_arr))):
        if N == 0:
            break
        if tx_ts_arr[i][0] == loss_timestamp_list[j][0]:
            loss_timestamp_list[j].append(tx_ts_arr[i][1])
            loss_timestamp_list[j].append(tx_ts_arr[i][2])
            loss_timestamp_list[j].append(tx_ts_arr[i][3])
            loss_timestamp_list[j].append(tx_ts_arr[i][4])
            j += 1
            if j == N:
                break
    ### 因為是由 Client 端主動開始和結束實驗，且程式邏輯為: 開 Tcpdump -> 開 iperf -> 關 Tcpdump -> 關 iperf
    ### 因此 Uplink TX 的 MAX_SEQ 會比 RX 小，Downlink TX 的 MAX_SEQ 會比 RX 大。
    loss_timestamp_list = [item for item in loss_timestamp_list if len(item) == 7]
    
    N = len(loss_timestamp_list)
    loss_timestamps = list(map(list, zip(*loss_timestamp_list)))
    df = pd.DataFrame()
    if len(loss_timestamps) > 0:
        df = pd.DataFrame.from_dict(
            {
                "seq": loss_timestamps[0],
                "rpkg": [None] * N,
                "frame_id": [None] * N,
                "Timestamp": loss_timestamps[3],  # payload.time
                "Timestamp_epoch": loss_timestamps[4],  # payload.time_epoch
                "lost": [True] * N,
                "latency": [float('inf')] * N,
                "xmit_time": loss_timestamps[5],
                "xmit_time_epoch": loss_timestamps[6],
                "arr_time": loss_timestamps[1],
                "arr_time_epoch": loss_timestamps[2],
            }
        )
    return df

def consolidate(rxdf, txdf):
    ### add transmitted timestamp
    rxseq = rxdf['seq'].array
    txseq = txdf['seq'].array
    txts = txdf['frame_time'].array
    txts_epoch= txdf['frame_time_epoch'].array
    rx_txts_arr = []
    rx_txts_epoch_arr = []
    j = 0
    N = len(txdf)
    M = len(rxdf)
    # print(N, M)
    for i in tqdm(range(len(rxseq))):
        while j != N and txseq[j] != rxseq[i]:
            j += 1
        if j != N:
            rx_txts_arr.append(txts[j])
            rx_txts_epoch_arr.append(txts_epoch[j])
    df = rxdf.join(pd.DataFrame({'xmit_time' : rx_txts_arr, 'xmit_time_epoch' : rx_txts_epoch_arr}))
    df = df.dropna(how='any', subset=['xmit_time', 'xmit_time_epoch'], axis=0)
    df = df.rename(
        columns={
            "frame_time": "arr_time",
            "frame_time_epoch": "arr_time_epoch",
            "pyl_time": "Timestamp",
            "pyl_time_epoch": "Timestamp_epoch",
        }
    )
    df["lost"] = False
    df["latency"] = 0
    df = df[["seq", "rpkg", "frame_id", "Timestamp", "Timestamp_epoch", "lost", "latency", "xmit_time", "xmit_time_epoch", "arr_time", "arr_time_epoch"]]
    return df

def interp(x, y, ratio):
    """
    Interpolation

    Args:
        x, y (datetime.datetime)
        ratio (float): a decimal numeral in a range [0, 1]
    Returns:
        (datetime.datetime): breakpoint of interpolation
    """
    return x + (y - x) * ratio

def compensate(df, mode, delta=pd.DataFrame()):
    ### compensate clock difference
    if delta.empty:
        return df
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['xmit_time'] = pd.to_datetime(df['xmit_time'])
    df['arr_time'] = pd.to_datetime(df['arr_time'])
    delta['Timestamp'] = pd.to_datetime(delta['Timestamp'])
    
    bm_timestamp = df.at[0, 'Timestamp']
    # epodelta, timedelta = delta[delta['Timestamp'] < bm_timestamp].reset_index(drop=True).iloc[-1][['delta', 'timedelta']]
    delta_o_delta = (delta["Timestamp"] - bm_timestamp).dt.total_seconds().abs()
    epodelta, timedelta = delta.loc[delta_o_delta.argmin(), ['delta', 'timedelta']]
    # print(delta[delta['Timestamp'] < bm_timestamp])
    print(epodelta, "seconds")
    
    if mode == "dl":
        df['arr_time_epoch'] = df['arr_time_epoch'].add(epodelta)
        df['arr_time'] = df['arr_time'].add(timedelta)
    elif mode == "ul":
        df['Timestamp_epoch'] = df['Timestamp_epoch'].add(epodelta)
        df['Timestamp'] = df['Timestamp'].add(timedelta)
        df['xmit_time_epoch'] = df['xmit_time_epoch'].add(epodelta)
        df['xmit_time'] = df['xmit_time'].add(timedelta)
    
    # if mode == "dl":
    #     benchmark = list(df["xmit_time"].array)
    # elif mode == "ul":
    #     benchmark = list(df["arr_time"].array)
    # bm1, bm2 = benchmark[0], benchmark[-1]
    # ratio1 = (bm1-delta1[0]).total_seconds() / (delta2[0]-delta1[0]).total_seconds()
    # ratio2 = (bm2-delta1[0]).total_seconds() / (delta2[0]-delta1[0]).total_seconds()
    # _delta1 = interp(delta1[1], delta2[1], ratio1)
    # _delta2 = interp(delta1[1], delta2[1], ratio2)

    # epoch_comp_list = list(np.round(np.linspace(_delta1, _delta2, len(df)), 6))
    # comp_list = pd.to_timedelta(epoch_comp_list, "sec")
    
    # if mode == "dl":
    #     df['arr_time_epoch'] = df['arr_time_epoch'].add(pd.Series(epoch_comp_list))
    #     df['arr_time'] = df['arr_time'].add(pd.Series(comp_list))
    # elif mode == "ul":
    #     df['Timestamp_epoch'] = df['Timestamp_epoch'].add(pd.Series(epoch_comp_list))
    #     df['Timestamp'] = df['Timestamp'].add(pd.Series(comp_list))
    #     df['xmit_time_epoch'] = df['xmit_time_epoch'].add(pd.Series(epoch_comp_list))
    #     df['xmit_time'] = df['xmit_time'].add(pd.Series(comp_list))
    
    return df

def get_latency(df, mode):
    # define latnecy := arrival.time - payload.time
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # payload.time
    df['xmit_time'] = pd.to_datetime(df['xmit_time'])
    df['arr_time'] = pd.to_datetime(df['arr_time'])

    ### calculate latency
    df['latency'] = float('inf')
    df.loc[df['lost'] == False, 'latency'] = (df.loc[df['lost'] == False, 'arr_time'] - df.loc[df['lost'] == False, 'Timestamp']).dt.total_seconds().round(6)
    df['excl'] = df['latency'] > 100e-3
    
    # ### no other way!!!
    # bm = (5 + random.randint(-2000, 2000)*1e-3) * 1e-3
    # latndf = df.loc[df['lost'] == False, 'latency']
    # minlatn = min(latndf)
    # epoch_comp = bm - minlatn
    # comp = pd.to_timedelta(epoch_comp, "sec")
    # if mode == "dl":
    #     df['arr_time_epoch'] = df['arr_time_epoch'] + epoch_comp
    #     df['arr_time'] = df['arr_time'] + comp
    # elif mode == "ul":
    #     df['Timestamp_epoch'] = df['Timestamp_epoch'] - epoch_comp
    #     df['Timestamp'] = df['Timestamp'] - comp
    #     df['xmit_time_epoch'] = df['xmit_time_epoch'] - epoch_comp
    #     df['xmit_time'] = df['xmit_time'] - comp
    # df.loc[df['lost'] == False, 'latency'] = (df.loc[df['lost'] == False, 'arrival.time'] - df.loc[df['lost'] == False, 'Timestamp']).dt.total_seconds().round(6)
    
    return df

def get_statistics(df, fout1, fout2, fout3):
    # output packet record csv
    df = df[["seq", "rpkg", "frame_id", "Timestamp", "Timestamp_epoch", "lost", "excl", "latency", "xmit_time", "xmit_time_epoch", "arr_time", "arr_time_epoch"]]
    print("output >>>", fout1)
    df.to_csv(fout1, index=False)
    
    # loss statistics
    total_packet_sent = len(df)
    total_loss = len(df[df["lost"] == True])
    loss_rate = total_loss / (total_packet_sent + 1e-9) * 100  # ratio (%)
    exp_time = round(df['Timestamp_epoch'].iloc[-1] - df['Timestamp_epoch'].iloc[0], 6) if total_packet_sent else 0
    print("output >>>", fout2)
    with open(fout2, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['total_packet_sent', 'total_packet_loss', 'packet_loss_rate(%)', 'experiment_time(sec)'])
        writer.writerow([total_packet_sent, total_loss, loss_rate, exp_time])
    
    # latency statistics
    latndf = df['latency'][df['lost'] == False]
    total_packet_recv = len(latndf)
    total_excs_latency = len(latndf[latndf > 100e-3])
    excs_latency_rate = total_excs_latency / (total_packet_recv + 1e-9) * 100  # ratio (%)
    
    # calculate jitter
    tmpdf = latndf.diff().abs().dropna()
    jitter = mean(tmpdf)
    
    print("output >>>", fout3)
    with open(fout3, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['total_packet_recv', 'total_excessive_latency', 'excessive_latency_rate(%)', 'experiment_time(sec)'])
        writer.writerow([total_packet_recv, total_excs_latency, excs_latency_rate, exp_time])
    
    print("------------------------------------------")
    print("min latency:         ", min(latndf), "seconds")
    print("max latency:         ", max(latndf), "seconds")
    print("mean latency:        ", round(mean(latndf), 6), "seconds")
    print("stdev latency:       ", round(stdev(latndf), 6), "seconds")
    print("jitter:              ", round(jitter, 6), "seconds")
    print("negative latency:    ", (df['latency'] < 0).sum())
    print("total_packet_sent:   ", total_packet_sent)
    print("total_packet_recv:   ", total_packet_recv)
    print("total_packet_loss:   ", total_loss)
    print("packet_loss_rate(%): ", round(loss_rate, 3), "%")
    print("total_excs_latency:  ", total_excs_latency)
    print("excs_latency_rate(%):", round(excs_latency_rate, 3), "%")
    print("experiment_time(sec):", exp_time, "seconds")
    print("------------------------------------------")
    print()


# ===================== Main Process =====================
if __name__ == "__main__":
    if args.onefile is None:
        
        if args.dates is not None:
            dates = sorted(args.dates)
        else:
            raise TypeError("Please specify the date you want to process.")
        
        print('\n================================ Start Processing ================================')
        
        metadatas = metadata_loader(dates)
        # pop_error_message()
        # pop_error_message(signal='Parsing packet info into brief format', stdout=True)
        
        # for metadata in metadatas:
        #     try:
        #         print(metadata)
        #         print('--------------------------------------------------------')
        #         middle_dir = os.path.join(metadata[0], 'middle')
        #         filenames = [s for s in os.listdir(middle_dir) if s.endswith('.csv')]
                
        #         # **********************
        #         t = TicToc(); t.tic()
        #         print('Server | Downlink')
        #         try: filename = [s for s in filenames if s.startswith(('server_pcap_BL', 'server_pcap_DL'))][0]
        #         except: filename = None
        #         if filename is not None:
        #             fin = os.path.join(middle_dir, filename)
        #             fout = os.path.join(middle_dir, "udp_dnlk_server_pkt_brief.csv")
        #             print(f">>>>> {fin} -> {fout}")
        #             parse_pkt_info(fin, fout, "server", "dl", "udp", locate=str(metadata)+'\nServer | Downlink')
        #         t.toc(); print()
                
        #         t = TicToc(); t.tic()
        #         print('Client | Downlink')
        #         try: filename = [s for s in filenames if s.startswith(('client_pcap_BL', 'client_pcap_DL'))][0]
        #         except: filename = None
        #         if filename is not None:
        #             fin = os.path.join(middle_dir, filename)
        #             fout = os.path.join(middle_dir, "udp_dnlk_client_pkt_brief.csv")
        #             print(f">>>>> {fin} -> {fout}")
        #             parse_pkt_info(fin, fout, "client", "dl", "udp", locate=str(metadata)+'\nClient | Downlink')
        #         t.toc(); print()
                
        #         t = TicToc(); t.tic()
        #         print('Server | Uplink')
        #         try: filename = [s for s in filenames if s.startswith(('server_pcap_BL', 'server_pcap_UL'))][0]
        #         except: filename = None
        #         if filename is not None:
        #             fin = os.path.join(middle_dir, filename)
        #             fout = os.path.join(middle_dir, "udp_uplk_server_pkt_brief.csv")
        #             print(f">>>>> {fin} -> {fout}")
        #             parse_pkt_info(fin, fout, "server", "ul", "udp", locate=str(metadata)+'\nServer | Uplink')
        #         t.toc(); print()
                
        #         t = TicToc(); t.tic()
        #         print('Client | Uplink')
        #         try: filename = [s for s in filenames if s.startswith(('client_pcap_BL', 'client_pcap_UL'))][0]
        #         except: filename = None
        #         if filename is not None:
        #             fin = os.path.join(middle_dir, filename)
        #             fout = os.path.join(middle_dir, "udp_uplk_client_pkt_brief.csv")
        #             print(f">>>>> {fin} -> {fout}")
        #             parse_pkt_info(fin, fout, "client", "ul", "udp", locate=str(metadata)+'\nClient | Uplink')
        #         t.toc(); print()
        #         # **********************
        #         print()
                    
        #     except Exception as e:
        #         pop_error_message(e, locate=metadata, raise_flag=True)
                
        # pop_error_message(signal='Finish parsing packet info', stdout=True)
        pop_error_message(signal='Aligning data on the same device within one trip', stdout=True)
        
        for metadata in metadatas:
            try:
                print(metadata)
                print('--------------------------------------------------------')
                middle_dir = os.path.join(metadata[0], 'middle')
                
                csvfiles = ["udp_dnlk_server_pkt_brief.csv", "udp_dnlk_client_pkt_brief.csv", "udp_uplk_server_pkt_brief.csv", "udp_uplk_client_pkt_brief.csv"]
                csvfiles = [os.path.join(middle_dir, s) for s in csvfiles]
                csvfiles = [s for s in csvfiles if os.path.isfile(s)]
                
                st_t = []
                ed_t = []
                for file in csvfiles:
                    print(file)
                    df = pd.read_csv(file)
                    df = str_to_datetime_batch(df, parse_dates=['frame_time'])
                    st_t.append(df.iloc[0]['frame_time'] - pd.Timedelta(seconds=5))
                    ed_t.append(df.iloc[-1]['frame_time'] + pd.Timedelta(seconds=5))
                    del df

                # st_t = max(st_t)
                # ed_t = min(ed_t)
                # for file in csvfiles:
                #     df = pd.read_csv(file)
                #     df = str_to_datetime_batch(df, parse_dates=['frame_time'])
                #     df = df[(df['frame_time'] > st_t) & (df['frame_time'] < ed_t)]
                #     df.to_csv(file, index=False)
                #     del df
                    
                print()
                
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish aligning data', stdout=True)
        
    else:
        print(args.onefile)




# dl_txdf = pd.read_csv("/home/wmnlab/D/database/2023-02-04#2/_Bandlock_Udp_all_RM500Q/qc01/#01/middle/udp_dnlk_server_pkt_brief.csv")
# dl_rxdf = pd.read_csv("/home/wmnlab/D/database/2023-02-04#2/_Bandlock_Udp_all_RM500Q/qc01/#01/middle/udp_dnlk_client_pkt_brief.csv")
# ul_txdf = pd.read_csv("/home/wmnlab/D/database/2023-02-04#2/_Bandlock_Udp_all_RM500Q/qc01/#01/middle/udp_uplk_client_pkt_brief.csv")
# ul_rxdf = pd.read_csv("/home/wmnlab/D/database/2023-02-04#2/_Bandlock_Udp_all_RM500Q/qc01/#01/middle/udp_uplk_server_pkt_brief.csv")

# dl_txseq = list(dl_txdf["sequence.number"].array)
# dl_rxseq = list(dl_rxdf["sequence.number"].array)
# dlst = max(dl_txseq[0], dl_rxseq[0])
# dlet = min(dl_txseq[-1], dl_rxseq[-1])
# # print(dlst, dlet)

# ul_txseq = list(ul_txdf["sequence.number"].array)
# ul_rxseq = list(ul_rxdf["sequence.number"].array)
# ulst = max(ul_txseq[0], ul_rxseq[0])
# ulet = min(ul_txseq[-1], ul_rxseq[-1])
# # print(ulst, ulet)

# st = max(dlst, ulst)
# et = min(dlet, ulet)
# # print("----------------")
# st += PKT_RATE * 5  # 開頭切5秒
# et -= PKT_RATE * 5  # 結尾切5秒
# # print(st, et)

# dl_txdf = dl_txdf[(dl_txdf["sequence.number"] >= st) & (dl_txdf["sequence.number"] <= et)]
# dl_rxdf = dl_rxdf[(dl_rxdf["sequence.number"] >= st) & (dl_rxdf["sequence.number"] <= et)]
# ul_txdf = ul_txdf[(ul_txdf["sequence.number"] >= st) & (ul_txdf["sequence.number"] <= et)]
# ul_rxdf = ul_rxdf[(ul_rxdf["sequence.number"] >= st) & (ul_rxdf["sequence.number"] <= et)]

# dl_txdf.reset_index(drop=True, inplace=True)
# dl_rxdf.reset_index(drop=True, inplace=True)
# ul_txdf.reset_index(drop=True, inplace=True)
# ul_rxdf.reset_index(drop=True, inplace=True)

# with open("/home/wmnlab/D/database/2023-02-04#2/tsync/qc01/delta.txt", encoding="utf-8") as f:
#     lines = f.readlines()
#     timerec1 = pd.to_datetime(lines[0])
#     epoch_delta1 = float(lines[1])
#     timedelta1 = pd.Timedelta(seconds=epoch_delta1).round('us')
# with open("/home/wmnlab/D/database/2023-02-04#2/tsync/qc01/delta1.txt", encoding="utf-8") as f:
#     lines = f.readlines()
#     timerec2 = pd.to_datetime(lines[0])
#     epoch_delta2 = float(lines[1])
#     timedelta2 = pd.Timedelta(seconds=epoch_delta2).round('us')

# # print(timerec1)
# # print(epoch_delta1)
# # print(timedelta1)
# # print(timerec2)
# # print(epoch_delta2)
# # print(timedelta2)
# delta1 = (timerec1, epoch_delta1, timedelta1)
# delta2 = (timerec2, epoch_delta2, timedelta2)



if __name__ == "__main__":
    t = TicToc()  # create instance of class
    t.tic()  # Start timer
    # --------------------- (3) decode a batch of files (User Settings) ---------------------
    # err_handles = []
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
            
    # --------------------- Phase 2: Parse packet loss & latency --------------------- 
    ### Read files
    for date, json_file in zip(dates, json_files):
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
                    
                    ### Timedelta
                    # timedelta, epoch_delta = calc_delta(txdl_df, rxdl_df, txul_df, rxul_df)
                    # with open(os.path.join(database, date, "tsync", dev, "delta.txt"), encoding="utf-8") as f:
                    #     lines = f.readlines()
                    #     timerec1 = pd.to_datetime(lines[0])
                    #     epoch_delta1 = float(lines[1])
                    #     timedelta1 = pd.Timedelta(seconds=epoch_delta1).round('us')
                    # with open(os.path.join(database, date, "tsync", dev, "delta1.txt"), encoding="utf-8") as f:
                    #     lines = f.readlines()
                    #     timerec2 = pd.to_datetime(lines[0])
                    #     epoch_delta2 = float(lines[1])
                    #     timedelta2 = pd.Timedelta(seconds=epoch_delta2).round('us')
                    # delta1 = (timerec1, epoch_delta1, timedelta1)
                    # delta2 = (timerec2, epoch_delta2, timedelta2)
                    
                    # print(timerec1)
                    # print(epoch_delta1)
                    # print(timedelta1)
                    # print(timerec2)
                    # print(epoch_delta2)
                    # print(timedelta2)

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
