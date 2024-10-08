import os
import sys
import argparse
import csv
import json
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
import math
import random

__all__ = [
    'get_loss',
    'consolidate',
    'compensate',
    'get_latency',
    'get_statistics'
]

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
database = "/Users/jackbedford/Desktop/MOXA/Code/data/"
dates = [
    "2023-06-14"
]
json_files = ["time_sync_lpt3.json"]
json_files = [os.path.join(database, date, json_file) for date, json_file in zip(dates, json_files)]
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
    # "qc02",
    "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (16, []),
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (4, ["#{:02d}".format(i+1) for i in range(12, 16)]),
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (6, ["#{:02d}".format(i+1) for i in range(6)]),
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (3, ["#{:02d}".format(i+1) for i in range(3, 6)]),
    "Modem_Action_Test": (2, ["#{:02d}".format(i+1) for i in range(2)]),
}

class Payload:
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

DATA_RATE = 1000e3  # bits-per-second
PKT_RATE = DATA_RATE / Payload.LENGTH / 8  # packets-per-second
print("packet_rate (pps):", PKT_RATE, "\n")
# *****************************************************************************

# --------------------- Util Functions ---------------------
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

def error_handling(err_handle):
    """
    Print the error messages during the process.
    
    Args:
        err_handle (str-tuple): (input_filename, output_filename, error_messages : traceback.format_exc())
    Returns:
        (bool): check if the error_messages occurs, i.e., whether it is None.
    """
    if err_handle[2]:
        print()
        print("**************************************************")
        print("File decoding from '{}' into '{}' was interrupted.".format(err_handle[0], err_handle[1]))
        print()
        print(err_handle[2])
        return True
    return False

def to_utc8(ts):
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
        ts_datetime = dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f')
    except:
        ts_datetime = dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
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
        ts_string = dt.datetime.strftime(ts, '%Y-%m-%d %H:%M:%S.%f')
    except:
        ts_string = dt.datetime.strftime(ts, '%Y-%m-%d %H:%M:%S')
    return ts_string

def get_loss(rxdf):
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
                loss_time = [round(item[0]), to_utc8(item[1]), item[1]]  # (expected) arrival timestamp
                loss_timestamp_list.append(loss_time)
        # Update information
        timestamp_store = timestamp
        _eseq = timestamp[0] + 1
    
    N = len(loss_timestamp_list)
    loss_timestamps = list(map(list, zip(*loss_timestamp_list)))
    df = pd.DataFrame()
    if len(loss_timestamps) > 0:
        df = pd.DataFrame.from_dict(
            {
                "seq": loss_timestamps[0],
                "rpkg": [None] * N,
                "frame_id": [None] * N,
                "Timestamp": [None] * N,  # payload.time
                "Timestamp_epoch": [None] * N,  # payload.time_epoch
                "lost": [True] * N,
                "latency": [float('inf')] * N,
                "xmit_time": [None] * N,
                "xmit_time_epoch": [None] * N,
                "arr_time": loss_timestamps[1],
                "arr_time_epoch": loss_timestamps[2],
            }
        )
    return df

def consolidate(rxdf):
    ### add transmitted timestamp
    df = rxdf.rename(
        columns={
            "frame_time": "arr_time",
            "frame_time_epoch": "arr_time_epoch",
            "pyl_time": "Timestamp",
            "pyl_time_epoch": "Timestamp_epoch",
        }
    )
    df["xmit_time"] = None
    df["xmit_time_epoch"] = None
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


# json_file = "/home/wmnlab/D/database/2023-03-08/time_sync_lpt3.json"
# json_object = {}
# if os.path.isfile(json_file):
#     with open(json_file, 'r') as f:
#         json_object = json.load(f)
# delta = pd.DataFrame.from_dict(json_object, orient='index', columns=['delta']).reset_index(names='Timestamp')
# delta['Timestamp'] = pd.to_datetime(delta['Timestamp'])
# delta['timedelta'] = pd.to_timedelta(delta['delta'], unit='seconds')

# ### Uplink
# rxdf = pd.read_csv("/home/wmnlab/D/database/2023-03-08/_Bandlock_Udp_B1_B3_B7_B8_RM500Q/qc00/#01/middle/udp_uplk_server_pkt_brief.csv")
# rxseq = list(rxdf['seq'].array)

# st = rxseq[0] + PKT_RATE * 5  # 開頭切5秒
# et = rxseq[-1] - PKT_RATE * 5  # 結尾切5秒
# rxdf = rxdf[(rxdf["seq"] >= st) & (rxdf["seq"] <= et)].copy().reset_index(drop=True)

# # fout1 = os.path.join(target_dir1, "udp_uplk_loss_latency.csv")
# # fout2 = os.path.join(target_dir2, "udp_uplk_loss_statistics.csv")
# # fout3 = os.path.join(target_dir2, "udp_uplk_excl_statistics.csv")

# losdf = get_loss(rxdf.copy())
# latdf = consolidate(rxdf.copy())
# df = pd.concat([losdf, latdf], axis=0)
# df = df.sort_values(by=["seq"]).reset_index(drop=True)
# df = compensate(df.copy(), "ul", delta.copy())
# df = get_latency(df.copy(), "ul")
# print(df)
# # get_statistics(df.copy(), fout1, fout2, fout3)


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
                    rxdf = pd.read_csv(os.path.join(source_dir, "udp_dnlk_client_pkt_brief.csv"))
                    rxseq = list(rxdf['seq'].array)
                    
                    st = rxseq[0] + PKT_RATE * 5  # 開頭切5秒
                    et = rxseq[-1] - PKT_RATE * 5  # 結尾切5秒
                    rxdf = rxdf[(rxdf["seq"] >= st) & (rxdf["seq"] <= et)].copy().reset_index(drop=True)
                    
                    fout1 = os.path.join(target_dir1, "udp_dnlk_loss_latency.csv")
                    fout2 = os.path.join(target_dir2, "udp_dnlk_loss_statistics.csv")
                    fout3 = os.path.join(target_dir2, "udp_dnlk_excl_statistics.csv")
                    
                    losdf = get_loss(rxdf.copy())
                    latdf = consolidate(rxdf.copy())
                    df = pd.concat([latdf, losdf], axis=0)
                    df = df.sort_values(by=["seq"]).reset_index(drop=True)
                    df = compensate(df.copy(), "dl", delta.copy())
                    df = get_latency(df.copy(), "dl")
                    get_statistics(df.copy(), fout1, fout2, fout3)
                    
                    # ### Uplink
                    # rxdf = pd.read_csv(os.path.join(source_dir, "udp_uplk_server_pkt_brief.csv"))
                    # rxseq = list(rxdf['seq'].array)

                    # st = rxseq[0] + PKT_RATE * 5  # 開頭切5秒
                    # et = rxseq[-1] - PKT_RATE * 5  # 結尾切5秒
                    # rxdf = rxdf[(rxdf["seq"] >= st) & (rxdf["seq"] <= et)].copy().reset_index(drop=True)
                    
                    # fout1 = os.path.join(target_dir1, "udp_uplk_loss_latency.csv")
                    # fout2 = os.path.join(target_dir2, "udp_uplk_loss_statistics.csv")
                    # fout3 = os.path.join(target_dir2, "udp_uplk_excl_statistics.csv")
                    
                    # losdf = get_loss(rxdf.copy())
                    # latdf = consolidate(rxdf.copy())
                    # df = pd.concat([latdf, losdf], axis=0)
                    # df = df.sort_values(by=["seq"]).reset_index(drop=True)
                    # df = compensate(df.copy(), "ul", delta.copy())
                    # df = get_latency(df.copy(), "ul")
                    # get_statistics(df.copy(), fout1, fout2, fout3)
                    
                    t1.toc()
    t.toc()  # Time elapsed since t.tic()
