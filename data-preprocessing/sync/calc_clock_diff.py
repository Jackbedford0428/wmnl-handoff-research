#!/usr/bin/python3
### Filename: 

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

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
# date = "2023-01-12"
dates = [
         "2023-02-04", 
         "2023-02-04#1",
         "2023-02-04#2",
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
    # "qc00",
    "qc01",
    "qc02",
    "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    "tsync": (2, []),
    # "tsync": (1, ["1"]),
    # "tsync": (1, None),
    # "tsync": (2, []),
    # "_tsync": (2, []),
    # "_Bandlock_Udp": (4, ["#01", "#02", "#03", "#04"]),
    # "_Bandlock_Udp": (4, ["#03", "#04", "#05", "#06"]),
    # "_Bandlock_Udp": (4, []),
    # "_Bandlock_Udp": (6, []),
    # "_Bandlock_Udp_B1_B3":  (4, []),
    # "_Bandlock_Udp_B3_B28": (4, []),
    # "_Bandlock_Udp_B28_B1": (4, []),
    # "_Modem_Phone_Comparative_Exeriments": (6, []),
}
# *****************************************************************************

# **************************** Auxiliary Functions ****************************
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

def truncate(txdf, rxdf):
    """
    Truncate the rows of lost packets.
    """
    tx_arr = list(zip(txdf['sequence.number'].array, txdf['Timestamp'].array, txdf['Timestamp_epoch'].array))
    rx_arr = list(zip(rxdf['sequence.number'].array, rxdf['Timestamp'].array, rxdf['Timestamp_epoch'].array))
    _tx_arr = []
    i, j = 0, 0
    N = len(rx_arr)
    # print(tx_arr[i][0], rx_arr[j][0])
    for i in range(len(tx_arr)):
        if j == N:
            break
        # print(tx_arr[i][0], rx_arr[j][0])
        while j != N and tx_arr[i][0] > rx_arr[j][0]:
            j += 1
        # if tx_arr[i][0] < rx_arr[j][0]:
        #     pass  # i += 1 by for-loop
        if j != N and tx_arr[i][0] == rx_arr[j][0]:
            _tx_arr.append(tx_arr[i])
            j += 1
    ### Since the transmission is stopped by client,
    ### the ending sequence of Downlink-Tx/Uplink-Rx (Server-Side) is larger than Downlink-Rx/Uplink-Tx (Client-Side).
    M = min(len(_tx_arr), len(rx_arr))
    return _tx_arr[:M], rx_arr[:M]


def del_outliers(num_list):
    """
    Remove outliers and return the remaining list.
    """
    if not len(num_list):
        return [], []
    
    _num_list = sorted(num_list)
    upper_q = np.percentile(_num_list, 75)
    lower_q = np.percentile(_num_list, 25)
    iqr = (upper_q - lower_q) * 1.5
    q_set = (lower_q - iqr, upper_q + iqr)
    # print(q_set)

    result_list = []
    ret_list = []
    # for i, x in enumerate(_num_list):
    #     if x >= q_set[0] and x <= q_set[1]:
    #         result_list.append(x)
    #         ret_list.append(i)
    for i, x in enumerate(num_list):
        if x >= q_set[0] and x <= q_set[1]:
            result_list.append(x)
            ret_list.append(i)
    
    return result_list, ret_list

def calc_delta(txdl_df, rxdl_df, txul_df, rxul_df):
    """
    Calculate the time delta between server and client.
        If the client is behind server, delta > 0
        If the client is ahead of server, delta < 0
        server clock := client clock + delta
        
    Returns:
        delta (datetime.timedelta)
        delta (float)
    """
    ### Since the transmission is launched by client, the starting time of Uplink is ahead of Downlink.
    # seq_diff = round(500e-3 * PKT_RATE)
    # txul_df = txul_df[txul_df['sequence.number'] > seq_diff].reset_index(drop=True)
    # rxul_df = rxul_df[rxul_df['sequence.number'] > seq_diff].reset_index(drop=True)
    
    txdl_arr, rxdl_arr = truncate(txdl_df, rxdl_df)
    # print(txdl_arr[:10], rxdl_arr[:10])
    txul_arr, rxul_arr = truncate(txul_df, rxul_df)
    # print(txul_arr[:10], rxul_arr[:10])

    M = min(len(txdl_arr), len(txul_arr))
    txdl_arr, rxdl_arr, txul_arr, rxul_arr = txdl_arr[:M], rxdl_arr[:M], txul_arr[:M], rxul_arr[:M]
    # print(txdl_arr, rxdl_arr)
    # print(txul_arr, rxul_arr)

    epoch_delta_list = []
    for ts1, ts2, ts3, ts4 in zip(txdl_arr, rxdl_arr, txul_arr, rxul_arr):
        epoch_latency_dl = ts2[2] - ts1[2]
        epoch_latency_ul = ts4[2] - ts3[2]
        epoch_delta_list.append((epoch_latency_ul - epoch_latency_dl) / 2)
    
    # print(type(pd.Timedelta(microseconds=10)))  # <class 'pandas._libs.tslibs.timedeltas.Timedelta'> : datetime64[ns]
    # print(type(dt.timedelta(microseconds=10)))  # <class 'datetime.timedelta'> : microseconds

    if not len(epoch_delta_list):
        epoch_delta = [0, 0]
        timedelta = [pd.Timedelta(seconds=0), pd.Timedelta(seconds=0)]
        return timedelta, epoch_delta
    
    epoch_delta = round(mean(epoch_delta_list), 6)
    timedelta = pd.Timedelta(seconds=epoch_delta).round('us')
    
    # # print("hello")
    # epoch_delta_list, _ = del_outliers(epoch_delta_list)

    # kde = stats.gaussian_kde(epoch_delta_list)
    # xx = np.linspace(min(epoch_delta_list), max(epoch_delta_list), 1000)
    # yy = kde(xx)
    # max_index = list(signal.argrelextrema(yy, np.greater)[0])
    # stdev_1 = P.closed(mean(epoch_delta_list) - stdev(epoch_delta_list), mean(epoch_delta_list) + stdev(epoch_delta_list))
    # _stdev =  stdev(epoch_delta_list)
    
    # epoch_deltas = [xx[i] for i in max_index if xx[i] in stdev_1] + [mean(epoch_delta_list), median(epoch_delta_list), mode(epoch_delta_list)]
    # # epoch_delta = [round(max(epoch_deltas), 6), round(min(epoch_deltas), 6)]
    # # epoch_delta = [round(1.05 * max(epoch_deltas), 6), round(0.95 * min(epoch_deltas), 6)]
    # # epoch_delta = [round(1.03 * max(epoch_deltas), 6), round(0.97 * min(epoch_deltas), 6)]
    # epoch_delta = [round(max(epoch_deltas) + 1.15 * _stdev, 6), round(min(epoch_deltas) - 1.15 * _stdev, 6)]
    # timedelta = [pd.Timedelta(seconds=secs).round('us') for secs in epoch_delta]
    
    timestamps = list(map(list, zip(*txdl_arr)))
    timestamp = to_utc8(round(median(timestamps[2]), 6))
    
    print(timestamp)
    print(f"{epoch_delta} seconds")

    ### Use mean, median, or mode: https://www.scribbr.com/statistics/central-tendency/
    ### Pandas.Timedelta: https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
    ### Pandas.Timedelta.round: https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.round.html
    return timedelta, epoch_delta, timestamp
# *****************************************************************************

# ****************************** Utils Functions ******************************
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
# *****************************************************************************


if __name__ == "__main__":
    def main():
        txdl_df = pd.read_csv(os.path.join(source_dir, "udp_dnlk_server_pkt_brief.csv"))
        rxdl_df = pd.read_csv(os.path.join(source_dir, "udp_dnlk_client_pkt_brief.csv"))
        txul_df = pd.read_csv(os.path.join(source_dir, "udp_uplk_client_pkt_brief.csv"))
        rxul_df = pd.read_csv(os.path.join(source_dir, "udp_uplk_server_pkt_brief.csv"))
        txdl_df['Timestamp'] = pd.to_datetime(txdl_df['Timestamp'])  # transmitted time from server
        rxdl_df['Timestamp'] = pd.to_datetime(rxdl_df['Timestamp'])  # arrival time to client
        txul_df['Timestamp'] = pd.to_datetime(txul_df['Timestamp'])  # transmitted time from client
        rxul_df['Timestamp'] = pd.to_datetime(rxul_df['Timestamp'])  # arrival time to server
        txdl_df['payload.time'] = pd.to_datetime(txdl_df['payload.time'])
        rxdl_df['payload.time'] = pd.to_datetime(rxdl_df['payload.time'])
        txul_df['payload.time'] = pd.to_datetime(txul_df['payload.time'])
        rxul_df['payload.time'] = pd.to_datetime(rxul_df['payload.time'])
        _, epoch_delta, timestamp = calc_delta(txdl_df, rxdl_df, txul_df, rxul_df)
        # print(timedelta, epoch_delta)
        filename = "delta.txt"
        if trace in ['1', '2', '3']:
            filename = "delta{}.txt".format(trace)
        with open(os.path.join(target_dir, filename), 'w') as f:
            f.write(str(timestamp)+'\n')
            f.write(str(epoch_delta)+'\n')
        print()
    
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
    err_handles = []
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
                    if expr == "tsync":
                        source_dir = os.path.join(database, date, expr, dev, trace)
                        target_dir = os.path.join(database, date, expr, dev)
                    makedir(target_dir)
                    filenames = os.listdir(source_dir)
                    main()
    t.toc()  # Time elapsed since t.tic()
    # *****************************************************************************