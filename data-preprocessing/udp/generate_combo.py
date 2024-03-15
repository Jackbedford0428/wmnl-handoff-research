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
import math
import random
import itertools

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
# database = "/Users/jackbedford/Desktop/MOXA/Code/data/"
dates = [
    "2023-04-10",
]
json_files = [
    "time_sync_lpt3.json",
]
json_files = [os.path.join(database, date, json_file) for date, json_file in zip(dates, json_files)]

exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    # "_Bandlock_Udp_B3_B7_B8_RM500Q": (6, ["#{:02d}".format(i+1) for i in range(6)]),
    # "_Bandlock_Udp_All_RM500Q": (4, ["#{:02d}".format(i+1) for i in range(4)]),
    # "_Bandlock_Udp_B3_B7_B8_RM500Q": (1, ["#{:02d}".format(i+1) for i in range(1, 2)]),
    # "_Bandlock_Udp_All_RM500Q": (1, ["#{:02d}".format(i+1) for i in range(1, 2)]),
    # "_Bandlock_Udp_B1_B3_B7_B8_RM500Q": (6, ["#{:02d}".format(i+1) for i in range(6)]),
    # "_Bandlock_Udp_All_LTE_B1B3_B1B8_RM500Q": (4, ["#{:02d}".format(i+1) for i in range(4)]),
    "_Experiment1": (2, ["#{:02d}".format(i+1) for i in range(2)]),
    "_Experiment2": (2, ["#{:02d}".format(i+1) for i in range(2)]),
    "_Experiment3": (2, ["#{:02d}".format(i+1) for i in range(2)]),
}
_devices = [
    [
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
        "qc01",
        "qc02",
        "qc03",
    ],
    [
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
        "qc01",
        "qc02",
        "qc03",
    ],
    [
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
        "qc01",
        "qc02",
        "qc03",
    ],
]
_schemes = [
    ["All@0", "All@1", "All@2", "All@3"],
    ["B1", "B3", "B7", "B8"],
    ["LTE", "All", "B7B8", "B7"],
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
                xs = list(itertools.combinations(range(len(schemes)), 2))
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