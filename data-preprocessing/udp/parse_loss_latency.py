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
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *

__all__ = [
    "get_loss_v2",
    "get_latency",
    "get_statistics",
]


THRESHOLD = 100e-3  # (seconds)


# ===================== Utils =====================
HASH_SEED = time.time()
LOGFILE = os.path.basename(__file__).replace('.py', '') + '_' + query_datetime() + '-' + generate_hex_string(HASH_SEED, 5) + '.log'
RECORDS = os.path.basename(__file__).replace('.py', '') + '_' + query_datetime() + '-' + generate_hex_string(HASH_SEED, 5) + '_records.log'

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
def get_loss(txdf, rxdf):
    
    def consolidate(txdf, rxdf, loss_df):
        rxseq = rxdf['seq'].array
        txseq = txdf['seq'].array
        txts = txdf['frame_time'].array
        txts_epoch = txdf['frame_time_epoch'].array
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
        # add transmitted timestamp
        df = rxdf.join(pd.DataFrame({'tx_time' : rx_txts_arr, 'tx_time_epoch' : rx_txts_epoch_arr}))
        # drop lost rows in df (rxdf)
        df = df.dropna(how='any', subset=['tx_time', 'tx_time_epoch'], axis=0)
        df = df.rename(columns={
                "frame_time": "rx_time",
                "frame_time_epoch": "rx_time_epoch",
                "pyl_time": "Timestamp",
                "pyl_time_epoch": "Timestamp_epoch",
            })
        df["lost"] = False
        df = df[["seq", "rpkg", "frame_id", "Timestamp", "Timestamp_epoch", "lost", "tx_time", "tx_time_epoch", "rx_time", "rx_time_epoch"]]
        # add lost rows back to df
        df = pd.concat([df, loss_df], axis=0)
        df = df.sort_values(by=["seq"]).reset_index(drop=True)
        return df

    timestamp_list = list(map(list, zip(rxdf['seq'].astype(int).array, rxdf['frame_time_epoch'].astype(float).array)))
    timestamp_store = timestamp_list[0]
    loss_timestamp_list = []
    count = 0  # to count the total number of packet losses
    next_eseq = timestamp_list[0][0]  # next expected sequence number: ignore the first-N lost packets if existing.
    for i in tqdm(range(len(rxdf))):
        timestamp = timestamp_list[i]
        if timestamp[0] == next_eseq:
            # received packet's sequence number is as expected.
            pass
        else:
            # packet losses occur
            # 可處理連續掉 N 個封包的狀況
            # timestamp_store: 前一刻收到的封包
            # timestamp: 此時此刻收到的封包
            # next_eseq 為預期收到的封包 sequence number (前一刻收到的 seq number + 1)
            # rxdf.loc[i, 'sequence.number'] 為此時此刻收到的封包 seq
            # rxdf.loc[i, 'sequence.number']-pointer+2 == 遺漏的封包數+2 (頭+尾)，因此要掐頭去尾才是實際遺漏的封包
            n = timestamp[0] - next_eseq + 2
            loss_linspace = np.linspace(timestamp_store, timestamp, n)
            loss_linspace = loss_linspace[1:-1]  # 掐頭去尾
            for item in loss_linspace:
                count += 1
                loss_time = [round(item[0]), epoch_to_datetime(item[1]), item[1]]  # (expected) received timestamp
                loss_timestamp_list.append(loss_time)
        # update information
        timestamp_store = timestamp
        next_eseq = timestamp[0] + 1
    
    # add payload & transmitted timestamp
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
    # 因為是由 client 端主動開始和結束實驗，且實驗順序為: 開 Tcpdump -> 開 iperf -> 關 Tcpdump -> 關 iperf
    # 因此 uplink Tx 的 MAX_SEQ 會比 Rx 小，Downlink Tx 的 MAX_SEQ 會比 Rx 大。
    loss_timestamp_list = [item for item in loss_timestamp_list if len(item) == 7]
    
    N = len(loss_timestamp_list)
    loss_timestamps = list(map(list, zip(*loss_timestamp_list)))
    df = pd.DataFrame()
    if len(loss_timestamps) > 0:
        df = pd.DataFrame.from_dict({
                "seq": loss_timestamps[0],
                "rpkg": [None] * N,
                "frame_id": [None] * N,
                "Timestamp": loss_timestamps[3],  # payload.time
                "Timestamp_epoch": loss_timestamps[4],  # payload.time_epoch
                "lost": [True] * N,
                "tx_time": loss_timestamps[5],
                "tx_time_epoch": loss_timestamps[6],
                "rx_time": loss_timestamps[1],
                "rx_time_epoch": loss_timestamps[2],
            })
    
    df = consolidate(txdf, rxdf, df)
    return df


def get_loss_v2(txdf, rxdf):
    df = txdf.merge(rxdf, on=['seq','pyl_time','pyl_time_epoch'], how='left', suffixes=('_tx', '_rx'), indicator=True, sort=False)
    df['lost'] = df['_merge'].ne('both')
    df = df.rename(columns={
                "rpkg_rx": "rpkg",
                "frame_id_rx": "frame_id",
                "pyl_time": "Timestamp",
                "pyl_time_epoch": "Timestamp_epoch",
                "frame_time_rx": "rx_time",
                "frame_time_epoch_rx": "rx_time_epoch",
                "frame_time_tx": "tx_time",
                "frame_time_epoch_tx": "tx_time_epoch",
        })
    df[['rpkg', 'frame_id']] = df[['rpkg', 'frame_id']].astype('Int64')
    df = df[["seq", "rpkg", "frame_id", "Timestamp", "Timestamp_epoch", "lost", "tx_time", "tx_time_epoch", "rx_time", "rx_time_epoch"]]
    df = df.sort_values(by=["seq"]).reset_index(drop=True)
    return df


def get_latency(df, direction, sync_mapping, thr=100e-3):
    # define latnecy := arrival.time(rx_time) - payload.time(Timestamp)
    
    def compensate(df, direction, sync_mapping):
        if sync_mapping is None:
            return df
        
        sync_mapping = {str_to_datetime(key): value for key, value in sync_mapping.items()}
        
        def find_nearest_key(timestamp):
            nearest_timestamp = min(sync_mapping.keys(), key=lambda x: abs(x - timestamp))
            return nearest_timestamp, sync_mapping[nearest_timestamp]
        
        st_t, _ = find_nearest_key(df.iloc[0]['Timestamp'])
        ed_t, _ = find_nearest_key(df.iloc[-1]['Timestamp'])
        
        sync_mapping_df = pd.DataFrame(sync_mapping.values(), index=sync_mapping.keys(), columns=['delta']).reset_index(names='Timestamp')
        sync_mapping_df = sync_mapping_df[(sync_mapping_df['Timestamp'] >= st_t) & (sync_mapping_df['Timestamp'] <= ed_t)]
        # sync_mapping = sync_mapping_df.set_index('Timestamp')['delta'].to_dict()
        
        sync_mapping_df['prev_Timestamp'] = sync_mapping_df['Timestamp'].shift(1)
        sync_mapping_df['next_Timestamp'] = sync_mapping_df['Timestamp'].shift(-1)
        sync_mapping_df['lower_bound'] = pd.Timestamp.min
        sync_mapping_df.loc[sync_mapping_df['prev_Timestamp'].notna(), 'lower_bound'] = (sync_mapping_df['prev_Timestamp'] + (sync_mapping_df['Timestamp'] - sync_mapping_df['prev_Timestamp']) / 2).dt.round(freq='us')
        sync_mapping_df['upper_bound'] = pd.Timestamp.max
        sync_mapping_df.loc[sync_mapping_df['next_Timestamp'].notna(), 'upper_bound'] = (sync_mapping_df['Timestamp'] + (sync_mapping_df['next_Timestamp'] - sync_mapping_df['Timestamp']) / 2).dt.round(freq='us')
        
        # print(sync_mapping_df)
        
        if direction == 'dl':
            target_columns = ['rx_time']
        else:
            target_columns = ['tx_time', 'Timestamp']
        dev_timestamp = target_columns[0]
            
        for i, row in sync_mapping_df.iterrows():
            lower, upper = row['lower_bound'], row['upper_bound']
            epoch_delta = round(row['delta'], 6)
            delta = pd.Timedelta(seconds=epoch_delta)
            for col in target_columns:
                df.loc[(df[dev_timestamp] >= lower) & (df[dev_timestamp] < upper), f'{col}_epoch'] = df[f'{col}_epoch'].add(epoch_delta).round(6)
                df.loc[(df[dev_timestamp] >= lower) & (df[dev_timestamp] < upper), col] = df[col].add(delta).dt.round(freq='us')
            df.loc[(df[dev_timestamp] >= lower) & (df[dev_timestamp] < upper), 'delta'] = epoch_delta
        
        return df
    
    df = compensate(df, direction=direction, sync_mapping=sync_mapping)
    df['latency'] = float('inf')
    df.loc[~df['lost'], 'latency'] = (df['rx_time'] - df['Timestamp']).dt.total_seconds().round(6)
    df['excl'] = df['latency'] > thr
    
    df = df[["seq", "rpkg", "frame_id", "Timestamp", "Timestamp_epoch", "lost", "excl", "latency", "tx_time", "tx_time_epoch", "rx_time", "rx_time_epoch", "delta"]]
    return df


def get_statistics(df, fout, thr=100e-3, dump_log=RECORDS):
    rows = []
    rows.append(['experimental duration', round(df['Timestamp_epoch'].iloc[-1] - df['Timestamp_epoch'].iloc[0], 6) if len(df) else 0.0, 'sec'])
    rows.append(['total packets sent', len(df), ''])
    rows.append(['total packets recv', len(df) - sum(df['lost']), ''])
    rows.append(['total packets lost', sum(df['lost']), ''])
    rows.append(['packet loss rate (%)', df['lost'].mean() * 100, '%'])
    rows.append([f'total excessive latency (>{int(thr * 1000)}ms)', (~df['lost'] & df['excl']).sum(), ''])
    rows.append(['excessive latency rate (%)', (~df['lost'] & df['excl']).mean() * 100, '%'])
    rows.append(['negative latency', (df['latency'] < 0).sum(), ''])
    rows.append(['negative latency rate (%)', (df['latency'] < 0).mean() * 100, '%'])
    
    latency_df = df[~df['lost']]
    rows.append(['minimum latency', latency_df['latency'].min(), 'sec'])
    rows.append(['maximum latency', latency_df['latency'].max(), 'sec'])
    rows.append(['mean latency', latency_df['latency'].mean(), 'sec'])
    rows.append(['median latency', latency_df['latency'].median(), 'sec'])
    rows.append(['stdev latency', latency_df['latency'].std(), 'sec'])
    
    # calculate jitter
    jitter_df = latency_df['latency'].diff().abs().dropna()
    jitter = np.mean(jitter_df)
    
    rows.append(['latency jitter', jitter, 'sec'])
    
    with open(fout, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'value', 'unit'])
        writer.writerows(rows)
    
    max_length = max(len(sublist[0]) for sublist in rows) + 1
    
    with open(dump_log, "a") as f:
        print("-----------------------------------------------------")
        f.write("-----------------------------------------------------\n")
        for row in rows:
            key, value, unit = row[0], row[1], row[2]
            print(f'{key}:'.ljust(max_length), round(value, 6), unit)
            f.write(''.join([f'{key}: '.ljust(max_length + 1), ' ', str(round(value, 6)), ' ', unit, '\n']))
        print("-----------------------------------------------------")
        f.write("-----------------------------------------------------\n\n")


# ===================== Main Process =====================
if __name__ == "__main__":
    # ===================== Arguments =====================
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--onefile", type=str, help="input filepath")
    parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
    args = parser.parse_args()
    
    if args.onefile is None:
        
        if args.dates is not None:
            dates = sorted(args.dates)
        else:
            raise TypeError("Please specify the date you want to process.")
        
        metadatas = metadata_loader(dates)
        print('\n================================ Start Processing ================================')
        
        pop_error_message(signal='Parsing loss & latency (s/c time synchronize)', stdout=True)
        for metadata in metadatas:
            try:
                print(metadata)
                print('--------------------------------------------------------')
                middle_dir = os.path.join(metadata[0], 'middle')
                data_dir = os.path.join(metadata[0], 'data')
                stats_dir = os.path.join(metadata[0], 'stats')
                makedir(data_dir)
                makedir(stats_dir)
                
                sync_dir = os.path.abspath(os.path.join(metadata[0], '../../..', 'sync'))
                if metadata[3] == 'Phone':
                    sync_file = os.path.join(sync_dir, 'time_sync_{}.json'.format(metadata[1]))
                else:
                    sync_file = os.path.join(sync_dir, 'time_sync_{}.json'.format(metadata[4]))
                
                if os.path.isfile(sync_file):
                    with open(sync_file, 'r') as f:
                        sync_mapping = json.load(f)
                else:
                    sync_mapping = None
                
                for direction in ['dl', 'ul']:
                    if direction == 'dl':
                        txdf, rxdf = generate_dataframe([
                            os.path.join(middle_dir, "udp_dnlk_server_pkt_brief.csv"),
                            os.path.join(middle_dir, "udp_dnlk_client_pkt_brief.csv"),
                        ], dtype={'seq': int, 'rpkg': int, 'frame_id': int,
                                'frame_time_epoch': float, 'pyl_time_epoch': float}, parse_dates=['frame_time', 'pyl_time'])
                        fout = os.path.join(data_dir, "udp_dnlk_loss_latency.csv")
                        fout1 = os.path.join(stats_dir, "udp_dnlk_loss_latency_statistics.csv")
                    else:
                        txdf, rxdf = generate_dataframe([
                            os.path.join(middle_dir, "udp_uplk_client_pkt_brief.csv"),
                            os.path.join(middle_dir, "udp_uplk_server_pkt_brief.csv"),
                        ], dtype={'seq': int, 'rpkg': int, 'frame_id': int,
                                'frame_time_epoch': float, 'pyl_time_epoch': float}, parse_dates=['frame_time', 'pyl_time'])
                        fout = os.path.join(data_dir, "udp_uplk_loss_latency.csv")
                        fout1 = os.path.join(stats_dir, "udp_uplk_loss_latency_statistics.csv")
                
                    # ******************************************************************
                    t = TicToc(); t.tic()
                    # print('sync_mapping:', sync_mapping)
                    print(f">>>>> {fout}")
                    df = get_loss_v2(txdf.copy(), rxdf.copy())
                    df = get_latency(df.copy(), direction=direction, sync_mapping=sync_mapping, thr=THRESHOLD)
                    df.to_csv(fout, index=False)
                    print(f">>>>> {fout1}")
                    with open(RECORDS, "a") as f:
                        f.write(f">>>>> {fout1}\n")
                    get_statistics(df, fout1, thr=THRESHOLD)
                    t.toc(); print()
                    # ******************************************************************
                
                print()
                    
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish parsing loss & latency', stdout=True)
        
    else:
        print(args.onefile)
