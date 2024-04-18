#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Filename: xml_mi_rrc_sync.py
"""
This script is to synchronize device time to our lab server.

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2024-03-29
"""

"""
    Future Development Plans:
    
"""
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
    "mi_compensate",
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
def mi_compensate(fin, sync_mapping):
    if sync_mapping is None:
        return
    
    df = generate_dataframe(fin, parse_dates=['Timestamp', 'Timestamp_BS'])
    if df.empty:
        df.reindex([*df.columns[:2], 'delta', *df.columns[2:-1]], axis='columns').to_csv(fin, index=False)
        return
    
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
    
    dev_timestamp = 'Timestamp'
    for i, row in sync_mapping_df.iterrows():
        lower, upper = row['lower_bound'], row['upper_bound']
        epoch_delta = round(row['delta'], 6)
        delta = pd.Timedelta(seconds=epoch_delta)
        df.loc[(df[dev_timestamp] >= lower) & (df[dev_timestamp] < upper), dev_timestamp] = df[dev_timestamp].add(delta).dt.round(freq='us')
        df.loc[(df[dev_timestamp] >= lower) & (df[dev_timestamp] < upper), 'delta'] = epoch_delta
    
    rearranged_columns = [*df.columns[:2], 'delta', *df.columns[2:-1]]
    df = df[rearranged_columns]
    df = df[(df['Timestamp'] - df['Timestamp_BS'] - pd.Timedelta(hours=8)).dt.total_seconds() < 30].reset_index(drop=True)
    df.to_csv(fin, index=False)
    # df.to_csv('test.csv', index=False)
    return


# ===================== Main Process =====================
if __name__ == "__main__":
    if args.onefile is None:
        
        if args.dates is not None:
            dates = sorted(args.dates)
        else:
            raise TypeError("Please specify the date you want to process.")
        
        metadatas = metadata_loader(dates)
        print('\n================================ Start Processing ================================')
        
        pop_error_message(signal='Patch up S/BS time synchorizing for mi2log files', stdout=True)
        for metadata in metadatas:
            try:
                print(metadata)
                print('--------------------------------------------------------')
                raw_dir = os.path.join(metadata[0], 'raw')
                middle_dir = os.path.join(metadata[0], 'middle')
                data_dir = os.path.join(metadata[0], 'data')
                makedir(data_dir)
                
                sync_dir = os.path.abspath(os.path.join(metadata[0], '../../..', 'sync'))
                sync_file = os.path.join(sync_dir, 'time_sync_{}.json'.format(metadata[4]))
                if os.path.isfile(sync_file):
                    with open(sync_file, 'r') as f:
                        sync_mapping = json.load(f)
                else:
                    sync_mapping = None
                
                # print('sync_mapping:', sync_mapping)
                
                try:
                    filenames = [s for s in os.listdir(raw_dir) if s.startswith('diag_log') and s.endswith(('.xml', '.txt'))]
                except:
                    filenames = [s for s in os.listdir(middle_dir) if s.startswith('diag_log') and s.endswith(('.xml', '.txt'))]
                
                fin = os.path.join(raw_dir, filenames[0])
                # ******************************************************************
                # t = TicToc(); t.tic()
                # pop_error_message(signal='Compensating rrc.csv')
                # fout = os.path.join(data_dir, filenames[0].replace('.xml', '_rrc.csv').replace('.txt', '_rrc.csv'))
                # print(f">>>>> {fout}")
                # mi_compensate(fout, sync_mapping=sync_mapping)
                # t.toc(); print()
                
                # t = TicToc(); t.tic()
                # pop_error_message(signal='Compensating ml1.csv')
                # fout = os.path.join(data_dir, filenames[0].replace('.xml', '_ml1.csv').replace('.txt', '_ml1.csv'))
                # print(f">>>>> {fout}")
                # mi_compensate(fout, sync_mapping=sync_mapping)
                # t.toc(); print()
                
                # t = TicToc(); t.tic()
                # pop_error_message(signal='Compensating nr_ml1.csv')
                # fout = os.path.join(data_dir, filenames[0].replace('.xml', '_nr_ml1.csv').replace('.txt', '_nr_ml1.csv'))
                # print(f">>>>> {fout}")
                # mi_compensate(fout, sync_mapping=sync_mapping)
                # t.toc(); print()
                # ******************************************************************
                
                print()
                    
            except Exception as e:
                pop_error_message(e, locate=metadata)
                
        pop_error_message(signal='Finish S/BS time synchorizing for mi2log files', stdout=True)
        
    else:
        print(args.onefile)
