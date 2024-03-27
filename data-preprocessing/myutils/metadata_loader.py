#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import traceback
import random
import time

__all__ = [
    "makedir",
    "generate_hex_string",
    "query_datetime",
    # "pop_error_message",
    "metadata_loader",
]

def makedir(dirpath):
    if not os.path.isdir(dirpath):
        print('makedir:', dirpath)
        os.makedirs(dirpath)

def generate_hex_string(seed, length=16):
    # 設定亂數種子
    random.seed(seed)
    # 生成隨機數
    random_number = random.getrandbits(length * 4)  # 16進制的位數需要4位二進制數表示
    # 轉換為16進位制字串
    hex_string = hex(random_number)[2:]  # [2:]是因為hex()函數生成的字符串開頭是'0x'，需要去掉
    return hex_string.zfill(length)  # 確保字串長度為length，不足的話在前面補0

def query_datetime():
    return time.strftime('%Y-%m-%d-%H:%M:%S')

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

def metadata_loader(dates, ue='any'):
    dates = sorted(dates)
    with open(os.path.join(os.path.dirname(__file__), "..", "db_path.txt"), "r") as f:
            PATHS_TO_DATABASE = [s.strip() for s in f.readlines()]
    
    date_paths = []
    
    for date in dates:
        not_found_in_database = []
        
        for db_path in PATHS_TO_DATABASE:
            date_path = os.path.join(db_path, date)
            if os.path.isdir(date_path):
                date_paths.append(date_path)
            else:
                not_found_in_database.append(date_path)
        
        if len(not_found_in_database) == len(PATHS_TO_DATABASE):
            error_message = "[Errno 2] No such file or directory:\n"
            for date_path in not_found_in_database:
                error_message += "  '{}'\n".format(date_path)
            raise FileNotFoundError(error_message.strip())
    
    metadatas = []
    
    for i, date_path in enumerate(date_paths):
        print('------------------')
        print(os.path.basename(date_path))
        
        yml_file = os.path.join(date_path, os.path.basename(date_path) + '.yml')
        with open(yml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        for exp_name, exp in data.items():
            print(' ', exp_name, '->', 'Skip:', exp['skip'], '|', 'UE:', exp['ue'], '|', 'Laptop:', exp['laptop'], '|', 'Route:', exp['route'])
            print('   ', exp['devices'])
            
            if ue == 'any':
                if exp['skip']:
                    continue
            else:
                if exp['skip'] and exp['ue'] != ue:
                # if exp['skip'] and exp['ue'] != 'Phone':
                # if exp['skip'] and exp['ue'] != 'Modem':
                    continue
            
            for dev in exp['devices']:
                print('   ', dev)
                
                for trip in exp['ods']:
                    if trip == 0:
                        continue
                    else:
                        trip = f'#{trip:02d}'
                    data_dir = os.path.join(date_path, exp_name, dev, trip)
                    print('     ', data_dir, os.path.isdir(data_dir))
                    
                    metadatas.append((data_dir, dev, trip, exp['ue'], exp['laptop']))
                    
    return metadatas
