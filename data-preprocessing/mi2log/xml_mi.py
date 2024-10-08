#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Filename: xml_mi.py
"""
This script requires the txt or xml file which is generated from mi_offline_analysis.py and the mi2log file.
The rows show the information of each diagnostic mode packets (dm_log_packet) from MobileInsight.
The columns are indicators about whether a packet has the type of the message or not.

Author: Sheng-Ru Zeng
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
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *
from xml_mi_rrc import *
from xml_mi_nr_ml1 import *
from xml_mi_ml1 import *
from xml_mi_sync import *


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


# ===================== Main Process =====================
if __name__ == "__main__":
    # ===================== Arguments =====================
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--onefile", type=str, nargs=5, help="input arguments")
    parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
    args = parser.parse_args()
    
    if args.onefile is None:
        if args.dates is not None:
            dates = sorted(args.dates)
            metadatas = metadata_loader(dates)
            for metadata in metadatas:
                print(metadata)
        else:
            raise TypeError("Please specify the date you want to process.")
    else:
        metadatas = [tuple(args.onefile)]
        print(metadatas)
    
    print('\n================================ Start Processing ================================')
    t0 = TicToc(); t0.tic()
    
    # pop_error_message(signal='Converting mi2log_xml to *.csv', stdout=True)
    try:
        for metadata in metadatas:
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
            t = TicToc(); t.tic()
            # pop_error_message(signal='Converting mi2log_xml to rrc.csv')
            fout = os.path.join(data_dir, filenames[0].replace('.xml', '_rrc.csv').replace('.txt', '_rrc.csv'))
            print(f">>>>> {fin} -> {fout}")
            xml_to_csv_rrc(fin, fout)
            print(">>>>> Compensating...")
            mi_compensate(fout, sync_mapping=sync_mapping)
            t.toc(); print()
            
            t = TicToc(); t.tic()
            # pop_error_message(signal='Converting mi2log_xml to ml1.csv')
            fout = os.path.join(data_dir, filenames[0].replace('.xml', '_ml1.csv').replace('.txt', '_ml1.csv'))
            print(f">>>>> {fin} -> {fout}")
            xml_to_csv_ml1(fin, fout)
            print(">>>>> Compensating...")
            mi_compensate(fout, sync_mapping=sync_mapping)
            t.toc(); print()
            
            t = TicToc(); t.tic()
            # pop_error_message(signal='Converting mi2log_xml to nr_ml1.csv')
            fout = os.path.join(data_dir, filenames[0].replace('.xml', '_nr_ml1.csv').replace('.txt', '_nr_ml1.csv'))
            print(f">>>>> {fin} -> {fout}")
            xml_to_csv_nr_ml1(fin, fout)
            print(">>>>> Compensating...")
            mi_compensate(fout, sync_mapping=sync_mapping)
            t.toc(); print()
            # ******************************************************************
            
            print()
                
    except Exception as e:
        print(traceback.format_exc())
    
    finally:
        print()
        t0.toc(); print()
