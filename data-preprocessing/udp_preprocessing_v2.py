#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Filename: udp_preprocessing_v2.py
"""
overall preprocessing v2

Author: Yuan-Jye Chen 2024-05-17
Update: Yuan-Jye Chen 2024-05-17
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
import pandas as pd

from myutils import *
from mi2log.xml_mi_rrc import xml_to_csv_rrc
from mi2log.xml_mi_nr_ml1 import xml_to_csv_nr_ml1
from mi2log.xml_mi_ml1 import xml_to_csv_ml1
from mi2log.xml_mi_sync import mi_compensate
from udp.udp_pcap_to_csv import pcap_to_csv
from udp.parse_pkt_info_readline import parse_pkt_info
from udp.parse_loss_latency import get_loss_v2
from udp.parse_loss_latency import get_latency
from udp.parse_loss_latency import get_statistics

# ===================== Arguments =====================
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--onefile", type=str, help="input filepath")
parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
args = parser.parse_args()


# ===================== Main Process =====================
if __name__ == "__main__":
    if args.dates is not None:
        dates = sorted(args.dates)
    else:
        raise TypeError("Please specify the date you want to process.")
    
    metadatas = metadata_loader(dates)
    
    print('\n================================ Start Processing ================================')
    t0 = TicToc(); t0.tic()
    
    print('mi2log_xml -> csv')
    
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
        fout = os.path.join(data_dir, filenames[0].replace('.xml', '_rrc.csv').replace('.txt', '_rrc.csv'))
        print(f">>>>> {fin} -> {fout}")
        xml_to_csv_rrc(fin, fout)
        print(">>>>> Compensating...")
        mi_compensate(fout, sync_mapping=sync_mapping)
        t.toc(); print()
        
        t = TicToc(); t.tic()
        fout = os.path.join(data_dir, filenames[0].replace('.xml', '_ml1.csv').replace('.txt', '_ml1.csv'))
        print(f">>>>> {fin} -> {fout}")
        xml_to_csv_ml1(fin, fout)
        print(">>>>> Compensating...")
        mi_compensate(fout, sync_mapping=sync_mapping)
        t.toc(); print()
        
        t = TicToc(); t.tic()
        fout = os.path.join(data_dir, filenames[0].replace('.xml', '_nr_ml1.csv').replace('.txt', '_nr_ml1.csv'))
        print(f">>>>> {fin} -> {fout}")
        xml_to_csv_nr_ml1(fin, fout)
        print(">>>>> Compensating...")
        mi_compensate(fout, sync_mapping=sync_mapping)
        t.toc(); print()
        # ******************************************************************
        
        print()
    
    
    print('pcap -> csv')
    
    for metadata in metadatas:
        print(metadata)
        print('--------------------------------------------------------')
        raw_dir = os.path.join(metadata[0], 'raw')
        middle_dir = os.path.join(metadata[0], 'middle')
        makedir(middle_dir)
        
        filenames = [s for s in os.listdir(raw_dir) if s.startswith(('server_pcap', 'client_pcap')) and s.endswith('.pcap')]
        for j, filename in enumerate(filenames):
            t = TicToc(); t.tic()
            fin = os.path.join(raw_dir, filename)
            fout = os.path.join(middle_dir, filename.replace('.pcap', '.csv'))
            print(f">>>>> {fin} -> {fout}")
            # ******************************************************************
            pcap_to_csv(fin, fout)
            # ******************************************************************
            t.toc(); print()
        
        print()

    
    print('parse packet info')
    
    for metadata in metadatas:
        print(metadata)
        print('--------------------------------------------------------')
        middle_dir = os.path.join(metadata[0], 'middle')
        
        filenames = [s for s in os.listdir(middle_dir) if s.endswith('.csv')]
        # ******************************************************************
        t = TicToc(); t.tic()
        print('Server | Downlink')
        try: filename = [s for s in filenames if s.startswith(('server_pcap_BL', 'server_pcap_DL'))][0]
        except: filename = None
        if filename is not None:
            fin = os.path.join(middle_dir, filename)
            fout = os.path.join(middle_dir, "udp_dnlk_server_pkt_brief.csv")
            print(f">>>>> {fin} -> {fout}")
            parse_pkt_info(fin, fout, "server", "dl", "udp", locate=str(metadata)+'\nServer | Downlink')
        t.toc(); print()
        
        t = TicToc(); t.tic()
        print('Client | Downlink')
        try: filename = [s for s in filenames if s.startswith(('client_pcap_BL', 'client_pcap_DL'))][0]
        except: filename = None
        if filename is not None:
            fin = os.path.join(middle_dir, filename)
            fout = os.path.join(middle_dir, "udp_dnlk_client_pkt_brief.csv")
            print(f">>>>> {fin} -> {fout}")
            parse_pkt_info(fin, fout, "client", "dl", "udp", locate=str(metadata)+'\nClient | Downlink')
        t.toc(); print()
        
        t = TicToc(); t.tic()
        print('Server | Uplink')
        try: filename = [s for s in filenames if s.startswith(('server_pcap_BL', 'server_pcap_UL'))][0]
        except: filename = None
        if filename is not None:
            fin = os.path.join(middle_dir, filename)
            fout = os.path.join(middle_dir, "udp_uplk_server_pkt_brief.csv")
            print(f">>>>> {fin} -> {fout}")
            parse_pkt_info(fin, fout, "server", "ul", "udp", locate=str(metadata)+'\nServer | Uplink')
        t.toc(); print()
        
        t = TicToc(); t.tic()
        print('Client | Uplink')
        try: filename = [s for s in filenames if s.startswith(('client_pcap_BL', 'client_pcap_UL'))][0]
        except: filename = None
        if filename is not None:
            fin = os.path.join(middle_dir, filename)
            fout = os.path.join(middle_dir, "udp_uplk_client_pkt_brief.csv")
            print(f">>>>> {fin} -> {fout}")
            parse_pkt_info(fin, fout, "client", "ul", "udp", locate=str(metadata)+'\nClient | Uplink')
        t.toc(); print()
        # ******************************************************************
        print()
            
    
    print('align data in the same trip and the same device')
    
    for metadata in metadatas:
        print(metadata)
        print('--------------------------------------------------------')
        middle_dir = os.path.join(metadata[0], 'middle')
        
        csvfiles = ["udp_dnlk_server_pkt_brief.csv", "udp_dnlk_client_pkt_brief.csv", "udp_uplk_server_pkt_brief.csv", "udp_uplk_client_pkt_brief.csv"]
        csvfiles = [os.path.join(middle_dir, s) for s in csvfiles]
        csvfiles = [s for s in csvfiles if os.path.isfile(s)]
        
        st_seq = []
        ed_seq = []
        for file in csvfiles:
            print(file)
            df = pd.read_csv(file)
            df = str_to_datetime_batch(df, parse_dates=['frame_time'])
            # 去頭 0 秒
            try: st_seq.append(df[df['frame_time'] >= df.iloc[0]['frame_time'] + pd.Timedelta(seconds=0)].reset_index(drop=True).iloc[0]['seq'])
            except: st_seq.append(1)
            # 截尾 5 秒
            try: ed_seq.append(df[df['frame_time'] < df.iloc[-1]['frame_time'] - pd.Timedelta(seconds=5)].reset_index(drop=True).iloc[-1]['seq'])
            except: ed_seq.append(1)
            del df

        st_seq = max(st_seq)
        ed_seq = min(ed_seq)
        for file in csvfiles:
            df = pd.read_csv(file)
            df = df[(df['seq'] >= st_seq) & (df['seq'] < ed_seq)]
            df.to_csv(file, index=False)
            del df
            
        print()


    print('parse loss and latency')
    
    THRESHOLD = 100e-3  # (seconds)
    HASH_SEED = time.time()
    RECORDS = 'parse_loss_latency_' + query_datetime() + '-' + generate_hex_string(HASH_SEED, 5) + '_records.log'
    
    for metadata in metadatas:
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
        t0.toc(); print()
