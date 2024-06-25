#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Filename: parse_pkt_info_readline.py
"""
Analyze packet capture of udp experiments.
Need to convert .pcap into .csv by tshark with script udp_pcap_to_csv.py

若 pcap, ho.csv 檔案為空:
    start_time = '-'
    end_time = '-'
    exp_time = 0

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2022/10/09
"""

"""
    Future Development Plan
        (1) readline of csv 逐行（不要一次 handle 整個 dataframe，因為 payload 太長，RAM 會爆掉），目前先把 rxdf, txdf 分別處理，不要兩個都讀進來才處理（30min-500pps-pcap.csv 還夠用）
        (2) 未來改用 udp sniffer 時，最好先確認一下 seq 沒有 duplicate，在這裡先處理掉 duplicate 的問題，好讓 parse loss latency 可以直接 pass
    
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
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *

__all__ = [
    'parse_pkt_info',
]


class Payl:
    LENGTH = 250              # (Bytes)
    TAG = "000425d401df5e76"  # 2 71828 3 1415926 (hex)            : 8-bytes
    OFS_TIME = (16, 24)       # epoch time of 'yyyy/mm/dd hh:mm:ss': 4-bytes
    OFS_USEC = (24, 32)       # microsecond (usec)                 : 4-bytes
    OFS_SEQN = (32, 40)       # sequence number (start from 1)     : 4-bytes

XMIT = 4; RECV = 0
UDP = 17; TCP = 6


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
def parse_pkt_info(fin, fout, side, direction, protoc, locate='.'):   
    new_header = ["seq", "rpkg", "frame_id", "frame_time", "frame_time_epoch", "pyl_time", "pyl_time_epoch"]
    
    timestamp_list = []
    seq_set = set()
    with open(fin, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='@')
        header = next(csvreader)  # read the first line (header)
        
        for content in tqdm(csvreader):  # iterate through the rows in the csv file
            # print(content)
            row = {k: v for k, v in zip(header, content)}
            
            # server/client; uplink/downlink
            if (side == "server" and direction == "ul") and int(row['sll.pkttype']) != RECV:
                continue
            elif (side == "client" and direction == "ul") and int(row['sll.pkttype']) != XMIT:
                continue
            elif (side == "server" and direction == "dl") and int(row['sll.pkttype']) != XMIT:
                continue
            elif (side == "client" and direction == "dl") and int(row['sll.pkttype']) != RECV:
                continue
            
            # udp/tcp
            if row['ip.proto'].isdigit():
                ip_proto = int(row["ip.proto"])
            else:
                ip_proto = int(row["ip.proto"].split(',')[0])

            if protoc == "udp" and ip_proto != UDP:
                continue
            elif protoc == "tcp" and ip_proto != TCP:
                continue
            
            # check customized packet: length of udp header is 8 Bytes
            if protoc == "udp":
                length = row['udp.length']
                if not length.isdigit():
                    length = int(length.split(',')[0])
                else:
                    length = int(length)

                if not (length > Payl.LENGTH) and (length % Payl.LENGTH == 8):
                    continue

            elif protoc == "tcp" and Payl.TAG not in row['tcp.payload']:
                continue
            
            # decompress the repacked segments
            payload = row['udp.payload']
            # rpkg_num = int(row['udp.length']) // Payl.LENGTH
            rpkg_num = length // Payl.LENGTH
            offset = [s * Payl.LENGTH * 2 for s in list(range(rpkg_num))]  # 1-Byte: 2-hex-digits
            
            sequence_list = []
            payload_time_list = []
            payload_time_epoch_list = []
            for ofs in offset:
                try:
                    datetimedec = int(payload[ofs + Payl.OFS_TIME[0] : ofs + Payl.OFS_TIME[1]], 16)
                    microsec = int(payload[ofs + Payl.OFS_USEC[0] : ofs + Payl.OFS_USEC[1]], 16)
                    seq = int(payload[ofs + Payl.OFS_SEQN[0] : ofs + Payl.OFS_SEQN[1]], 16)
                except Exception as e:
                    pop_error_message(e, locate=str(locate)+'\n'+str(['frame.time'])+', '+str(payload), raise_flag=True)
                payload_time_epoch = round(datetimedec + microsec * 1e-6, 6)
                payload_time = epoch_to_datetime(payload_time_epoch)
                
                sequence_list.append(seq)
                payload_time_list.append(payload_time)
                payload_time_epoch_list.append(datetimedec + microsec * 1e-6)
            
            for (seq, pyl_time, pyl_time_epoch) in zip(sequence_list, payload_time_list, payload_time_epoch_list):
                if (seq, pyl_time_epoch) not in seq_set:
                    timestamp_list.append([seq, rpkg_num, int(row['frame.number']), pd.to_datetime(row['frame.time']).replace(tzinfo=None), float(row['frame.time_epoch']), pyl_time, pyl_time_epoch])
                    # print(pd.to_datetime(row['frame.time']))
                    # print(pd.to_datetime(row['frame.time']).tz_localize(None))  # Python 3.12 即將被棄用
                    # print(pd.to_datetime(row['frame.time']).replace(tzinfo=None))
                    # 看不懂這什麼鬼 warning, 目前解不掉:
                    # FutureWarning: Parsing '{res.tzname}' as tzlocal (dependent on system timezone) is deprecated
                    # and will raise in a future version. Pass the 'tz' keyword or call tz_localize after construction instead
                    seq_set.add((seq, pyl_time_epoch))

    timestamp_list = sorted(timestamp_list, key = lambda v : v[0])

    # print("output >>>", fout)
    with open(fout, 'w', newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(new_header)
        writer.writerows(timestamp_list)
    
    return


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
        
        pop_error_message(signal='Parsing packet info into brief format', stdout=True)
        for metadata in metadatas:
            try:
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
                    
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish parsing packet info', stdout=True)
        pop_error_message(signal='Aligning data on the same device within one trip', stdout=True)
        
        for metadata in metadatas:
            try:
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
                
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish aligning data', stdout=True)
        
    else:
        print(args.onefile)
