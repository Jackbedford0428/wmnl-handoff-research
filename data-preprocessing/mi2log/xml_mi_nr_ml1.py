#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Filename: xml_mi_nr_ml1.py
"""
This script requires the txt or xml file which is generated from mi_offline_analysis.py and the mi2log file.
The rows show the information of each diagnostic mode packets (dm_log_packet) from MobileInsight.
The columns are indicators about whether a packet has the type of the message or not.

Author: Sheng-Ru Zeng
Update: Yuan-Jye Chen 2024-03-27
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
from bs4 import BeautifulSoup
from pprint import pprint
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, parent_dir)

from myutils import *
# from xml_mi_sync import *

__all__ = [
    "xml_to_csv_nr_ml1",
]


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
def xml_to_csv_nr_ml1(fin, fout):
    f = open(fin, encoding="utf-8")  # diag_log_xxxx_xxxx_rrc.xml
    f2 = open(fout, 'w')  # diag_log_xxxx_xxxx_nr_ml1.csv
    
    delete = False
    
    # Writing the column names...
    # -------------------------------------------------
    columns = [
        "Timestamp", "Timestamp_BS", "type_id",
        "Raster ARFCN",
        "Num Cells",
        "Serving Cell Index",
        "Serving Cell PCI",
    ]
    f2.write(",".join(columns) + "\n")

    l = f.readline()

    max_length = 0
    while l:
        if r"<dm_log_packet>" in l:
            soup = BeautifulSoup(l, 'html.parser')
            # try:
            #     if r"</dm_log_packet>" in l:
            #         timestamp = soup.find(key='device_timestamp').get_text()
            # except:
            #     print('line:', iter_number)
            #     raise
            
            if r"</dm_log_packet>" in l:
                timestamp = soup.find(key='device_timestamp').get_text()
                
            timestamp_bs = soup.find(key="timestamp").get_text()
            type_id = soup.find(key="type_id").get_text()

            if type_id == '5G_NR_ML1_Searcher_Measurement_Database_Update_Ext':
                arfcn = soup.find(key="Raster ARFCN").get_text()
                num_cells = soup.find(key="Num Cells").get_text()
                serving_cell_idex = soup.find(key="Serving Cell Index").get_text()
                serving_cell_pci = soup.find(key="Serving Cell PCI").get_text()
                pcis = [pci.get_text() for pci in soup.findAll(key="PCI")]
                rsrps = [rsrp.get_text() for rsrp in soup.findAll(key="Cell Quality Rsrp")]
                rsrqs = [rsrq.get_text() for rsrq in soup.findAll(key="Cell Quality Rsrq")]

                A = []
                for i in range(int(num_cells)):    
                    A.append(pcis[i])
                    A.append(rsrps[i])
                    A.append(rsrqs[i])

                x = len([timestamp, timestamp_bs, type_id, arfcn, num_cells, serving_cell_idex, serving_cell_pci] + A)
                max_length = x if x > max_length else max_length
                f2.write(",".join([timestamp, timestamp_bs, type_id, arfcn, num_cells, serving_cell_idex, serving_cell_pci] + A) + "\n")
            
            else: # 只處理nr_ml1資料過濾其他type
                while l and r"</dm_log_packet>" not in l:
                    l = f.readline()
                # soup = BeautifulSoup(l, 'html.parser')
                # try:
                #     if r"</dm_log_packet>" in l:
                #         timestamp = soup.find(key='device_timestamp').get_text()
                # except:
                #     print('line:', iter_number)
                #     raise

            l = f.readline()
            
        else:
            print(l,"Error!")
            delete = True
            break 
            
    f2.close()
    f.close()
    
    if delete:
        os.system(f"rm {fout}")
    else:
        # csv Header process
        with open(fout, 'r') as csvinput:
            new_f = fout[:-4]+"_new.csv"
            l = csvinput.readline()
            x = len(l.split(','))
            X = []
            for i in range(int((max_length-x)/3)):
                X += [f"PCI{i}",f"RSRP{i}",f"RSRQ{i}"]
            X = columns+X
            with open(new_f, 'w') as csvoutput:
                csvoutput.write(",".join(X)+'\n')
                l = csvinput.readline()
                while l:
                    csvoutput.write(l)
                    l = csvinput.readline()
        os.system(f"rm {fout}") # Remove
        os.system(f"mv {new_f} {fout}") # Rename


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
        
        pop_error_message(signal='Converting mi2log_xml to nr_ml1.csv', stdout=True)
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
                t = TicToc(); t.tic()
                fout = os.path.join(data_dir, filenames[0].replace('.xml', '_nr_ml1.csv').replace('.txt', '_nr_ml1.csv'))
                print(f">>>>> {fin} -> {fout}")
                xml_to_csv_nr_ml1(fin, fout)
                mi_compensate(fout, sync_mapping=sync_mapping)
                t.toc(); print()
                # ******************************************************************
                
                print()
                    
            except Exception as e:
                pop_error_message(e, locate=metadata, raise_flag=True)
                
        pop_error_message(signal='Finish converting mi2log_xml to nr_ml1.csv', stdout=True)
        
    else:
        print(args.onefile)
