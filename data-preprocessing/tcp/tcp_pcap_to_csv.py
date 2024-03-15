#!/usr/bin/python3
### Filename: tcp_pcap_to_csv.py

"""
Convert tcp pcap into csv format.
Extract the features that you need.

Usages:

(1) decode only one files. It would decode inplace and change the suffix from '.pcap' into '.csv'.
$ python3 tcp_pcap_to_csv.py -i [input_filepath]

(2) decode files in one directory. If you do not set [output_dirpath], it would decode inplace.
$ python3 tcp_pcap_to_csv.py -D [input_dirpath]
$ python3 tcp_pcap_to_csv.py -D [input_dirpath] -O [output_dirpath]

(3) decode a batch of files => go to Users Settings and modify.
$ python3 tcp_pcap_to_csv.py

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2022-10-06
"""

"""
    Future Development Plan
        (1) Neglect filename start with ".~lock". (e.g., ".~lock.packet_info.csv#", ".~lock.client_pcap_BL_sm05_3210_3211_2022-09-29_16-24-57.csv#")
            => solved by not str.startswith()
    
"""
import subprocess
import os
import sys
import time
import argparse
import traceback
from operator import sub
from pprint import pprint
from pytictoc import TicToc

# --------------------- Arguments ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="input filepath")
parser.add_argument("-D", "--indir", type=str,
                    help="input dirctory path")
parser.add_argument("-O", "--outdir", type=str,
                    help="output dirctory path")
args = parser.parse_args()

# ********************* User Settings *********************
database = "/home/wmnlab/D/database/"
date = "2022-09-29"
db_path = os.path.join(database, date)
Exp_Name = {  # experiment_name:(number_of_experiment_rounds, list_of_experiment_round)
                # If the list is empty, it will list all directories in the current directory by default.
                # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    # "_Bandlock_Udp":(1, ["#01"]),
    # "_Bandlock_Udp":(5, ["#02", "#03", "#04", "#05", "#06"]),
    # "_Bandlock_Udp":(4, ["#01", "#02", "#03", "#04"]),
    # "_Bandlock_Udp":(6, []),
    # "_Bandlock_Udp":(4, []),
    # "_Bandlock_Tcp":(4, []),
    # "_Udp_Stationary_Bandlock":(1, []), 
    # "_Udp_Stationary_SameSetting":(1, []),
    "_Modem_Phone_Comparative_Exeriments": (6, []),
}
devices = sorted([
    # "sm03",
    # "sm04",
    # "sm05", 
    # "sm06",
    "sm07",
    "sm08",
])
# *********************************************************

# --------------------- Util Functions ---------------------
def pcap_to_csv(fin, fout):
    t1 = TicToc()  # create instance of class
    t1.tic()  # Start timer
    try:
        # Frame Number: (frame.number) the number of packet captured by tcpdump, start from 1
        # Arrival Time: (frame.time) packet arrival time 'Sep 29, 2022 16:24:58.254416000' (utc-8, CST)
        # Epoch Time: (frame.time_epoch) packet arrival time '1664439898.254416000' (utc-0)
        # Frame Length (frame.len)
        # Protocol (_ws.col.Protocol) ~= Protocol (ip.proto): TCP (6), UDP (17)
        # Packet type (sll.pkttype): Sent by us (4) => TX, Unicast to us (0) => RX
        # IP Length (ip.len)
        # Source Address (ip.src)
        # Destination Address (ip.dst)
        # UDP Length / TCP Length (udp.length / tcp.len)
        # Source Port (udp.srcport / tcp.srcport)
        # Destination (udp.dstport / tcp.srcport)
        # Data Length (data.len)
        # Data (data.data) == UDP payload / TCP payload (udp.payload / tcp.payload)
        # Info (_ws.col.Info): 48530 → 3211 [PSH, ACK] Seq=1 Ack=1 Win=128 Len=37 TSval=2269012485 TSecr=969346414
        s = "tshark -r {} -T fields \
            -e frame.number -e frame.time -e frame.time_epoch -e frame.len \
            -e sll.pkttype -e _ws.col.Protocol \
            -e ip.proto -e ip.len -e ip.src -e ip.dst \
            -e tcp.len -e tcp.srcport -e tcp.dstport \
            -e data.len -e tcp.payload -e _ws.col.Info \
            -e tcp.seq_raw -e tcp.seq -e tcp.nxtseq -e tcp.ack_raw -e tcp.ack \
            -e tcp.analysis.acks_frame -e tcp.analysis.ack_rtt \
            -e tcp.analysis.initial_rtt -e tcp.analysis.bytes_in_flight -e tcp.analysis.push_bytes_sent \
            -e tcp.analysis.retransmission -e tcp.analysis.fast_retransmission \
            -e tcp.analysis.out_of_order \
            -E header=y -E separator=@ > {}".format(fin, fout)
        p = subprocess.Popen(s, shell=True, preexec_fn=os.setpgrp)
        while p.poll() is None:
            # print(p.pid, p.poll())
            time.sleep(1)
    except:
        ### Record error message without halting the program
        t1.toc()
        return (fin, fout, traceback.format_exc())
    t1.toc()
    return (fin, fout, None)

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


if __name__ == "__main__":
    t = TicToc()  # create instance of class
    t.tic()  # Start timer
    # --------------------- (1) convert only one file (set arguments) ---------------------
    if args.input:
        fin = args.input
        ### Check the input filename format, and whether users specify output filename.
        if not fin.endswith(".pcap"):
            print("Input: '{}' does not endswith 'pcap', the program is terminated.".format(fin))
            sys.exit()
        fout = "{}.csv".format(fin[:-5])
        ### decoding ...
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
        err_handle = pcap_to_csv(fin, fout)
        flag = error_handling(err_handle)
        print()
        if not flag:
            print("**************************************************")
            print("No error occurs!!")
            print("**************************************************")
        t.toc()
        sys.exit()

    # --------------------- (2) convert files in one directory (set arguments) ---------------------
    if args.indir:
        err_handles = []
        input_path = args.indir
        ### Check if the input directory exists
        if not os.path.isdir(input_path):
            print("FileExistsError: directory '{}' does not exists, the program is terminated.".format(input_path))
            sys.exit()
        output_path = args.outdir if args.outdir else input_path
        filenames = os.listdir(input_path)
        pprint(filenames)
        for filename in filenames:
            # if not filename.endswith(".pcap"):
            if not filename.startswith(("server_pcap", "client_pcap")) or not filename.endswith(".pcap"):
                continue
            fin = os.path.join(input_path, filename)
            fout = os.path.join(output_path, "{}.csv".format(filename[:-5]))
            makedir(output_path)
            ### decoding ...
            print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
            err_handle = pcap_to_csv(fin, fout)
            err_handles.append(err_handle)
        print()
        ### Check errors
        flag = False
        for err_handle in err_handles:
            flag = error_handling(err_handle)
        if not flag:
            print("**************************************************")
            print("No error occurs!!")
            print("**************************************************")
        t.toc()
        sys.exit()

    # --------------------- (3) convert a batch of files (User Settings) ---------------------
    err_handles = []
    ### iteratively decode every pcap file
    for _exp, (_times, _rounds) in Exp_Name.items():
        ### Check if these directories exist
        exp_path = os.path.join(db_path, _exp)
        print(exp_path)
        exp_dirs = []
        for i, dev in enumerate(devices):
            if _rounds:
                exp_dirs.append([os.path.join(exp_path, dev, _round) for _round in _rounds])
            else:
                _rounds = sorted(os.listdir(os.path.join(exp_path, dev)))
                exp_dirs.append([os.path.join(exp_path, dev, item) for item in _rounds])
            exp_dirs[i] = [item for item in exp_dirs[i] if os.path.isdir(item)]
            print(_times)
            pprint(exp_dirs[i])
            if len(exp_dirs[i]) != _times:
                print("************************************************************************************************")
                print("Warning: the number of directories does not match your specific number of experiment times.")
                print("************************************************************************************************")
                print()
                sys.exit()
        print()
        ### Check if pcap files exist, and then run decoding
        print(_exp)
        for j in range(_times):
            for i, dev in enumerate(devices):
                print(exp_dirs[i][j])
                dir = os.path.join(exp_dirs[i][j], "raw")
                filenames = os.listdir(dir)
                for filename in filenames:
                    # if not filename.endswith(".pcap"):
                    if not filename.startswith(("server_pcap", "client_pcap")) or not filename.endswith(".pcap"):
                        continue
                    # print(filename)
                    fin = os.path.join(dir, filename)
                    fout = os.path.join(dir, "..", "data", "{}.csv".format(filename[:-5]))
                    makedir(os.path.join(dir, "..", "data"))
                    ### decoding ...
                    print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
                    err_handle = pcap_to_csv(fin, fout)
                    err_handles.append(err_handle)
            print()
    ### Check errors
    flag = False
    for err_handle in err_handles:
        flag = error_handling(err_handle)
    if not flag and err_handles:
        print("**************************************************")
        print("No error occurs!!")
        print("**************************************************")
    t.toc()
