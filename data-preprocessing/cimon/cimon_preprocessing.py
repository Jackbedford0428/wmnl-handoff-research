#!/usr/bin/python3
### Filename: cimon_preprocessing.py

"""
Preprocess csv file of CellInfoMonitor into a good-to-read format that can be read by pandas.
Remove abnormal data by rules.
It would generate a middle file with suffix '_new.csv' and an final output file with suffix '_preproc.csv'

Usages:

(1) decode only one files. It would decode inplace and change the suffix from '.csv' into '_new.csv' and '_preproc.csv'.
    if you do not set [version], the default (latest) version is 'v2'.
$ python3 cimon_preprocessing.py -i [input_filepath]
$ python3 cimon_preprocessing.py -i [input_filepath] -v [version]

(2) decode files in one directory. If you do not set [output_filepath], it would decode inplace.
    if you do not set [version], the default (latest) version is 'v2'.
$ python3 cimon_preprocessing.py -D [input_dirpath]
$ python3 cimon_preprocessing.py -D [input_dirpath] -v [version]
$ python3 cimon_preprocessing.py -D [input_dirpath] -O [output_dirpath] -v [version]

(3) decode a batch of files => go to Users Settings and modify.
$ python3 cimon_preprocessing.py

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2022-10-09
"""

"""
    Future Development Plans:
    
"""
import os
import sys
import argparse
import pandas as pd
from pprint import pprint
from tqdm import tqdm
from pytictoc import TicToc

# --------------------- Arguments ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="input filepath")
parser.add_argument("-D", "--indir", type=str,
                    help="input dirctory path")
parser.add_argument("-O", "--outdir", type=str,
                    help="output dirctory path for middle files and final output files")
parser.add_argument("-v", "--version", type=str,
                    help="CellInfoMonitor (latest version, v2)", default='v2')
args = parser.parse_args()

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
date = "2022-12-22"
devices = sorted([
    # "sm00",
    # "sm01",
    # "sm02",
    # "sm03",
    # "sm04",
    "sm05",
    "sm06",
    "sm07",
    "sm08",
    # "qc00",
    # "qc01",
    # "qc02",
    # "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    # "tsync": (1, None),
    # "_Bandlock_Udp": (4, ["#01", "#02", "#03", "#04"]),
    # "_Bandlock_Udp": (4, ["#03", "#04", "#05", "#06"]),
    # "_Bandlock_Udp": (4, []),
    # "_Bandlock_Udp": (6, []),
    # "_Bandlock_Udp_B1_B3":  (6, []),
    # "_Bandlock_Udp_B3_B28": (2, []),
    # "_Bandlock_Udp_B28_B1": (2, []),
    "_Bandlock_Udp_B1_B3": (4, []),
    "_Bandlock_Udp_B3_B7": (4, []),
    "_Bandlock_Udp_B7_B8": (4, []),
    "_Bandlock_Udp_B8_B1": (4, []),
}
cimon_version = 'v2'
# *****************************************************************************

# ******************************************************************************************************
# https://www.quora.com/How-are-cells-defined-in-LTE-if-a-single-eNodeB-can-serve-multiple-cells
# According to 3GPP, a cell (called ECI, E-UTRAN Cell Identity, in LTE) is a carrier on a sector.
# A carrier is a tunnel or a carrier frequency (earfcn) that a sector used to serve users.
# A base station, called an eNodeB (eNB) in 3GPP, may support multiple carriers on multiple sectors.
# In the case of an eNodeB, it may support up to 256 cells, with the final 8 bits of the 28-bit ECGI (full cell identity) indicating the Cell Identity within the eNodeB.
# https://www.finetopix.com/showthread.php/24559-Cell-Sector-Carrier
# An eNodeB usually contains 3 sectors (each antenna cover 120˚)
# Each sector is assigned a PCI - primary cell identification.
# ******************************************************************************************************

# --------------------- Util Functions ---------------------
HEADER1 = "Date,GPSLat,GPSLon,GPSSpeed,NetLat,NetLon,NetSpeed,RxRate,TxRate,DLBandwidth,ULBandwidth,LTE,LTE_RSRP,LTE_RSRQ,NR,NR_SSRSRP,NR_SSRSRQ,NR_CSIRSRP,NR_CSIRSRQ,Type,MNC,MCC,CID,PCI,SigStrength,earfcn,PCI1,SigStrength1,earfcn1"
LEN1 = 3
HEADER2 = "Date,GPSLat,GPSLon,GPSSpeed,RxRate,TxRate,DLBandwidth,ULBandwidth,MNC,MCC,CID,PCI,LTE_RSRP,LTE_RSRQ,NR_SSRSRP,NR_SSRSRQ,earfcn,PCI1,LTE_RSRP1,LTE_RSRQ1,earfcn1"
LEN2 = 4

def add_header(csv_input, csv_output, version='v2'):
    """
    Add headers if needed.
    """
    fp_input = open(csv_input, 'r')
    lines = fp_input.readlines()  # neglect '\n' when reading the file
    
    ### First round traversal to get the max length of columns
    max_row_size = 0
    for line in lines:
        row_size = len(line.split(','))
        if row_size > max_row_size:
            max_row_size = row_size
    
    ### Check if the column names exist
    need_header = True
    line_0 = lines[0]
    if line_0[:4] == 'Date':
        need_header = False
    
    ### Second round traversal to fill in the content to a new file
    fp_output = open(csv_output, 'w')
    print(csv_output)

    if version == 'v1':
        column_names = HEADER1
    else:
        column_names = HEADER2
    row_size = len(column_names.split(','))

    if need_header:
        for i in range(max_row_size - row_size):
            column_names = column_names + ',-'  # '-'
            # column_names = column_names + ','   # np.nan
        column_names = column_names + '\n'
        fp_output.write(column_names)
        
    for line in lines:
        row_size = len(line.split(','))
        append_line = line[:-1]
        for i in range(max_row_size - row_size):
            append_line = append_line + ',-'  # '-'
            # append_line = append_line + ','   # np.nan
        append_line = append_line + '\n'
        fp_output.write(append_line)
    fp_input.close()
    fp_output.close()
    return row_size, max_row_size

def rename_columns(df, version='v2'):
    ### rename CID
    df.rename(columns = {'Date':'Timestamp', 'CID':'ECI'}, inplace=True)
    ### reindex MNC & MCC
    curr_columns = df.columns.tolist()
    df = df.reindex(curr_columns[:curr_columns.index('MNC')] + \
                    ['MCC', 'MNC'] + \
                    curr_columns[curr_columns.index('ECI'):],
                    axis=1)
    return df

def anomaly_remove(df, version='v2'):
    """
    Remove the rows with abnormal data such as Nan, weird PLMN or ECI.
    """
    if version == 'v1':
        subset = "Type,MNC,MCC,ECI,PCI,SigStrength,earfcn,LTE_RSRP,LTE_RSRQ".split(',')
    else:
        subset = "MNC,MCC,ECI,PCI,LTE_RSRP,LTE_RSRQ,earfcn".split(',')
    print("------------------------------")
    print("detecting anomalies...")
    ### Check unexpected Nan
    print("----------------------")
    print("Unexpected Nan:")
    z = df[df[subset].isnull().any(axis=1)][['Timestamp', 'GPSLat', 'GPSLon'] + subset]
    if len(z) > 0:
        print('Index: {}'.format(z.index.values))
        print("rows with an unexpected Nan")
        print(z)
        print("removing {}...".format(z.index.values))
    else:
        print("No unexpected Nan occurs!")
    df = df.dropna(axis='index', how='any', subset=subset)
    ### Check abnormal PLMN or ECI
    print("----------------------")
    print("Abnormal PLMN or ECI:")
    z = df[(df['MCC'] == 0) | (df['MNC'] == 0)][['Timestamp', 'GPSLat', 'GPSLon'] + subset]
    if len(z) > 0:
        print('Index: {}'.format(z.index.values))
        print("rows with abnormal PLMN or ECI")
        print(z)
        print("removing {}...".format(z.index.values))
    else:
        print("No abnormal PLMN or ECI occurs!")
    df = df.loc[(df['MCC'] != 0) & (df['MNC'] != 0)]
    ### Reset index (retain the original indices)
    df = df.reset_index(drop=False)
    return df

def add_enb(df, version='v2'):
    """
    Add info of eNB and Cell ID.
    """
    print("------------------------------")
    print('adding eNB & Cell ID...')
    curr_columns = df.columns.tolist()
    df = df.reindex(curr_columns[:curr_columns.index('PCI')] + \
                    ['eNB', 'C.ID'] + \
                    curr_columns[curr_columns.index('PCI'):],
                    axis=1)
    for i in range(len(df)):
        if not pd.isna(df.loc[i, 'ECI']):
            df.loc[i, 'eNB'] = str(round(int(df.loc[i, 'ECI']) // 256))
            df.loc[i, 'C.ID'] = str(round(int(df.loc[i, 'ECI']) % 256))
    return df

def aggregate_neighb_pci(df, version='v2'):
    print("------------------------------")
    print("aggregating PCI & earfcn...")
    new_columns1 = ['neighb.num']
    new_columns2 = ['neighb.PCI.num', 'neighb.PCI', 'neighb.earfcn.num', 'neighb.earfcn']
    curr_columns = df.columns.tolist()
    subset = curr_columns[curr_columns.index('PCI1'):]
    df = df.reindex(curr_columns[:curr_columns.index('PCI1')] + \
                    new_columns1 + \
                    curr_columns[curr_columns.index('PCI1'):] + \
                    new_columns2,
                    axis=1)
    
    for i in tqdm(range(len(df))):
        z = df.loc[i, subset]
        z = z[~z.isin(['-'])]  # extract the columns that is not '-'
        # z = z[z.notnull()]     # extract the columns that is not np.nan

        ### Collect neighboring pci & earfcn list
        pci_set = set()
        earfcn_set = set()
        pci_list = []
        rsrp_list = []
        rsrq_list = []
        earfcn_list = []
        if version == 'v1':
            df.loc[i, "neighb.num"] = str(round(len(z) // LEN1))
            for j in range(0, len(z), LEN1):
                pci = int(z.iloc[j])
                rsrp = z.iloc[j+1]
                earfcn = int(z.iloc[j+2])
                pci_list.append(str(round(pci)))
                rsrp_list.append(rsrp)
                earfcn_list.append(str(round(earfcn)))
                if pci not in pci_set:
                    pci_set.add(pci)
                if earfcn not in earfcn_set:
                    earfcn_set.add(earfcn)
        else:
            df.loc[i, "neighb.num"] = str(round(len(z) // LEN2))
            for j in range(0, len(z), LEN2):
                pci = int(z.iloc[j])
                rsrp = z.iloc[j+1]
                rsrq = z.iloc[j+2]
                earfcn = int(z.iloc[j+3])
                pci_list.append(str(round(pci)))
                rsrp_list.append(rsrp)
                rsrq_list.append(rsrq)
                earfcn_list.append(str(round(earfcn)))
                if pci not in pci_set:
                    pci_set.add(pci)
                if earfcn not in earfcn_set:
                    earfcn_set.add(earfcn)
        pci_set = sorted(pci_set)
        earfcn_set = sorted(earfcn_set)

        ### Fill in the blank
        _len = round(len(pci_set))
        df.loc[i, 'neighb.PCI.num'] = str(_len)
        df.loc[i, 'neighb.PCI'] = '@'.join([str(round(s)) for s in pci_set])# if _len != 0 else '-'
        _len = round(len(earfcn_set))
        df.loc[i, 'neighb.earfcn.num'] = str(_len)
        df.loc[i, 'neighb.earfcn'] = '@'.join([str(round(s)) for s in earfcn_set])# if _len != 0 else '-'
        _len = len(pci_list)
        df.loc[i, 'PCI1'] = '@'.join(pci_list)# if _len != 0 else '-'
        df.loc[i, 'earfcn1'] = '@'.join(earfcn_list)# if _len != 0 else '-'
        if version == 'v1':
            df.loc[i, 'SigStrength1'] = '@'.join(rsrp_list)# if _len != 0 else '-'
        else:
            df.loc[i, 'LTE_RSRP1'] = '@'.join(rsrp_list)# if _len != 0 else '-'
            df.loc[i, 'LTE_RSRQ1'] = '@'.join(rsrq_list)# if _len != 0 else '-'
    ### remove useless columns
    curr_columns = df.columns.tolist()
    subset = curr_columns[:curr_columns.index('earfcn1')+1] + curr_columns[curr_columns.index('neighb.PCI.num'):]
    df = df[subset]
    ### reindex columns
    curr_columns = df.columns.tolist()
    if version == 'v1':
        move_columns = ['PCI', 'earfcn', 'SigStrength', 'neighb.num', 'PCI1', 'earfcn1', 'SigStrength1']
    else:
        move_columns = ['PCI', 'earfcn', 'LTE_RSRP', 'LTE_RSRQ', 'NR_SSRSRP', 'NR_SSRSRQ', 'neighb.num', 'PCI1', 'earfcn1', 'LTE_RSRP1', 'LTE_RSRQ1']
    df = df.reindex(curr_columns[:curr_columns.index('PCI')] + \
                    move_columns + \
                    curr_columns[curr_columns.index('neighb.PCI.num'):],
                    axis=1)
    return df

def analyze_handover(df, version='v2'):
    """
    Analyze handover and classify.
    """
    print("------------------------------")
    print("analyzing handover...")
    new_columns = ['Handover.state', 'Handover.band-change', 'Handover.type', 'Handover.duration(sec)']
    curr_columns = df.columns.tolist()
    df = df.reindex(curr_columns[:curr_columns.index('neighb.num')] + \
                    new_columns + \
                    curr_columns[curr_columns.index('neighb.num'):],
                    axis=1)
    ### initialize the previous info
    _enb = df.loc[0, 'eNB']
    _pci = df.loc[0, 'PCI']
    _earfcn = df.loc[0, 'earfcn']

    for i in tqdm(range(len(df))):
        # df.loc[i, 'Handover.state'] = '-'
        # df.loc[i, 'Handover.band-change'] = '-'
        # df.loc[i, 'Handover.type'] = '-'
        # df.loc[i, 'Handover.duration(sec)'] = '-'
        if (df.loc[i, 'PCI'] != _pci) or (df.loc[i, 'earfcn'] != _earfcn):
            ### after handover (current row)
            df.loc[i, 'Handover.state'] = 'end'
            ### starting handover (previous row)
            df.loc[i-1, 'Handover.state'] = 'start'
            df.loc[i-1, 'Handover.duration(sec)'] = (df.loc[i, 'Timestamp'] - df.loc[i-1, 'Timestamp']).seconds
            if df.loc[i, 'eNB'] != _enb:
                df.loc[i-1, 'Handover.type'] = 'inter-eNB'
            elif df.loc[i, 'PCI'] != _pci:
                df.loc[i-1, 'Handover.type'] = 'intra-eNB'
            elif df.loc[i, 'earfcn'] != _earfcn:
                df.loc[i-1, 'Handover.type'] = 'band-change-only'
            if df.loc[i, 'earfcn'] != _earfcn:
                df.loc[i-1, 'Handover.band-change'] = True
            else:
                df.loc[i-1, 'Handover.band-change'] = False
        _enb = df.loc[i, 'eNB']
        _pci = df.loc[i, 'PCI']
        _earfcn = df.loc[i, 'earfcn']
    return df

def cimon_preproc(fin, fmiddle, fout, version='v2'):
    row_size, max_row_size = add_header(fin, fmiddle, version)
    df = pd.read_csv(fmiddle, dtype=str)
    df = rename_columns(df, version)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = anomaly_remove(df, version)
    df = add_enb(df, version)
    df = aggregate_neighb_pci(df, version)
    df = analyze_handover(df, version)
    df.to_csv(fout, index=False)

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
        ### Check the input filename format, and whether users specify middle and final output filenames.
        # if "cimon" not in fin or not fin.endswith(".csv") or fin.endswith("_new.csv") or fin.endswith("_preproc.csv"):
        if not fin.endswith(".csv") or fin.endswith(("_new.csv", "_preproc.csv")):
            print("Input: '{}' does not meet the right format, the program is terminated.".format(fin))
            sys.exit()
        fmiddle = "{}_new.csv".format(fin[:-4])
        fout = "{}_preproc.csv".format(fin[:-4])
        ### preprocessing ...
        print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
        cimon_preproc(fin, fmiddle, fout, args.version)
        print()
        t.toc()  # Time elapsed since t.tic()
        sys.exit()
    
    # --------------------- (2) convert files in one directory (set arguments) ---------------------
    if args.indir:
        input_path = args.indir
        ### Check if the input directory exists
        if not os.path.isdir(input_path):
            print("FileExistsError: directory '{}' does not exists, the program is terminated.".format(input_path))
            sys.exit()
        output_path = args.outdir if args.outdir else input_path
        filenames = os.listdir(input_path)
        pprint(filenames)
        for filename in filenames:
            # if "cimon" not in fin or not fin.endswith(".csv") or fin.endswith("_new.csv") or fin.endswith("_preproc.csv"):
            if not filename.startswith("cimon") or not filename.endswith(".csv") or filename.endswith(("_new.csv", "_preproc.csv")):
                continue
            fin = os.path.join(input_path, filename)
            fmiddle = os.path.join(output_path, "{}_new.csv".format(filename[:-4]))
            fout = os.path.join(output_path, "{}_preproc.csv".format(filename[:-4]))
            makedir(output_path)
            ### preprocessing ...
            print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
            cimon_preproc(fin, fmiddle, fout, args.version)
        print()
        t.toc()  # Time elapsed since t.tic()
        sys.exit()
    
    # --------------------- (3) convert a batch of files (User Settings) ---------------------
    ### iteratively preprocess every cimon file
    for _exp, (_times, _rounds) in exps.items():
        ### Check if the directories exist
        exp_path = os.path.join(database, date, _exp)
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
        ### Check if cimon files exist, and then run preprocessing
        print(_exp)
        for j in range(_times):
            for i, dev in enumerate(devices):
                print(exp_dirs[i][j])
                dir = os.path.join(exp_dirs[i][j], "raw")
                filenames = os.listdir(dir)
                for filename in filenames:
                    # if "cimon" not in filename or not filename.endswith(".csv") or filename.endswith("_new.csv") or filename.endswith("_preproc.csv"):
                    if not filename.startswith("cimon") or not filename.endswith(".csv") or filename.endswith(("_new.csv", "_preproc.csv")):
                        continue
                    # print(filename)
                    fin = os.path.join(dir, filename)
                    fmiddle = os.path.join(dir, "..", "middle", "{}_new.csv".format(filename[:-4]))
                    fout = os.path.join(dir, "..", "data", "{}_preproc.csv".format(filename[:-4]))
                    makedir(os.path.join(dir, "..", "middle"))
                    makedir(os.path.join(dir, "..", "data"))
                    ### preprocessing ...
                    print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
                    cimon_preproc(fin, fmiddle, fout, cimon_version)
            print()
    t.toc()  # Time elapsed since t.tic()
