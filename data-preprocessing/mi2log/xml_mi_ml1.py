######xml_mi.py#########
#==============instructions==============
######This file requires the txt file which is generated from offline_analysis.py and the mi2log file
######The rows shows the information of each diag mode packets (dm_log_packet) from Mobile Insight 
######The columns are indicators about whether a packet has the type of the message

from bs4 import BeautifulSoup
import sys
import os
from pprint import pprint
import argparse
from pytictoc import TicToc
from itertools import chain

# --------------------- Arguments ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str,
                    help="input filepath")
parser.add_argument("-D", "--indir", type=str,
                    help="input dirctory path")
parser.add_argument("-O", "--outdir", type=str,
                    help="output dirctory path")
args = parser.parse_args()

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
date = "2022-11-29"
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
    "_Bandlock_Udp_B1_B3":  (4, []),
    "_Bandlock_Udp_B3_B28": (4, []),
    "_Bandlock_Udp_B28_B1": (4, []),
}
# *****************************************************************************

# **************************** Auxiliary Functions ****************************
def xml_to_csv_ml1(fin, fout):
    f = open(fin, encoding="utf-8")
    f2 = open(fout, 'w') # _ml1.csv
    # print("ml1 >>>>>")
    # Writing the column names...
    # -------------------------------------------------
    f2.write(",".join(["time", "type_id",
        "PCI",
        "RSRP(dBm)",
        "RSRQ(dB)",
        "Serving Cell Index",
        "EARFCN",
        "Number of Neighbor Cells",
        "Number of Detected Cells",
        "PCI1",
        "LTE_RSRP1",
        "LTE_RSRQ1"
        ])+'\n')

    l = f.readline()

    #For each dm_log_packet, we will check that whether strings in type_list are shown in it.
    #If yes, type_code will record what types in type_list are shown in the packet.
    #-------------------------------------------------
    type_list = [
        
    ]

    while l:
        if r"<dm_log_packet>" in l:
            # type_code = ["0"] * len(type_list)
            
            soup = BeautifulSoup(l, 'html.parser')
            timestamp = soup.find(key="timestamp").get_text()
            type_id = soup.find(key="type_id").get_text()

            try:
                PCI = soup.find(key="Serving Physical Cell ID").get_text() ## This is current serving cell PCI.
            except:
                PCI = "-"

            if type_id == 'LTE_PHY_Connected_Mode_Intra_Freq_Meas':
                rsrps = [rsrp.get_text() for rsrp in soup.findAll(key="RSRP(dBm)")]
                rsrqs = [rsrq.get_text() for rsrq in soup.findAll(key="RSRQ(dB)")]
                serving_cell = soup.find(key="Serving Cell Index").get_text()
                earfcn = soup.find(key="E-ARFCN").get_text()
                n_nei_c = soup.find(key="Number of Neighbor Cells").get_text()
                n_det_c = soup.find(key="Number of Detected Cells").get_text()
                PCIs = [pci.get_text() for pci in soup.findAll(key="Physical Cell ID")] ## This is neighbor measured cells PCI.
                if int(n_det_c) != 0:
                    PCIs = PCIs[:-int(n_det_c)]
                A = [[PCIs[i], rsrps[i+1], rsrqs[i+1]] for i in range(len(PCIs))] ## Information of neighbor cell
                A = list(chain.from_iterable(A))
                f2.write(",".join([timestamp, type_id, PCI, rsrps[0], rsrqs[0], serving_cell, earfcn, n_nei_c, n_det_c] + A)+'\n')
            elif type_id == 'LTE_PHY_Connected_Mode_Neighbor_Measurement': # or type_id == 'LTE_PHY_Serv_Cell_Measurement': ## 無法parse暫時忽略
                f2.write(",".join([timestamp, type_id, PCI, '-', '-', '-', '-', '-', '-']  )+'\n')
                pass
            else: # 只處理ml1資料過濾其他type
                while l and r"</dm_log_packet>" not in l:
                    l = f.readline()

            l = f.readline()

        else:
            print(l,"Error!")
            break 
            
    f2.close()
    f.close()
# *****************************************************************************

# ****************************** Utils Functions ******************************
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
# *****************************************************************************


if __name__ == "__main__":
    def fgetter():
        files_collection = []
        tags = "diag_log"
        for filename in filenames:
            if filename.startswith(tags) and filename.endswith(".txt"):
                files_collection.append(filename)
        return files_collection
    
    def main():
        files_collection = fgetter()
        if len(files_collection) == 0:
            print("No candidate file.")
        for filename in files_collection:
            fin = os.path.join(source_dir, filename)
            fout = os.path.join(target_dir, "{}_ml1.csv".format(filename[:-4]))
            print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
            xml_to_csv_ml1(fin, fout)
        print()

    # ******************************* Check Files *********************************
    for expr, (times, traces) in exps.items():
        print(os.path.join(database, date, expr))
        for dev in devices:
            if not os.path.isdir(os.path.join(database, date, expr, dev)):
                print("|___ {} does not exist.".format(os.path.join(database, date, expr, dev)))
                continue
            
            print("|___", os.path.join(database, date, expr, dev))
            if traces == None:
                # print(os.path.join(database, date, expr, dev))
                continue
            elif len(traces) == 0:
                traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
            
            print("|    ", times)
            traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
            if len(traces) != times:
                print("***************************************************************************************")
                print("Warning: the number of traces does not match the specified number of experiment times.")
                print("***************************************************************************************")
            for trace in traces:
                print("|    |___", os.path.join(database, date, expr, dev, trace))
        print()
    # *****************************************************************************

    # ******************************** Processing *********************************
    t = TicToc()  # create instance of class
    t.tic()       # Start timer
    err_handles = []
    for expr, (times, traces) in exps.items():
        for dev in devices:
            if not os.path.isdir(os.path.join(database, date, expr, dev)):
                print("{} does not exist.\n".format(os.path.join(database, date, expr, dev)))
                continue

            if traces == None:
                print("------------------------------------------")
                print(date, expr, dev)
                print("------------------------------------------")
                source_dir = os.path.join(database, date, expr, dev)
                target_dir = os.path.join(database, date, expr, dev)
                makedir(target_dir)
                filenames = os.listdir(source_dir)
                main()
                continue
            elif len(traces) == 0:
                traces = sorted(os.listdir(os.path.join(database, date, expr, dev)))
            
            traces = [trace for trace in traces if os.path.isdir(os.path.join(database, date, expr, dev, trace))]
            for trace in traces:
                print("------------------------------------------")
                print(date, expr, dev, trace)
                print("------------------------------------------")
                source_dir = os.path.join(database, date, expr, dev, trace, "middle")
                target_dir = os.path.join(database, date, expr, dev, trace, "middle")
                makedir(target_dir)
                filenames = os.listdir(source_dir)
                main()
    t.toc()  # Time elapsed since t.tic()
    # *****************************************************************************




    # t = TicToc()  # create instance of class
    # t.tic()  # Start timer
    # # --------------------- (1) convert only one file (set arguments) ---------------------
    # if args.input:
    #     fin = args.input
    #     ### Check the input filename format, and whether users specify output filename.
    #     if not fin.endswith(".txt"):
    #         print("Input: '{}' does not endswith 'txt', the program is terminated.".format(fin))
    #         sys.exit()
    #     fout = "{}_rrc.csv".format(fin[:-4])
    #     ### decoding ...
    #     print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
    #     xml_to_csv_rrc(fin, fout)
    #     print()
    #     t.toc()
    #     sys.exit()

    # # --------------------- (2) convert files in one directory (set arguments) ---------------------
    # if args.indir:
    #     err_handles = []
    #     input_path = args.indir
    #     ### Check if the input directory exists
    #     if not os.path.isdir(input_path):
    #         print("FileExistsError: directory '{}' does not exists, the program is terminated.".format(input_path))
    #         sys.exit()
    #     output_path = args.outdir if args.outdir else input_path
    #     filenames = os.listdir(input_path)
    #     pprint(filenames)
    #     for filename in filenames:
    #         # if not filename.endswith(".pcap"):
    #         if not filename.startswith("diag_log") or not filename.endswith(".txt"):
    #             continue
    #         fin = os.path.join(input_path, filename)
    #         fout = os.path.join(output_path, "{}_rrc.csv".format(filename[:-4]))
    #         makedir(output_path)
    #         ### decoding ...
    #         print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
    #         xml_to_csv_rrc(fin, fout)
    #     print()
    #     t.toc()
    #     sys.exit()

    # # --------------------- (3) decode a batch of files (User Settings) ---------------------
    # ### iteratively decode every diag_log.txt file
    # for _exp, (_times, _rounds) in Exp_Name.items():
    #     ### Check if the directories exist
    #     exp_path = os.path.join(db_path, _exp)
    #     print(exp_path)
    #     exp_dirs = []
    #     for i, dev in enumerate(devices):
    #         if _rounds:
    #             exp_dirs.append([os.path.join(exp_path, dev, _round) for _round in _rounds])
    #         else:
    #             _rounds = sorted(os.listdir(os.path.join(exp_path, dev)))
    #             exp_dirs.append([os.path.join(exp_path, dev, item) for item in _rounds])
    #         exp_dirs[i] = [item for item in exp_dirs[i] if os.path.isdir(item)]
    #         print(_times)
    #         pprint(exp_dirs[i])
    #         if len(exp_dirs[i]) != _times:
    #             print("************************************************************************************************")
    #             print("Warning: the number of directories does not match your specific number of experiment times.")
    #             print("************************************************************************************************")
    #             print()
    #             sys.exit()
    #     print()
    #     ### Check if a diag_log.txt file exists, and then run decoding
    #     print(_exp)
    #     for j in range(_times):
    #         for i, dev in enumerate(devices):
    #             print(exp_dirs[i][j])
    #             dir = os.path.join(exp_dirs[i][j], "data")
    #             filenames = os.listdir(dir)
    #             for filename in filenames:
    #                 # if "diag_log" not in filename or not filename.endswith(".mi2log"):
    #                 if not filename.startswith("diag_log") or not filename.endswith(".txt"):
    #                     continue
    #                 # print(filename)
    #                 fin = os.path.join(dir, filename)
    #                 fout = os.path.join(dir, "{}_rrc.csv".format(filename[:-4]))
    #                 # makedir(os.path.join(dir, "..", "data"))
    #                 ### decoding ...
    #                 print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
    #                 xml_to_csv_rrc(fin, fout)
    #         print()
    # t.toc()
