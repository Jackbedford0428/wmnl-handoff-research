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
def xml_to_csv_nr_ml1(fin, fout):
    f = open(fin, encoding="utf-8")
    f2 = open(fout, 'w') # _nr_ml1.csv
    # print("nr_ml1 >>>>>")
    # Writing the column names...
    # -------------------------------------------------
    f2.write(",".join(["time", "type_id",
        "Raster ARFCN",
        "Num Cells",
        "Serving Cell Index",
        "Serving Cell PCI",
        "PCI1",
        "RSRP1",
        "RSRP2",

        ])+'\n')

    l = f.readline()

    while l:
        if r"<dm_log_packet>" in l:
            soup = BeautifulSoup(l, 'html.parser')
            timestamp = soup.find(key="timestamp").get_text()
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

                f2.write(",".join([timestamp, type_id, arfcn, num_cells, serving_cell_idex, serving_cell_pci] + A)+'\n')
            
            else: # 只處理nr_ml1資料過濾其他type
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
            fout = os.path.join(target_dir, "{}_nr_ml1.csv".format(filename[:-4]))
            print(">>>>> convert from '{}' into '{}'...".format(fin, fout))
            xml_to_csv_nr_ml1(fin, fout)
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
