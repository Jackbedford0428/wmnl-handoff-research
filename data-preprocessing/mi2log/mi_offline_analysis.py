#!/usr/bin/python3
### Filename: mi_offline_analysis.py

"""
Offline analysis by replaying logs, modified from 'MobileInsight-6.0.0/examples/offline-analysis-example.py'
Installation of MobileInsight package is needed.

Usages:

(1) decode only one files. It would decode inplace and change the suffix from '.mi2log' into '.txt'.
$ python3 mi_offline_analysis.py -i [input_filepath]

(2) decode files in one directory, If you do not set [output_dirpath], it would decode inplace.
$ python3 mi_offline_analysis.py -D [input_dirpath]
$ python3 mi_offline_analysis.py -D [input_dirpath] -O [output_dirpath]

(3) decode a batch of files => go to Users Settings and modify.
$ python3 mi_offline_analysis.py

Author: Yuan-Jye Chen
Update: Yuan-Jye Chen 2022-10-06
"""

"""
    Future Development Plan
        (1) No plans for now.
    
"""
import os
import sys
import argparse
import traceback
from pprint import pprint
from pytictoc import TicToc

### Import MobileInsight modules
from mobile_insight.monitor import OfflineReplayer
from mobile_insight.analyzer import MsgLogger, NrRrcAnalyzer, LteRrcAnalyzer, WcdmaRrcAnalyzer, LteNasAnalyzer, UmtsNasAnalyzer, LteMacAnalyzer, LtePhyAnalyzer, LteMeasurementAnalyzer

__all__ = [
    'mi_decode',
    'error_handling'
]

# --------------------- Arguments ---------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--input", type=str,
#                     help="input filepath")
# parser.add_argument("-D", "--indir", type=str,
#                     help="input dirctory path")
# parser.add_argument("-O", "--outdir", type=str,
#                     help="output dirctory path")
# args = parser.parse_args()

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
# date = "2022-12-26"
dates = [
         "2023-05-07",
]
devices = sorted([
    "sm00",
    "sm01",
    "sm02",
    "sm03",
    "sm04",
    "sm05",
    "sm06",
    "sm07",
    # "sm08",
    # "qc00",
    # "qc01",
    # "qc02",
    # "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    "_Bandlock_8_Schemes_Phone": (7, ["#{:02d}".format(i+1) for i in range(7)]),
}
# *****************************************************************************

# **************************** Auxiliary Functions ****************************
def mi_decode(fin, fout):
    try:
        ### Initialize a monitor
        src = OfflineReplayer()
        # src.set_input_path("./logs/")
        src.set_input_path(fin)
        src.enable_log_all()

        # src.enable_log("LTE_PHY_Serv_Cell_Measurement")
        # src.enable_log("5G_NR_RRC_OTA_Packet")
        # src.enable_log("LTE_RRC_OTA_Packet")
        # src.enable_log("LTE_NB1_ML1_GM_DCI_Info")

        logger = MsgLogger()
        logger.set_decode_format(MsgLogger.XML)
        logger.set_dump_type(MsgLogger.FILE_ONLY)
        logger.save_decoded_msg_as(fout)
        logger.set_source(src)

        ### Analyzers
        nr_rrc_analyzer = NrRrcAnalyzer()
        nr_rrc_analyzer.set_source(src)  # bind with the monitor

        lte_rrc_analyzer = LteRrcAnalyzer()
        lte_rrc_analyzer.set_source(src)  # bind with the monitor

        wcdma_rrc_analyzer = WcdmaRrcAnalyzer()
        wcdma_rrc_analyzer.set_source(src)  # bind with the monitor

        # lte_nas_analyzer = LteNasAnalyzer()
        # lte_nas_analyzer.set_source(src)

        # umts_nas_analyzer = UmtsNasAnalyzer()
        # umts_nas_analyzer.set_source(src)

        lte_mac_analyzer = LteMacAnalyzer()
        lte_mac_analyzer.set_source(src)

        lte_phy_analyzer = LtePhyAnalyzer()
        lte_phy_analyzer.set_source(src)

        lte_meas_analyzer = LteMeasurementAnalyzer()
        lte_meas_analyzer.set_source(src)

        # print lte_meas_analyzer.get_rsrp_list() 
        # print lte_meas_analyzer.get_rsrq_list()

        ### Start the monitoring
        src.run()
    except:
        ### Record error message without halting the program
        return (fin, fout, traceback.format_exc())
    return (fin, fout, None)
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
# *****************************************************************************


if __name__ == "__main__":
    def fgetter():
        files_collection = []
        tags = "diag_log"
        for filename in filenames:
            if filename.startswith(tags) and filename.endswith(".mi2log"):
                files_collection.append(filename)
        return files_collection
    
    def main():
        files_collection = fgetter()
        if len(files_collection) == 0:
            print("No candidate file.")
        for filename in files_collection:
            fin = os.path.join(source_dir, filename)
            fout = os.path.join(target_dir, "{}.txt".format(filename[:-7]))
            print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
            err_handle = mi_decode(fin, fout)
            err_handles.append(err_handle)
        print()
    
    # ******************************* Check Files *********************************
    for date in dates:
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
    for date in dates:
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
                    source_dir = os.path.join(database, date, expr, dev, trace, "raw")
                    target_dir = os.path.join(database, date, expr, dev, trace, "middle")
                    makedir(target_dir)
                    filenames = os.listdir(source_dir)
                    main()
    ### Check errors
    flag = False
    for err_handle in err_handles:
        flag = error_handling(err_handle)
    if not flag and err_handles:
        print("**************************************************")
        print("No error occurs!!")
        print("**************************************************")
    t.toc()  # Time elapsed since t.tic()
    # *****************************************************************************



    # t = TicToc()  # create instance of class
    # t.tic()  # Start timer
    # # --------------------- (1) decode only one file (set arguments) ---------------------
    # if args.input:
    #     fin = args.input
    #     ### Check the input filename format, and whether users specify output filename.
    #     if not fin.endswith(".mi2log"):
    #         print("Input: '{}' does not endswith 'mi2log', the program is terminated.".format(fin))
    #         sys.exit()
    #     fout = "{}.txt".format(fin[:-7])
    #     ### decoding ...
    #     print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
    #     err_handle = mi_decode(fin, fout)
    #     flag = error_handling(err_handle)
    #     print()
    #     if not flag:
    #         print("**************************************************")
    #         print("No error occurs!!")
    #         print("**************************************************")
    #     t.toc()
    #     sys.exit()

    # # --------------------- (2) decode files in one directory (set arguments) ---------------------
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
    #         fin = os.path.join(input_path, filename)
    #         # if "diag_log" not in fin or not fin.endswith(".mi2log"):
    #         if not fin.startswith("diag_log") or not fin.endswith(".mi2log"):
    #             continue
    #         fout = os.path.join(output_path, "{}.txt".format(filename[:-7]))
    #         makedir(output_path)
    #         ### decoding ...
    #         print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
    #         err_handle = mi_decode(fin, fout)
    #         err_handles.append(err_handle)
    #     print()
    #     ### Check errors
    #     flag = False
    #     for err_handle in err_handles:
    #         flag = error_handling(err_handle)
    #     if not flag:
    #         print("**************************************************")
    #         print("No error occurs!!")
    #         print("**************************************************")
    #     t.toc()
    #     sys.exit()

    # # --------------------- (3) decode a batch of files (User Settings) ---------------------
    # err_handles = []
    # ### iteratively decode every mi2log file
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
    #     ### Check if a mi2log file exists, and then run decoding
    #     print(_exp)
    #     for j in range(_times):
    #         for i, dev in enumerate(devices):
    #             print(exp_dirs[i][j])
    #             dir = os.path.join(exp_dirs[i][j], "raw")
    #             filenames = os.listdir(dir)
    #             for filename in filenames:
    #                 # if "diag_log" not in filename or not filename.endswith(".mi2log"):
    #                 if not filename.startswith("diag_log") or not filename.endswith(".mi2log"):
    #                     continue
    #                 fin = os.path.join(dir, filename)
    #                 fout = os.path.join(dir, "..", "data", "{}.txt".format(filename[:-7]))
    #                 makedir(os.path.join(dir, "..", "data"))
    #                 ### decoding ...
    #                 print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
    #                 err_handle = mi_decode(fin, fout)
    #                 err_handles.append(err_handle)
    #         print()
    # ### Check errors
    # flag = False
    # for err_handle in err_handles:
    #     flag = error_handling(err_handle)
    # if not flag and err_handles:
    #     print("**************************************************")
    #     print("No error occurs!!")
    #     print("**************************************************")
    # t.toc()
