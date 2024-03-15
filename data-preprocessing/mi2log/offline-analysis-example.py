#!/usr/bin/python
# Filename: offline-analysis-example.py
import os
import sys
import traceback

"""
Offline analysis by replaying logs
"""

# Import MobileInsight modules
from mobile_insight.monitor import OfflineReplayer
from mobile_insight.analyzer import MsgLogger, NrRrcAnalyzer, LteRrcAnalyzer, WcdmaRrcAnalyzer, LteNasAnalyzer, UmtsNasAnalyzer, LteMacAnalyzer, LtePhyAnalyzer, LteMeasurementAnalyzer


if __name__ == "__main__":

    # Set path
    try: 
        dirpaths = [sys.argv[1]]
    except:
        dirpaths = ["/home/wmnlab/Desktop/testspace/data/0620/xm05",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm06",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm07",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm09",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm10",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm11",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm15",
                    "/home/wmnlab/Desktop/testspace/data/0620/xm16"]
    # Error recording list
    error_handling = []

    # Iteratively run analysis code
    for dirpath in dirpaths:
        filenames = os.listdir(dirpath)
        for filename in filenames:
            if not filename.endswith(".mi2log"):
                continue
            filepath = os.path.join(dirpath, filename)
            print("path:", filepath)
            try:
                # Initialize a monitor
                src = OfflineReplayer()
                # src.set_input_path("./logs/")
                src.set_input_path(filepath)
                src.enable_log_all()

                # src.enable_log("LTE_PHY_Serv_Cell_Measurement")
                # src.enable_log("5G_NR_RRC_OTA_Packet")
                # src.enable_log("LTE_RRC_OTA_Packet")
                # src.enable_log("LTE_NB1_ML1_GM_DCI_Info")

                logger = MsgLogger()
                logger.set_decode_format(MsgLogger.XML)
                logger.set_dump_type(MsgLogger.FILE_ONLY)
                # logger.save_decoded_msg_as("./test.txt")
                logger.save_decoded_msg_as(filepath + ".txt")
                logger.set_source(src)

                # # Analyzers
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

                # Start the monitoring
                src.run()
            except:
                # Record error message without halting the program
                error_handling.append((filepath, traceback.format_exc()))
    for item in error_handling:
        print("decoding of FILE: %s was interrupted. Error message:" % item[0])
        print(item[1])
