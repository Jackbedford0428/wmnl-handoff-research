###### parse_handover.py ######
# ============== instructions ==============
###### This file requires the txt file which is generated from offline_analysis.py and the mi2log file
###### The rows shows the information of each diag mode packets (dm_log_packet) from Mobile Insight 
###### The columns are indicators about whether a packet has the type of the message

from bs4 import BeautifulSoup
import sys
import os
import csv
import pandas as pd
import datetime as dt
from pprint import pprint
from pytictoc import TicToc

# ******************************* User Settings *******************************
database = "/home/wmnlab/D/database/"
# date = "2022-12-22"
dates = [
         "2023-02-04", 
         "2023-02-04#1",
         "2023-02-04#2",
         ]
devices = sorted([
    # "sm00",
    # "sm01",
    # "sm02",
    # "sm03",
    # "sm04",
    # "sm05",
    # "sm06",
    # "sm07",
    # "sm08",
    # "qc00",
    "qc01",
    "qc02",
    "qc03",
])
exps = {  # experiment_name: (number_of_experiment_rounds, list_of_experiment_round)
            # If the list is None, it will not list as directories.
            # If the list is empty, it will list all directories in the current directory by default.
            # If the number of experiment times != the length of existing directories of list, it would trigger warning and skip the directory.
    "_Bandlock_Udp_B3_B7_B8_RM500Q": (2, []),
    "_Bandlock_Udp_all_RM500Q": (2, []),
    # "tsync": (1, None),
    # "_Bandlock_Udp": (4, ["#01", "#02", "#03", "#04"]),
    # "_Bandlock_Udp": (4, ["#03", "#04", "#05", "#06"]),
    # "_Bandlock_Udp": (4, []),
    # "_Bandlock_Udp": (6, []),
    # "_Bandlock_Udp_B1_B3":  (6, []),
    # "_Bandlock_Udp_B3_B28": (2, []),
    # "_Bandlock_Udp_B28_B1": (2, []),
    # "_Bandlock_Udp_B1_B3": (4, []),
    # "_Bandlock_Udp_B3_B7": (4, []),
    # "_Bandlock_Udp_B7_B8": (4, []),
    # "_Bandlock_Udp_B8_B1": (4, []),
    # "_Mobile_Bandlock_Test": (1, None),
    # "_Modem_Phone_Comparative_Exeriments": (6, []),
}
# *****************************************************************************

# ******************************************************************************************************
# https://www.commresearch.com.tw/blog/ViewArticle.aspx?guid=384d9fe0-8d68-499e-85ad-09b421974670
# https://www.uj5u.com/qukuanlian/298167.html
# https://blog.csdn.net/travel_life/article/details/110194765
# https://www.telecomhall.net/t/what-is-endc-in-5g/12243
# https://rootmetrics.com/en-US/content/5g-faq-what-is-endc
# MCG (Master Node, MN) MeNB -----[Dual Connectivity, DC]----- SCG (Secondary Node, SN) SgNB
# Under MCG (Master Cell Group): PCell ---[Carrier Aggregation, CA]--- SCell.1 ---[CA]--- SCell.2
# Under SCG (Secondary Cell Group): PSCell ---[Carrier Aggregation, CA]--- SCell.1 ---[CA]--- SCell.2
# (Primary Cell, PCell), (Primary Secondary Cell, PSCell), (Secondary Cell, SCell)
# ******************************************************************************************************
# Successful Handover
# (1) lte_handover:       (eNB1)       -> (eNB2)            # lte_handoff_4G (in pure LTE)
# (2) SN_addition:        (eNB)        -> (eNB, gNB)        # nr_setup (addition, pure LTE to EN-DC)
# (3) endc_SN_change:     (eNB, gNB1)  -> (eNB, gNB2)       # nr_handoff_5G (in EN-DC)
# (4) SN_removal:         (eNB, gNB)   -> (eNB)             # nr_release (removal, EN-DC to pure LTE)
# (5) endc_MN_change:     (eNB1, gNB)  -> (eNB2, gNB)       # lte_handoff_5G (in EN-DC)
# (6) endc_MNSN_change:   (eNB1, gNB1) -> (eNB2, gNB2)      # nr_lte_handoff_5G (in EN-DC)
# (7) lte2endc_MN_change: (eNB1)       -> (eNB2, gNB)       # eNB_to_MN (pure LTE to EN-DC)
# (8) endc2lte_MN_change: (eNB1, gNB)  -> (eNB2)            # MN_to_eNB (EN-DC to pure LTE)
# ******************************************************************************************************
# Handover Failure
# (1) scg_failure:        gNB handover (SN change) failure  # scg_failure
# (2) radio_link_failure: eNB handover (MN change) failure  # reestablish_type2
# (3) nas_recovery:       re-establishment reject           # reestablish_type3
# ******************************************************************************************************

# **************************** Auxiliary Functions ****************************
def parse_handover(fin, fout):
    def nr_pci_track(mode=1):
        if mode:
            # if df.loc[i, "PCI"] == 65535:  ## 65535 is for samgsung phone.
            if df.loc[i, "PCI"] in [0, 65535]:  ## 65535 is for samgsung phone, 0 is for xiaomi phone.
                nr_pci = '-'
                nr_freq = '-'
            else:
                nr_pci = str(round(df.loc[i, "PCI"]))
                nr_freq = str(round(df.loc[i, "Freq"]))
        else:
            # if df.loc[i, "PCI"] == 65535:  ## 65535 is for samgsung phone.
            if df.loc[k, "PCI"] in [0, 65535]:  ## 65535 is for samgsung phone, 0 is for xiaomi phone.
                nr_pci = '-'
                nr_freq = '-'
            else:
                nr_pci = str(round(df.loc[k, "PCI"]))
                nr_freq = str(round(df.loc[k, "Freq"]))
        return nr_pci, nr_freq
    
    def eci_track(mode=1):
        if mode:
            eci = df.loc[i, "Cell Identity"]
        else:
            eci = df.loc[k, "Cell Identity"]
        return eci
    
    df = pd.read_csv(fin)
    # df.loc[:, 'Timestamp'] = pd.to_datetime(df.loc[:, 'Timestamp']) + dt.timedelta(hours=8)
    df['Timestamp'] = pd.to_datetime(df['Timestamp']) + dt.timedelta(hours=8)
    if len(df):
        exp_time = (df['Timestamp'].iloc[-1] - df['Timestamp'].iloc[0]).total_seconds()
    else:
        exp_time = 0

    ### add new columns
    newCols = ['handoff_type', 'handoff_state', 'handoff_duration', 'nr_PCI', 'nr_Freq', 'eNB.ID', 'CID']
    df = df.reindex(df.columns.tolist() + newCols, axis=1)
    
    ### filter by packet type
    # key = (df.loc[:, 'type_id'] == 'LTE_RRC_OTA_Packet') | (df.loc[:, 'type_id'] == '5G_NR_RRC_OTA_Packet') | (df.loc[:, 'type_id'] == 'LTE_RRC_Serv_Cell_Info')
    # df = df.loc[key]
    # df = df.reset_index(drop=True)

    ### Initialization
    # ePCI = earfcn = nr_PCI = nr_arfcn = '-'
    # eNB = CID = TAC = MCC = MNC = '-'

    ### Successful Handover
    lte_handover_list = []        # (eNB1)       -> (eNB2)            # lte_handoff_4G (in pure LTE)
    SN_addition_list = []         # (eNB)        -> (eNB, gNB)        # nr_setup (addition, pure LTE to EN-DC)
    SN_removal_list = []          # (eNB, gNB)   -> (eNB)             # nr_release (removal, EN-DC to pure LTE)
    endc_SN_change_list = []      # (eNB, gNB1)  -> (eNB, gNB2)       # nr_handoff_5G (in EN-DC)
    endc_MN_change_list = []      # (eNB1, gNB)  -> (eNB2, gNB)       # lte_handoff_5G (in EN-DC)
    endc_MNSN_change_list = []    # (eNB1, gNB1) -> (eNB2, gNB2)      # nr_lte_handoff_5G (in EN-DC)
    lte2endc_MN_change_list = []  # (eNB1)       -> (eNB2, gNB)       # eNB_to_MN (pure LTE to EN-DC)
    endc2lte_MN_change_list = []  # (eNB1, gNB)  -> (eNB2)            # MN_to_eNB (EN-DC to pure LTE)
    ### Handover Failure
    scg_failure_list = []         # gNB handover (SN change) failure  # scg_failure
    radio_link_failure_list = []  # eNB handover (MN change) failure  # reestablish_type2
    nas_recovery_list = []        # re-establishment reject           # reestablish_type3
    
    ### Initializing...
    nr_pci = '-'
    nr_freq = '-'
    post_nr_pci = '-'
    post_nr_freq = '-'
    eci = '-'
    handover_num = 0
    ### Successful handover
    nr_handover = 0
    nr_handover_start_index = None
    lte_handover = 0
    lte_handover_start_index = None
    nr_release = 0
    nr_release_start_index = None
    nr_reserved = 0
    ### Handover failure
    lte_failure = 0
    lte_failure_start_index = None

    for i in range(len(df)):
    
        if df.loc[i, "type_id"] == "5G_NR_RRC_OTA_Packet" and not nr_reserved:
            nr_pci, nr_freq = nr_pci_track()
            if df.loc[i, "nr-rrc.t304"]:
                post_nr_pci = df.loc[i, "nr_physCellId"]
                post_nr_freq = df.loc[i, "ssbFrequency"] if df.loc[i, "ssbFrequency"] else nr_freq
                nr_reserved = 1
            continue
        if df.loc[i, "type_id"] == "LTE_RRC_Serv_Cell_Info":
            eci = eci_track()
            continue
        
        if df.loc[i, "nr-rrc.t304"]:
            if nr_handover == 0:    
                nr_handover = 1
                nr_handover_start_index = i
        if df.loc[i, "lte-rrc.t304"]:
            if lte_handover == 0:
                lte_handover = 1
                lte_handover_start_index = i
        if df.loc[i, "nr-Config-r15: release (0)"]:
            if nr_release == 0:
                nr_release = 1
                nr_release_start_index = i
        if (nr_handover or lte_handover or nr_release) and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            handover_num += 1
        
        ### Successful handover message classifying
        ### Involves LTE handover and / or NR release
        if lte_handover and not nr_handover and not nr_release and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            lte_handover = 0
            nr_reserved = 0
            df.loc[lte_handover_start_index, 'handoff_type'] = 'lte_handover'
            df.loc[lte_handover_start_index, 'handoff_state'] = 'start'
            df.loc[lte_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            df.loc[lte_handover_start_index, 'nr_PCI'] = nr_pci
            df.loc[lte_handover_start_index, 'nr_Freq'] = nr_freq
            df.loc[lte_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[lte_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
            df.loc[i, 'handoff_type'] = 'lte_handover'
            df.loc[i, 'handoff_state'] = 'end'
            df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            k = i
            post_nr_pci, post_nr_freq = nr_pci, nr_freq
            while df.loc[k, 'type_id'] != "5G_NR_RRC_OTA_Packet":
                k += 1
                if k == len(df):
                    k = i
                    break
            if k != i:
                post_nr_pci, post_nr_freq = nr_pci_track(0)
                nr_pci, nr_freq = post_nr_pci, post_nr_freq
            df.loc[i, 'nr_PCI'] = post_nr_pci
            df.loc[i, 'nr_Freq'] = post_nr_freq
            k = i
            while df.loc[k, 'type_id'] != "LTE_RRC_Serv_Cell_Info":
                k += 1
                if k == len(df) or df.loc[k, 'lte-rrc.t304'] == 1:
                    k = i
                    break
            if k != i:
                eci = eci_track(0)
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            lte_handover_list.append((df.loc[lte_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
            # lte_handover_list.append(1)
        elif lte_handover and not nr_handover and nr_release and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            lte_handover = 0
            nr_release = 0
            nr_reserved = 0
            df.loc[lte_handover_start_index, 'handoff_type'] = 'endc2lte_MN_change'
            df.loc[lte_handover_start_index, 'handoff_state'] = 'start'
            df.loc[lte_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            df.loc[lte_handover_start_index, 'nr_PCI'] = nr_pci
            df.loc[lte_handover_start_index, 'nr_Freq'] = nr_freq
            df.loc[lte_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[lte_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
            df.loc[i, 'handoff_type'] = 'endc2lte_MN_change'
            df.loc[i, 'handoff_state'] = 'end'
            df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            df.loc[i, 'nr_PCI'] = post_nr_pci
            df.loc[i, 'nr_Freq'] = post_nr_freq
            k = i
            while df.loc[k, 'type_id'] != "LTE_RRC_Serv_Cell_Info":
                k += 1
                if k == len(df) or df.loc[k, 'lte-rrc.t304'] == 1:
                    k = i
                    break
            if k != i:
                eci = eci_track(0)
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            endc2lte_MN_change_list.append((df.loc[lte_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
            # endc2lte_MN_change_list.append(1)
            # nr_pci == '-'
        ### Involves only NR handover (including NR setup)
        elif nr_handover and not lte_handover and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            nr_handover = 0
            nr_reserved = 0
            if df.loc[nr_handover_start_index, "dualConnectivityPHR: setup (1)"]:  # This if-else statement classifies whether it is nr addition or nr handover
                df.loc[nr_handover_start_index, 'handoff_type'] = 'SN_addition'
                df.loc[nr_handover_start_index, 'handoff_state'] = 'start'
                df.loc[nr_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[nr_handover_start_index, 'nr_PCI'] = nr_pci
                df.loc[nr_handover_start_index, 'nr_Freq'] = nr_freq
                df.loc[nr_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[nr_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
                df.loc[i, 'handoff_type'] = 'SN_addition'
                df.loc[i, 'handoff_state'] = 'end'
                df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[i, 'nr_PCI'] = post_nr_pci
                df.loc[i, 'nr_Freq'] = post_nr_freq
                df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
                SN_addition_list.append((df.loc[nr_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
                # SN_addition_list.append(1)
            else:
                df.loc[nr_handover_start_index, 'handoff_type'] = 'endc_SN_change'
                df.loc[nr_handover_start_index, 'handoff_state'] = 'start'
                df.loc[nr_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[nr_handover_start_index, 'nr_PCI'] = nr_pci
                df.loc[nr_handover_start_index, 'nr_Freq'] = nr_freq
                df.loc[nr_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[nr_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
                df.loc[i, 'handoff_type'] = 'endc_SN_change'
                df.loc[i, 'handoff_state'] = 'end'
                df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[i, 'nr_PCI'] = post_nr_pci
                df.loc[i, 'nr_Freq'] = post_nr_freq
                df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
                endc_SN_change_list.append((df.loc[nr_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
                # endc_SN_change_list.append(1)
                
            ### Additional Judgement
            # if df.loc[nr_handover_start_index, "dualConnectivityPHR: setup (1)"] and nr_pci != None:
            #     print("Warning: dualConnectivityPHR setup may not mean nr cell addition", mi_file, i)
            # if df.loc[nr_handover_start_index, "dualConnectivityPHR: setup (1)"]==0 and not (nr_pci != None and nr_pci != df.loc[nr_handover_start_index, "nr_pci"]): 
            #     print("Warning: nr-rrc.t304 without dualConnectivityPHR setup may not mean nr cell handover", mi_file, i, nr_handover_start_index, df.loc[nr_handover_start_index, "nr_pci"], nr_pci)
            
            ### nr_pci update
            # nr_pci = df.loc[nr_handover_start_index, "nr_pci"]

        ### Involves both LTE & NR handover (including NR setup)
        elif lte_handover and nr_handover and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            lte_handover = 0
            nr_handover = 0
            nr_reserved = 0
            # print(type(nr_pci))
            # print(type(df.loc[lte_handover_start_index, "nr_physCellId"]))
            # if nr_pci == df.loc[lte_handover_start_index, "nr_pci"]:
            if nr_pci == str(round(df.loc[lte_handover_start_index, "nr_physCellId"])): 
                df.loc[lte_handover_start_index, 'handoff_type'] = 'endc_MN_change'
                df.loc[lte_handover_start_index, 'handoff_state'] = 'start'
                df.loc[lte_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[lte_handover_start_index, 'nr_PCI'] = nr_pci
                df.loc[lte_handover_start_index, 'nr_Freq'] = nr_freq
                df.loc[lte_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[lte_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
                df.loc[i, 'handoff_type'] = 'endc_MN_change'
                df.loc[i, 'handoff_state'] = 'end'
                df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
                df.loc[i, 'nr_PCI'] = nr_pci
                df.loc[i, 'nr_Freq'] = nr_freq
                k = i
                while df.loc[k, 'type_id'] != "LTE_RRC_Serv_Cell_Info":
                    k += 1
                    if k == len(df) or df.loc[k, 'lte-rrc.t304'] == 1:
                        k = i
                        break
                if k != i:
                    eci = eci_track(0)
                df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
                endc_MN_change_list.append((df.loc[lte_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
                # endc_MN_change_list.append(1)
            else:
                if df.loc[nr_handover_start_index, "dualConnectivityPHR: setup (1)"]:  # This if-else statement classifies whether it is nr addition or nr handover
                    df.loc[nr_handover_start_index, 'handoff_type'] = 'lte2endc_MN_change'
                    df.loc[nr_handover_start_index, 'handoff_state'] = 'start'
                    df.loc[nr_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                    df.loc[nr_handover_start_index, 'nr_PCI'] = nr_pci
                    df.loc[nr_handover_start_index, 'nr_Freq'] = nr_freq
                    df.loc[nr_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                    df.loc[nr_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
                    df.loc[i, 'handoff_type'] = 'lte2endc_MN_change'
                    df.loc[i, 'handoff_state'] = 'end'
                    df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_handover_start_index, "Timestamp"]).total_seconds()
                    df.loc[i, 'nr_PCI'] = post_nr_pci
                    df.loc[i, 'nr_Freq'] = post_nr_freq
                    k = i
                    while df.loc[k, 'type_id'] != "LTE_RRC_Serv_Cell_Info":
                        k += 1
                        if k == len(df) or df.loc[k, 'lte-rrc.t304'] == 1:
                            k = i
                            break
                    if k != i:
                        eci = eci_track(0)
                    df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                    df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
                    lte2endc_MN_change_list.append((df.loc[nr_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
                    # lte2endc_MN_change_list.append(1)
                else:
                    df.loc[lte_handover_start_index, 'handoff_type'] = 'endc_MNSN_change'
                    df.loc[lte_handover_start_index, 'handoff_state'] = 'start'
                    df.loc[lte_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
                    df.loc[lte_handover_start_index, 'nr_PCI'] = nr_pci
                    df.loc[lte_handover_start_index, 'nr_Freq'] = nr_freq
                    df.loc[lte_handover_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                    df.loc[lte_handover_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
                    df.loc[i, 'handoff_type'] = 'endc_MNSN_change'
                    df.loc[i, 'handoff_state'] = 'end'
                    df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
                    df.loc[i, 'nr_PCI'] = post_nr_pci
                    df.loc[i, 'nr_Freq'] = post_nr_freq
                    k = i
                    while df.loc[k, 'type_id'] != "LTE_RRC_Serv_Cell_Info":
                        k += 1
                        if k == len(df) or df.loc[k, 'lte-rrc.t304'] == 1:
                            k = i
                            break
                    if k != i:
                        eci = eci_track(0)
                    df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
                    df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
                    endc_MNSN_change_list.append((df.loc[lte_handover_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
                    # endc_MNSN_change_list.append(1)
            
            ### nr_pci update
            # nr_pci = df.loc[lte_handover_start_index, "nr_pci"]

        ### Involves only NR release
        # if nr_release and df.loc[i, "rrcConnectionReconfigurationComplete"]:
        elif not lte_handover and nr_release and df.loc[i, "rrcConnectionReconfigurationComplete"]:
            nr_release = 0
            nr_reserved = 0
            # if lte_handover:
            #     df.loc[lte_handover_start_index, 'handoff_type'] = 'endc2lte_MN_change'
            #     df.loc[lte_handover_start_index, 'handoff_state'] = 'start'
            #     df.loc[lte_handover_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            #     df.loc[i, 'handoff_type'] = 'endc2lte_MN_change'
            #     df.loc[i, 'handoff_state'] = 'end'
            #     df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[lte_handover_start_index, "Timestamp"]).total_seconds()
            #     endc2lte_MN_change_list.append(1)
            # else:
            df.loc[nr_release_start_index, 'handoff_type'] = 'SN_removal'
            df.loc[nr_release_start_index, 'handoff_state'] = 'start'
            df.loc[nr_release_start_index, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_release_start_index, "Timestamp"]).total_seconds()
            df.loc[nr_release_start_index, 'nr_PCI'] = nr_pci
            df.loc[nr_release_start_index, 'nr_Freq'] = nr_freq
            df.loc[nr_release_start_index, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[nr_release_start_index, 'CID'] = eci % 256 if eci != '-' else '-'
            df.loc[i, 'handoff_type'] = 'SN_removal'
            df.loc[i, 'handoff_state'] = 'end'
            df.loc[i, 'handoff_duration'] = (df.loc[i, "Timestamp"] - df.loc[nr_release_start_index, "Timestamp"]).total_seconds()
            df.loc[i, 'nr_PCI'] = post_nr_pci
            df.loc[i, 'nr_Freq'] = post_nr_freq
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            SN_removal_list.append((df.loc[nr_release_start_index, "Timestamp"], df.loc[i, "Timestamp"]))
            # SN_removal_list.append(1)

            ### reset nr_pci
            # nr_pci = '-'
        
        ### Handover failure classifying
        if df.loc[i, "scgFailureInformationNR-r15"]:  ## nr_failure
            df.loc[i, 'handoff_type'] = 'scg_failure'
            df.loc[i, 'handoff_state'] = 'trigger'
            df.loc[i, 'nr_PCI'] = df.loc[i, "nr_physCellId"] if df.loc[i, "nr_physCellId"] else nr_pci   # report the failed target nr_pci
            df.loc[i, 'nr_Freq'] = df.loc[i, "ssbFrequency"] if df.loc[i, "nr_physCellId"] else nr_freq  # report the failed target nr_freq
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            # df.loc[i, 'nr_PCI'] = nr_pci    # report the original nr_pci
            # df.loc[i, 'nr_Freq'] = nr_freq  # report the original nr_freq
            scg_failure_list.append((df.loc[i, "Timestamp"], df.loc[i, "Timestamp"]))
            # scg_failure_list.append(1)
        
        if df.loc[i, "rrcConnectionReestablishmentRequest"]:
            if lte_failure == 0:
                lte_failure = 1
                lte_failure_start_index = i
        
        if lte_failure and df.loc[i, "rrcConnectionReestablishmentComplete"]:
            lte_failure = 0
            df.loc[lte_failure_start_index, 'handoff_type'] = 'radio_link_failure'
            df.loc[lte_failure_start_index, 'handoff_state'] = 'trigger'
            df.loc[lte_failure_start_index, 'nr_PCI'] = nr_pci    # report the original nr_pci
            df.loc[lte_failure_start_index, 'nr_Freq'] = nr_freq  # report the original nr_freq
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            radio_link_failure_list.append((df.loc[lte_failure_start_index, "Timestamp"], df.loc[lte_failure_start_index, "Timestamp"]))
            # radio_link_failure_list.append(1)
        
        elif lte_failure and df.loc[i, "rrcConnectionReestablishmentReject"]:
            lte_failure = 0
            df.loc[lte_failure_start_index, 'handoff_type'] = 'nas_recovery'
            df.loc[lte_failure_start_index, 'handoff_state'] = 'trigger'
            df.loc[lte_failure_start_index, 'nr_PCI'] = nr_pci    # report the original nr_pci
            df.loc[lte_failure_start_index, 'nr_Freq'] = nr_freq  # report the original nr_freq
            df.loc[i, 'eNB.ID'] = eci // 256 if eci != '-' else '-'
            df.loc[i, 'CID'] = eci % 256 if eci != '-' else '-'
            nas_recovery_list.append((df.loc[lte_failure_start_index, "Timestamp"], df.loc[lte_failure_start_index, "Timestamp"]))
            # nas_recovery_list.append(1)

    ### Handover Statistics
    ss = "lte_handover, SN_addition, SN_removal, endc_SN_change, endc_MN_change, endc_MNSN_change, lte2endc_MN_change, endc2lte_MN_change, scg_failure, radio_link_failure, nas_recovery, succ_handoff, fail_handoff, overall_handoff"
    print(ss)
    ss = ss.split(', ')
    event_type_list = [lte_handover_list, SN_addition_list, SN_removal_list, endc_SN_change_list, endc_MN_change_list, endc_MNSN_change_list, lte2endc_MN_change_list, endc2lte_MN_change_list, scg_failure_list, radio_link_failure_list, nas_recovery_list]
    ss2 = []
    for item in event_type_list:
        ss2.append(len(item))
    ss2.append(sum(ss2[:8]))
    ss2.append(sum(ss2[8:11]))
    ss2.append(sum(ss2[:11]))
    print(', '.join([str(s) for s in ss2]))

    ### select subset columns
    subset = [
        'Timestamp', 'type_id', 'handoff_type', 'handoff_state', 'handoff_duration', 
        'eNB.ID', 'CID', 'PCI', 'Freq', 'nr_PCI', 'nr_Freq',
        # 'UL_DL',
        # 'measurementReport',                        # 手機給基地台資訊
        # 'rrcConnectionReconfiguration',             # 基地台叫手機做的事情，會包住 nr-rrc.t304 或 lte-rrc.t304 或 nr-Config-r15: release (0) 或 nr-Config-r15: setup (1)
        # 'rrcConnectionReestablishmentRequest',      # eNB 斷線，手機要求重連
        # 'rrcConnectionReestablishment',             # eNB 斷線，基地台提供重連資訊
        # 'rrcConnectionReestablishmentReject',       # 目標基地台沒有準備好，無法重連
        # 'rrcConnectionSetup',                       # 手機從 idle 到 active 或 inactive 到 active
        # 'rrcConnectionSetupComplete',               # 手機跟基地台說：我進到 active 啦！
        # 'lte-rrc.nr_SecondaryCellGroupConfig_r15',  # MCG (LTE), SCG (NR)
        # 'rrcConnectionReconfigurationComplete',     # 手機跟基地台說：我做完了！
        # 'scgFailureInformationNR-r15',              # gNB 斷線 (request)
        # 'nr-rrc.t304',                              # NR Handover
        # 'lte-rrc.t304',                             # LTE Handover
        # 'nr-Config-r15: release (0)',               # 5G 切 4G
        # 'nr-Config-r15: setup (1)',                 # 4G 切 5G
        # 'dualConnectivityPHR: setup (1)',           # LTE to EN-DC (NR Setup)
        # 'dualConnectivityPHR: release (0)',         # EN-DC to LTE (NR Release)
        # 'nr-rrc.RRCReconfiguration_element',
        # 'nr-rrc.eventA',                            # Handover Policy
        # 'nr-rrc.spCellConfig_element',
        # 'lte-rrc.targetPhysCellId',
    ]
    df = df[subset].loc[df.loc[:, "handoff_type"].notnull()]
    df.rename(columns={'Freq':'EARFCN', 'nr_Freq':'NR_ARFCN', 'nr_PCI':'NR_PCI'}, inplace=True)
    df.to_csv(fout, index=False)
    return ss+['experiment_time(sec)'], ss2+[exp_time]
    # return [lte_handover_list, SN_addition_list, SN_removal_list, endc_SN_change_list, 
    #         endc_MN_change_list, endc_MNSN_change_list, lte2endc_MN_change_list, 
    #         endc2lte_MN_change_list, scg_failure_list, radio_link_failure_list, 
    #         nas_recovery_list]

def add_info(df, fout):
    handoff_types = df['handoff_type'].array
    handoff_states = df['handoff_state'].array
    LTE_PCIs = df['PCI'].array
    eNBs = df['eNB.ID'].array
    EARFCNs = df['EARFCN'].array
    NR_PCIs = df['NR_PCI'].array
    NR_ARFCNs = df['NR_ARFCN'].array
    handoff_types_1 = [[] for i in range(len(df))]
    for i, (handoff_type, handoff_state, pci, enb, earfcn, nr_pci, nr_arfcn) in enumerate(zip(handoff_types, handoff_states, LTE_PCIs, eNBs, EARFCNs, NR_PCIs, NR_ARFCNs)):
        if handoff_type in ['SN_addition', 'SN_removal', 'endc_SN_change']:
            handoff_types_1[i].append('SN_change')
            continue
        if handoff_state == 'start':
            tmp_id = i
            tmp_pci = pci
            tmp_enb = enb
            tmp_earfcn = earfcn
            tmp_nr_pci = nr_pci
            tmp_nr_arfcn = nr_arfcn
        elif handoff_state == 'trigger':
            continue
        elif handoff_state == 'end':
            ## sn-change
            if nr_pci != tmp_nr_pci or nr_arfcn != tmp_nr_arfcn:
                handoff_types_1[i].append('SN_change')
                handoff_types_1[tmp_id].append('SN_change')
            ## Intra-freq
            if earfcn == tmp_earfcn:
                handoff_types_1[i].append('Intra_freq')
                handoff_types_1[tmp_id].append('Intra_freq')
            else:
                handoff_types_1[i].append('Inter_freq')
                handoff_types_1[tmp_id].append('Inter_freq')
            ### Intra-BS
            if pci == tmp_pci:
                handoff_types_1[i].append('Intra_sector')
                handoff_types_1[tmp_id].append('Intra_sector')
            elif enb == tmp_enb:
                handoff_types_1[i].append('Intra_eNB')
                handoff_types_1[tmp_id].append('Intra_eNB')
            else:
                handoff_types_1[i].append('Inter_eNB')
                handoff_types_1[tmp_id].append('Inter_eNB')
    handoff_types_1 = ['+'.join(item) for item in handoff_types_1]
    ### add information
    df = df.join(pd.DataFrame({'handoff_type.1' : handoff_types_1}))
    df.to_csv(fout, index=False)
    return df
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
            if filename.startswith(tags) and filename.endswith("_rrc.csv"):
                files_collection.append(filename)
        return files_collection
    
    def main():
        files_collection = fgetter()
        if len(files_collection) == 0:
            print("No candidate file.")
        for filename in files_collection:
            fin = os.path.join(source_dir, filename)
            fout1 = os.path.join(target_dir1, "diag_log_ho-info.csv")
            fout2 = os.path.join(target_dir2, "diag_log_ho-statistics.csv")
            print(">>>>> decode from '{}' into '{}'...".format(fin, fout1))
            handover_type, handover_stats = parse_handover(fin, fout1)
            df = pd.read_csv(fout1)
            add_info(df, fout1)
            with open(fout2, "w", newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(handover_type)
                writer.writerow(handover_stats)
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
                    target_dir1 = os.path.join(database, date, expr, dev)
                    target_dir2 = os.path.join(database, date, expr, dev)
                    makedir(target_dir1)
                    makedir(target_dir2)
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
                    source_dir = os.path.join(database, date, expr, dev, trace, "data")
                    target_dir1 = os.path.join(database, date, expr, dev, trace, "data")
                    target_dir2 = os.path.join(database, date, expr, dev, trace, "statistics")
                    makedir(target_dir1)
                    makedir(target_dir2)
                    filenames = os.listdir(source_dir)
                    main()
    t.toc()  # Time elapsed since t.tic()
    # *****************************************************************************




    # t = TicToc()  # create instance of class
    # t.tic()  # Start timer
    # # --------------------- (3) decode a batch of files (User Settings) ---------------------
    # ### iteratively decode parse every diag_log_rrc.csv file
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
    #     ### Check if a diag_log_rrc.csv file exists, and then run decoding
    #     print(_exp)
    #     for j in range(_times):
    #         for i, dev in enumerate(devices):
    #             print(exp_dirs[i][j])
    #             dir = os.path.join(exp_dirs[i][j], "data")
    #             filenames = os.listdir(dir)
    #             for filename in filenames:
    #                 # if "diag_log" not in filename or not filename.endswith(".mi2log"):
    #                 if not filename.startswith("diag_log") or not filename.endswith("_rrc.csv"):
    #                     continue
    #                 # print(filename)
    #                 fin = os.path.join(dir, filename)
    #                 # fout = os.path.join(dir, "..", "analysis", "{}_parse-ho.csv".format(filename[:-4]))
    #                 fout = os.path.join(dir, "..", "analysis", "diag_log_ho-info.csv")
    #                 makedir(os.path.join(dir, "..", "analysis"))
    #                 ### decoding ...
    #                 print(">>>>> decode from '{}' into '{}'...".format(fin, fout))
    #                 handover_type, handover_stats = parse_handover(fin, fout)
    #                 # with open(os.path.join(dir, "..", "analysis", "{}_ho-statistics.csv".format(filename[:-4])), "w", newline='') as fp:
    #                 with open(os.path.join(dir, "..", "analysis", "diag_log_ho-statistics.csv"), "w", newline='') as fp:
    #                     writer = csv.writer(fp)
    #                     writer.writerow(handover_type)
    #                     writer.writerow(handover_stats)
    #                 df = pd.read_csv(fout)
    #                 add_info(df, fout)
    #             # sys.exit()
    #         print()
    # t.toc()
