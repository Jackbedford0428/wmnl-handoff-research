#!/usr/bin/env python3
# Filename: monitor-example.py

# Command usage:
# sudo python3 monitor-example.py -d DEVICE_NAME -p SERIAL_PORT -b BAUD
# sudo python3 monitor-example.py -d sm00 -p /dev/ttyUSB0 -b 9600

import os
import sys
import datetime as dt
import argparse
import json

# Import MobileInsight modules
from mobile_insight.monitor import OnlineMonitor
# from mobile_insight.analyzer import MsgLogger
from myMsgLogger import MyMsgLogger


if __name__ == "__main__":
    with open('../device_to_serial.json', 'r') as f:
        json_data = json.load(f)
        device_to_serial = json_data["device_to_serial"]
        serial_to_device = json_data["serial_to_device"]
        
    with open('../password.txt', 'r', encoding='utf-8') as f:
        password = f.readline().strip()
        
    with open('../savedir.txt', 'r', encoding='utf-8') as f:
        savedir = f.readline().strip()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, help="device: e.g., qc00", default="unam")
    parser.add_argument("-p", "--serial_port", type=str, help="Serial DM Port")
    parser.add_argument("-b", "--baudrate", type=int, help='baud rate', default=9600)
    parser.add_argument("-f", "--file", type=str, help='save file: *.mi2log')
    parser.add_argument("-dp", "--decode_file", type=str, help='save file: *.xml')
    args = parser.parse_args()

    # Set Path
    now = dt.datetime.today()
    date = [str(x) for x in [now.year, now.month, now.day]]
    date = [x.zfill(2) for x in date]
    date = '-'.join(date)
    
    mi2log_path1 = f'{savedir}/{date}/mi2log'  # mobileinsight log: *.mi2log
    if not os.path.isdir(mi2log_path1):
        print(f"makedir: {mi2log_path1}")
        os.makedirs(mi2log_path1)
        
    mi2log_path2 = f'{savedir}/{date}/mi2log_xml'  # mobileinsight log: *.xml
    if not os.path.isdir(mi2log_path2):
        print(f"makedir: {mi2log_path2}")
        os.makedirs(mi2log_path2)

    dev = args.device
    ser = args.serial_port
    baudrate = args.baudrate
    
    if dev.startswith('sm'):
        ser = os.path.join("/dev/serial/by-id", "usb-SAMSUNG_SAMSUNG_Android_{}-if00-port0".format(device_to_serial[dev]))
    elif dev.startswith('qc'):
        ser = os.path.join("/dev/serial/by-id", "usb-Quectel_RM500Q-GL_{}-if00-port0".format(device_to_serial[dev]))
    
    if ser is None:
        print("Error: please specify a valid Serial DM Port.")
        print(' '.join(["Usage: sudo python3", __file__, "<-p SERIAL_PORT_NAME> [-d <DEVICE_NAME>] [-b <BAUD_RATE>]"]))
        sys.exit(1)

    # os.system(f"sudo systemctl restart serial-getty@{ser}.service")
    
    now = dt.datetime.today()
    n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    n = [x.zfill(2) for x in n]  # zero-padding to two digit
    n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
    
    savepath_mi2log = os.path.join(mi2log_path1, "diag_log_{}_{}.mi2log".format(dev, n))
    savepath_xml = os.path.join(mi2log_path2, "diag_log_{}_{}.xml".format(dev, n))
    
    # Initialize a 3G/4G monitor
    src = OnlineMonitor()
    src.set_serial_port(ser)  # the serial port to collect the traces
    src.set_baudrate(baudrate)  # the baudrate of the port

    # Save the monitoring results as *.mi2log file
    if args.file is not None:
        src.save_log_as(args.file)
    else:
        src.save_log_as(savepath_mi2log)

    # Enable all messages
    # src.enable_log_all()
    
    # Enable 3G/4G/5G messages to be monitored. Here we enable RRC (radio resource control) monitoring
    src.enable_log("5G_NR_RRC_OTA_Packet")
    src.enable_log("LTE_RRC_OTA_Packet")
    src.enable_log("WCDMA_RRC_OTA_Packet")
    
    # Enable other messages
    src.enable_log("LTE_RRC_Serv_Cell_Info")
    # src.enable_log("LTE_PHY_Serv_Cell_Measurement")
    # src.enable_log("LTE_PHY_Connected_Mode_Intra_Freq_Meas")
    # src.enable_log("LTE_PHY_Connected_Mode_Neighbor_Measurement")
    
    # Sheng-ru's needed
    src.enable_log('5G_NR_ML1_Searcher_Measurement_Database_Update_Ext')
    src.enable_log('LTE_PHY_Connected_Mode_Intra_Freq_Meas')
    src.enable_log('LTE_PHY_Connected_Mode_Neighbor_Measurement')
    src.enable_log('LTE_PHY_Inter_RAT_Measurement')
    src.enable_log('LTE_PHY_Inter_Freq_Log')

    # Dump the messages to std I/O. Comment it if it is not needed.
    dumper = MyMsgLogger()
    dumper.set_source(src)
    dumper.set_decoding(MyMsgLogger.XML)  # decode the message as xml
    dumper.set_dump_type(MyMsgLogger.ALL)
    
    # Save the decoded monitoring results as *.xml file
    if args.decode_file is not None:
        dumper.save_decoded_msg_as(args.decode_file)
    else:
        dumper.save_decoded_msg_as(savepath_xml)

    # Start the monitoring
    src.run()
