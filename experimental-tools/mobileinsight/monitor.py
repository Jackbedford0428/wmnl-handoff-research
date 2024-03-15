#!/usr/bin/env python3
# Filename: monitor-example.py

# Command usage:
# sudo python3 monitor-example.py -d DEVICE_NAME -p SERIAL_PORT -b BAUD
# sudo python3 monitor-example.py -d sm00 -p /dev/ttyUSB0 -b 9600

import os
import sys
import datetime as dt
import argparse
from device_to_serial import device_to_serial, serial_to_device

# Import MobileInsight modules
from mobile_insight.monitor import OnlineMonitor
from mobile_insight.analyzer import MsgLogger

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", type=str,
                    help="device name (allows only 1 device)", default="unam")
parser.add_argument("-p", "--serial_port", type=str,
                    help="serial DM port")
parser.add_argument("-b", "--baud", type=int,
                    help="baudrate", default=9600)
args = parser.parse_args()

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
    # Set Path
    now = dt.datetime.today()
    date = [str(x) for x in [now.year, now.month, now.day]]
    date = [x.zfill(2) for x in date]
    date = '-'.join(date)
    makedir("./log/{}".format(date))
    
    dirpath = "./log/{}/{}".format(date, "mi2log")  # mobileinsight log

    if args.device == "unam":
        if not args.serial_port:
            print("Error: please specify a valid serial DM port.")
            print(' '.join(["Usage: sudo python3", __file__, "<-d DEVICE_NAME> [-p SERIAL_PORT_NAME] [-b BAUNRATE]"]))
            print(' '.join(["       sudo python3", __file__, "<-p SERIAL_PORT_NAME> [-b BAUNRATE]"]))
            sys.exit(1)
    else:
        if args.device.startswith('sm'):
            args.serial_port = os.path.join("/dev/serial/by-id", "usb-SAMSUNG_SAMSUNG_Android_{}-if00-port0".format(device_to_serial[args.device]))
        elif args.device.startswith('qc'):
            args.serial_port = os.path.join("/dev/serial/by-id", "usb-Quectel_RM500Q-GL_{}-if00-port0".format(device_to_serial[args.device]))

    # os.system(f"sudo systemctl restart serial-getty@{args.serial_port}.service")
    
    # Initialize a 3G/4G monitor
    src = OnlineMonitor()
    src.set_serial_port(args.serial_port)  # the serial port to collect the traces
    src.set_baudrate(args.baud)  # the baudrate of the port

    # Save the monitoring results as an offline log
    now = dt.datetime.today()
    n = [str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second]]
    n = [x.zfill(2) for x in n]  # zero-padding to two digit
    n = '-'.join(n[:3]) + '_' + '-'.join(n[3:])
    savepath = os.path.join(dirpath, "diag_log_{}_{}.mi2log".format(args.device, n))
    src.save_log_as(savepath)

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
    # Shen-ru's needed
    src.enable_log('5G_NR_ML1_Searcher_Measurement_Database_Update_Ext')
    src.enable_log('LTE_PHY_Connected_Mode_Intra_Freq_Meas')
    src.enable_log('LTE_PHY_Connected_Mode_Neighbor_Measurement')
    src.enable_log('LTE_PHY_Inter_RAT_Measurement')
    src.enable_log('LTE_PHY_Inter_Freq_Log')

    # Dump the messages to std I/O. Comment it if it is not needed.
    dumper = MsgLogger()
    dumper.set_source(src)
    dumper.set_decoding(MsgLogger.XML)  # decode the message as xml

    # Start the monitoring
    src.run()
