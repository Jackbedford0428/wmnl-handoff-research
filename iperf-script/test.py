import os
import sys
import time
import socket
import threading
import datetime as dt
import argparse
import subprocess
import re
import signal

parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host", type=str,
                    help="server ip address", default="140.112.20.183")
                    # help="server ip address", default="210.65.88.213")
parser.add_argument("-p", "--port", type=int, nargs='+',  # input list of port numbers sep by 'space'
                    help="port to bind")
parser.add_argument("-d", "--device", type=str,
                    help="device name (only allow 1 device)")
parser.add_argument("-u", "--udp", action="store_true",  # needs not value, "True" if set "-u"
                    help="use UDP rather than TCP")
parser.add_argument("-b", "--bitrate", type=str,
                    help="target bitrate in bits/sec (0 for unlimited)", default=["200k", "1M"])
parser.add_argument("-l", "--length", type=str,
                    help="length of buffer to read or write in bytes (packet size)", default=["250", "1250"])
parser.add_argument("-t", "--time", type=int,
                    help="time in seconds to transmit for (default 1 hour = 3600 secs)", default=3600)
parser.add_argument("-S", "--stream", type=str,
                    help="stream flow: uplink (ul), downlink (dl), bi-link (default bl)", default="bl")
args = parser.parse_args()

print(args.host)
print(args.port)
print(args.device)
print(args.udp)
print(args.bitrate)

if type(args.bitrate) is list:
    bitrate = args.bitrate[0] if args.udp else args.bitrate[1]
else:
    bitrate = args.bitrate
if type(args.length) is list:
    packet_size = args.length[0] if args.udp else args.length[1]
else:
    packet_size = args.length

# if args.port:
#     print("hello port")
# elif args.device:
#     print("hello device")
# else:
#     print("NONE")