import os
import sys
import argparse
import time
import traceback
from pytictoc import TicToc
from pprint import pprint
import json
import pandas as pd
from myutils import *
from mutiproc_0423 import Run_Test_List

# ===================== Main Process =====================
if __name__ == "__main__":
    # ===================== Arguments =====================
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dates", type=str, nargs='+', help="date folders to process")
    args = parser.parse_args()
    
    if args.dates is not None:
        dates = sorted(args.dates)
        metadatas = metadata_loader(dates)
        for metadata in metadatas:
            print(metadata)
    else:
        raise TypeError("Please specify the date you want to process.")

    Run_Test_List('python3 ./udp_preprocessing_v3.py', metadatas, cpu_count=8)
    