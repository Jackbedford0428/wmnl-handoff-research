#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import datetime as dt
from typing import List
from typing import Union
from .time_converter import str_to_datetime_batch

__all__ = [
    "generate_dataframe",
]

def generate_dataframe(filepaths: Union[str, List[str]],
                       sep=',', header='infer', index_col=None, dtype=None,
                       nrows=None, chunksize=None, usecols=None, low_memory=True,
                       parse_dates: Union[List[str], None] = None):
    
    if isinstance(filepaths, str):
        df = pd.read_csv(filepaths, sep=sep, header=header, index_col=index_col, dtype=dtype,
                        nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=low_memory)
        if chunksize is None and not df.empty:
            df = str_to_datetime_batch(df, parse_dates=parse_dates)
        return df
    
    dfs = []
    for filepath in filepaths:
        df = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col, dtype=dtype,
                        nrows=nrows, chunksize=chunksize, usecols=usecols, low_memory=low_memory)
        if chunksize is None and not df.empty:
            df = str_to_datetime_batch(df, parse_dates=parse_dates)
        dfs.append(df)
    return dfs


# ===================== Test =====================
if __name__ == "__main__":
    empty_file = "/home/wmnlab/F/database/2024-03-20/UDP_Bandlock_9S_Phone_A/sm08/#01/data/diag_log_sm08_2024-03-20_15-23-10_nr_ml1.csv"
    # df = pd.read_csv(empty_file, parse_dates=['Timestamp'])
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'].stack(), format='mixed').unstack()
    df = generate_dataframe(empty_file, parse_dates=['Timestamp'])
    print(df.head())
    