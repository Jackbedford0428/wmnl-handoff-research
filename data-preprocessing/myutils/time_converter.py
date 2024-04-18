#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import datetime as dt
from typing import List
from typing import Union
from packaging import version

__all__ = [
    "datetime_to_str",
    "str_to_datetime",
    "str_to_datetime_batch",
    "epoch_to_datetime",
    "datetime_to_epoch",
]

# pandas: pd.Timestamp("2022-09-29 16:24:58.252615") <class 'pandas._libs.tslibs.timestamps.Timestamp'>
# datetime: dt.datetime(2022, 9, 29, 16, 24, 58, 252615) <class 'datetime.datetime'>

def datetime_to_str(timestamp_datetime):
    return dt.datetime.strftime(timestamp_datetime, "%Y-%m-%d %H:%M:%S.%f")

def str_to_datetime(timestamp_str, format='pd'):
    if format == 'pd':
        return pd.to_datetime(timestamp_str)
    elif format == 'dt':
        try:
            return dt.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        except:
            try:
                return dt.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except:
                return dt.datetime.strptime(timestamp_str, "%Y-%m-%d")

def str_to_datetime_batch(df, parse_dates: Union[List[str], None] = None):
    if parse_dates is not None:
        if version.parse(pd.__version__) >= version.parse("2.0.0"):
            df[parse_dates] = pd.to_datetime(df[parse_dates].stack(), format='mixed').unstack()
        else:
            df[parse_dates] = pd.to_datetime(df[parse_dates].stack()).unstack()
    return df

def epoch_to_datetime(timestamp_epoch, format='pd', utc=8):
    if format == 'pd':
        return pd.to_datetime(timestamp_epoch, unit='s') + pd.Timedelta(hours=utc)
    elif format == 'dt':
        # return dt.datetime.utcfromtimestamp(timestamp_epoch) + dt.timedelta(hours=utc)  # Python 3.12 即將被棄用
        return dt.datetime.fromtimestamp(timestamp_epoch, dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=utc)

def datetime_to_epoch(timestamp_datetime, utc=8):
    timezone = dt.timezone(dt.timedelta(hours=utc))  # set timezone
    timestamp_datetime = timestamp_datetime.replace(tzinfo=timezone)
    return timestamp_datetime.timestamp()  # convert the datetime object to unix timestamp


# ===================== Test =====================
if __name__ == "__main__":
    test_cases = ['2024-03-25', '2024-03-25 13:34:01', '2024-03-25 13:34:01.774182']
    
    print("Test 'str_to_datetime'")
    for test_case in test_cases:
        print('------------------------')
        print(test_case)
        print("pd:", str_to_datetime(test_case, 'pd'), type(str_to_datetime(test_case, 'pd')))
        print("dt:", str_to_datetime(test_case, 'dt'), type(str_to_datetime(test_case, 'dt')))
    
    print()
    print("Test 'datetime_to_str'")
    for test_case in test_cases:
        print('------------------------')
        print(test_case)
        print("pd:", datetime_to_str(str_to_datetime(test_case, 'pd')), type(datetime_to_str(str_to_datetime(test_case, 'pd'))))
        print("dt:", datetime_to_str(str_to_datetime(test_case, 'dt')), type(datetime_to_str(str_to_datetime(test_case, 'dt'))))
        
    print()
    print("Test 'datetime_to_epoch'")
    for test_case in test_cases:
        print('------------------------')
        print(test_case)
        print("pd:", datetime_to_epoch(str_to_datetime(test_case, 'pd')), type(datetime_to_epoch(str_to_datetime(test_case, 'pd'))))
        print("dt:", datetime_to_epoch(str_to_datetime(test_case, 'dt')), type(datetime_to_epoch(str_to_datetime(test_case, 'dt'))))
    
    print()
    print("Test 'epoch_to_datetime'")
    for test_case in test_cases:
        print('------------------------')
        print(test_case)
        print("pd:", epoch_to_datetime(datetime_to_epoch(str_to_datetime(test_case, 'pd')), 'pd'), type(epoch_to_datetime(datetime_to_epoch(str_to_datetime(test_case, 'pd')), 'pd')))
        print("dt:", epoch_to_datetime(datetime_to_epoch(str_to_datetime(test_case, 'dt')), 'dt'), type(epoch_to_datetime(datetime_to_epoch(str_to_datetime(test_case, 'dt')), 'dt')))
    
    print()
