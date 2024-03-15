#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import datetime as dt
import portion as P
from dateutil.parser import parse


dirpath = "./log"
# dates = sorted(os.listdir(dirpath), reverse=True)
dates = sorted(os.listdir(dirpath))
if not dates:
    sys.exit()

os.system("echo wmnlab | sudo -S su")

def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False

def str_to_datetime(ts):
    """
    Convert a timestamp string in microseconds or milliseconds into datetime.datetime

    Args:
        ts (str): timestamp string (e.g., 2022-09-29 16:24:58.252615)
    Returns:
        (datetime.datetime)
    """
    return dt.datetime.strptime(ts, '%Y-%m-%d')

def datetime_to_str(ts):
    """
    Convert a datetime timestamp in microseconds into str

    Args:
        ts (datetime.datetime): datetime timestamp (e.g., datetime.datetime(2022, 9, 29, 16, 24, 58, 252615))
    Returns:
        (str): timestamp string (e.g., 2022-09-29 16:24:58.252615)
    """
    return dt.datetime.strftime(ts, '%Y-%m-%d')

def unified_date_format(date):
    now = dt.datetime.today()
    date = date.split('-')
    if len(date) == 3:
        pass
    if len(date) == 2:
        date.insert(0, str(now.year))
        _date = '-'.join(date)
        if str_to_datetime(_date) > now:
            date[0] = str(int(date[0]) - 1)
    else:
        print("date format is not correct!")
        sys.exit()
    date = [x.zfill(2) for x in date]
    date = '-'.join(date)
    return date
        
# now = dt.datetime.today()
# date = [str(x) for x in [now.year, now.month, now.day]]
# date = [x.zfill(2) for x in date]

if len(sys.argv) < 2:
    date = dates.pop()
    print(f"sudo rm -rf {os.path.join(dirpath, date)}")
    os.system(f"sudo rm -rf {os.path.join(dirpath, date)}")
elif len(sys.argv) == 2:
    if not is_date(sys.argv[1]):
        print("it is not a date!")
        sys.exit()
    date = unified_date_format(sys.argv[1])
    if date in dates:
        print(f"sudo rm -rf {os.path.join(dirpath, date)}")
        os.system(f"sudo rm -rf {os.path.join(dirpath, date)}")
else:
    if not is_date(sys.argv[1]):
        print("it is not a date!")
        sys.exit()
    elif is_date(sys.argv[2]):
        _date1 = str_to_datetime(unified_date_format(sys.argv[1]))
        _date2 = str_to_datetime(unified_date_format(sys.argv[2]))
        intv = P.closed(_date1, _date2)
        if intv.empty:
            print(f"date1 > date2: {_date1}, {_date2}")
        else:
            for date in dates:
                if str_to_datetime(date) in intv:
                    print(f"sudo rm -rf {os.path.join(dirpath, date)}")
                    os.system(f"sudo rm -rf {os.path.join(dirpath, date)}")
    elif sys.argv[2] == "-a":
        _date = str_to_datetime(unified_date_format(sys.argv[1]))
        dates = sorted(dates, reverse=True)
        for date in dates:
            if str_to_datetime(date) > _date:
                print(f"sudo rm -rf {os.path.join(dirpath, date)}")
                os.system(f"sudo rm -rf {os.path.join(dirpath, date)}")
            else:
                break
    elif sys.argv[2] == "-b":
        _date = str_to_datetime(unified_date_format(sys.argv[1]))
        for date in dates:
            if str_to_datetime(date) < _date:
                print(f"sudo rm -rf {os.path.join(dirpath, date)}")
                os.system(f"sudo rm -rf {os.path.join(dirpath, date)}")
            else:
                break
