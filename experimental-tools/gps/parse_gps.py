#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import csv


filename = "2023_0202_2016_GPS_info"

fp = open(filename, 'r')
lines = fp.readlines()  # neglect '\n' when reading the file
print(lines[0])

for line in lines:
    if '"TPV"' not in line:
        continue
    print(line)
    print(find(line, '"lon"'))
