#!/usr/bin/env bash

# This script is to capture the location info from the GPS dongle

path="log"
if [ ! -d $path ]
	then
		mkdir $path
fi
path="$path/`(date +%Y-%m-%d)`"
if [ ! -d $path ]
	then
		mkdir $path
fi
path="$path/gps"
if [ ! -d $path ]
	then
		mkdir $path
fi

# interval=1
# FILE_PATH="sample"
# FILE_PATH=`(date +%Y_%m%d_%H%M)`"_GPS_info"
FILE_PATH="$path/`(date +%Y-%m-%d_%H-%M-%S)`_GPS_info"
: > $FILE_PATH

echo "Start logging..."
echo "Press Ctrl-C or Ctrl-Z to stop"
while true
do
	(gpspipe -uu -w) >> $FILE_PATH
#	sleep $interval
done
