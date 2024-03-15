#!/usr/bin/env bash

# https://www.lammertbies.nl/comm/info/gps-time

# Install gpsd
sudo apt-get update
sudo apt-get install gpsd-clients gpsd

# Plugin GPS receiver
dmesg | tail -n 5

# Edit File: "/etc/modprobe.d/blacklist.conf"
## sudo vim /etc/modprobe.d/blacklist.conf
"""
# blacklist pl2303
"""
## sudo reboot (optional)
## However, gps module needs pl2303!

# Check DM port
ls /dev/serial/by-id
''' usb-Prolific_Technology_Inc._USB-Serial_Controller_BSEIb115818-if00-port0 '''  # Server
''' usb-Prolific_Technology_Inc._USB-Serial_Controller_CHCTb115818-if00-port0 '''  # Client

# Edit file "/dtc/default/gpsd"
sudo vim /etc/default/gpsd
'''
# Devices gpsd should collect to at boot time.
# They need to be read/writeable, either by user gpsd or the group dialout.
# DEVICES="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BSEIb115818-if00-port0"  # Server
# DEVICES="/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_CHCTb115818-if00-port0"  # Client
USBAUTO="true"

# Other options you want to pass to gpsd
GPSD_OPTIONS="-n"
START_DAEMON="true"
'''

# Restart gpsd
sudo service gpsd restart

# check gps
cgps
''' Ctrl+C to terminate '''


# Install ntp
sudo apt-get update
sudo apt-get install ntp

# Edit file "/etc/ntp.conf"
sudo vim /etc/ntp.conf
'''
# Specify one or more NTP servers.

# GPS Serial data reference (NTP0)
server 127.127.28.0 minpoll 4 maxpoll 4
fudge 127.127.28.0 time1 0 refid GPS

# GPS PPS reference (NTP1)
server 127.127.28.1 minpoll 4 maxpoll 4 prefer
fudge 127.127.28.1 refid PPS
'''

# Restart ntpd
sudo service ntp restart
