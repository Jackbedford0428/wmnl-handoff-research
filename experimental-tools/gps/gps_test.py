import serial
import socket

ser = serial.Serial('/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BSEIb115818-if00-port0', 4800, timeout=1)
latitude = ''
longitude = ''
def readgps(latitude,longitude):
    """Read the GPG LINE using the NMEA standard"""
    while True:
        line = ser.readline()
        if "GPGGA" in line:
            latitude = line[18:26] #Yes it is positional info for lattitude
            longitude = line[31:39] #do it again
            return(latitude,longitude)
    print("Finished")