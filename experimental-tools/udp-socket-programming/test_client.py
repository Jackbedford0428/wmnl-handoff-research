#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import socket

HOST = '140.112.20.183'
PORT = 3230
server_addr = (HOST, PORT)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 'qc01\0'.encode())
# s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 'sm06\0'.encode())
# s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 'sm08\0'.encode())
# s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 'eno1\0'.encode())
# s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 'wlp0s20f3\0'.encode())

while True:
    outdata = input('please input message: ')
    print('sendto ' + str(server_addr) + ': ' + outdata)
    s.sendto(outdata.encode(), server_addr)
    
    indata, addr = s.recvfrom(1024)
    print('recvfrom ' + str(addr) + ': ' + indata.decode())
