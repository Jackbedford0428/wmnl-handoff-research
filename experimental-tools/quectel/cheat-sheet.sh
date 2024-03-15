#!/bin/bash

# Edit file "/etc/resolv.conf"
sudo vim /etc/resolv.conf
'''
# nameserver 127.0.0.53
nameserver 8.8.8.8
options edns0 trust-ad
'''
