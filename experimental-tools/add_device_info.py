#!/usr/bin/env python3

import os

user_input = input("請填入 device.txt\n")
print(f"echo {user_input} > device.txt")
os.system(f"echo {user_input} > device.txt")
print("-----------------------------------")

user_input = input("請填入 password.txt\n")
print(f"echo {user_input} > password.txt")
os.system(f"echo {user_input} > password.txt")
print("-----------------------------------")

user_input = input("請填入 savedir.txt\n")
print(f"echo {user_input} > savedir.txt")
os.system(f"echo {user_input} > savedir.txt")
print("-----------------------------------")
