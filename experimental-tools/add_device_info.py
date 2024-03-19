#!/usr/bin/env python3

import os

user_input = input("請填入 device.txt")
print(f"echo {user_input} > device.txt")
os.system(f"echo {user_input} > device.txt")

user_input = input("請填入 password.txt")
print(f"echo {user_input} > password.txt")
os.system(f"echo {user_input} > password.txt")

user_input = input("請填入 savedir.txt")
print(f"echo {user_input} > savedir.txt")
os.system(f"echo {user_input} > savedir.txt")
