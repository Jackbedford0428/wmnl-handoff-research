#! /usr/bin/sh

# sudo apt install ntp
echo "sudo systemctl restart ntp.service"
sudo systemctl restart ntp.service
sleep 7

ntpq -p
sleep 3.5
ntpq -p
sleep 3.5

# echo "sudo systemctl stop ntp.service"
# sudo systemctl stop ntp.service
sleep 1

ntpq -p
