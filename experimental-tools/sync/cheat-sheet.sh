#! /usr/bin/sh

# Revise datetime manually
date
sudo date --set 2022-11-15
sudo date --set "2022-11-15 15:00:00:123456"

# Show ntp settings
ntptime
timedatectl
timedatectl show

# Start/stop ntp service
sudo systemctl stop ntp.service
sudo systemctl restart ntp.service
