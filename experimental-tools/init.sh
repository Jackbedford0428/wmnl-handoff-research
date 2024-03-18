#!/usr/bin/env bash

# NOTE: This will let anyone who belongs to the 'pcap' group
# execute 'tcpdump'

# NOTE2: User running the script MUST be a sudoer. It is
# convenient to be able to sudo without a password.

sudo groupadd pcap
sudo usermod -a -G pcap $USER
sudo chgrp pcap /usr/sbin/tcpdump
sudo setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump
sudo ln -s /usr/sbin/tcpdump /usr/bin/tcpdump

sudo /usr/local/nginx/sbin/nginx


# # 創建組 pcap（在 macOS 中使用 dscl）
# sudo dscl . -create /Groups/pcap
# sudo dscl . -create /Groups/pcap PrimaryGroupID 1001  # 替換為適當的 GID

# # 將當前用戶添加到 pcap 組
# sudo dscl . -append /Groups/pcap GroupMembership $USER

# # 修改 tcpdump 的所屬組
# sudo chown root:1001 /usr/sbin/tcpdump

# # 賦予 tcpdump 特定權限
# sudo chmod 750 /usr/sbin/tcpdump

# # 創建 tcpdump 符號鏈接（已更改權限）
# sudo ln -s /usr/sbin/tcpdump /usr/local/bin/tcpdump