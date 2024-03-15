#!/usr/bin/env bash

# NOTE: This will let anyone who belongs to the 'pcap' group
# execute 'tcpdump'

# NOTE2: User running the script MUST be a sudoer. It is
# convenient to be able to sudo without a password.

# sudo groupadd pcap
# sudo usermod -a -G pcap $USER
# sudo chgrp pcap /usr/sbin/tcpdump
# sudo setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump
# sudo ln -s /usr/sbin/tcpdump /usr/bin/tcpdump

# sudo /usr/local/nginx/sbin/nginx



# 创建组 pcap（在 macOS 中使用 dscl）
sudo dscl . -create /Groups/pcap
sudo dscl . -create /Groups/pcap PrimaryGroupID 1001  # 替换为适当的 GID

# 将当前用户添加到 pcap 组
sudo dscl . -append /Groups/pcap GroupMembership $USER

# 修改 tcpdump 的所属组
sudo chown root:1001 /usr/sbin/tcpdump

# 赋予 tcpdump 特定权限
sudo chmod 750 /usr/sbin/tcpdump

# 创建 tcpdump 符号链接（已更改权限）
sudo ln -s /usr/sbin/tcpdump /usr/local/bin/tcpdump