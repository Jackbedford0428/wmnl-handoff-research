實驗操作流程
===

CHT
- B1 525
- B3 1750, 1400
- B7 3050, 3400
- B8 3650

TWM
- B1 275
- B3 1275
- B28 9560

# 流程
1. 每趟實驗前，ntpq -p 或 sudo ntpd -q ntp.server.ip/name，確保時間有週期性/手動同步。
2. 量測 server/client 之間的時間誤差，自動記錄到 json 檔。
3. 啟動 MobileInsight。
4. 先後啟動 socket 或 iPerf 的 server 和 client。

# 模組操作
1. 獨立電源給 Hub 供電
2. Hub 插上筆電
3. 模組一一插上 Hub

> 只要發生筆電讀取不到模組的情況，一律按步驟 3 拔掉重插
> - Serial Exception
> - No such device
> - [cmderror]

# 連線操作
- 連網 ./dial-all.py
- 斷網 ./disconnect-all.py
- 確認連線狀態 sudo ./get-modem-connection-status.sh -i `device`
- 切頻 sudo ./band-setting.sh -i `device` -l `lte:band` -e `nr:band`
- 檢查連線頻帶 sudo ./band-setting.sh -i `device`
- 重置頻帶設定 ./reset-all-band.py
- Force Set RAT to LTE & 5G NR: sudo ./qc-at -i `device` -c 'at+qnwprefcfg="mode_pref",LTE:NR5G'

# 時間同步 offset
- behind server by (+)
- ahead of server by (-)
