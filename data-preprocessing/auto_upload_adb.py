
import os
import shutil

# 定義函數：自動檢測並新增尚未存在的資料夾
def makedir(dirpath, mode=0):  # mode=1: show message, mode=0: hide message
    if os.path.isdir(dirpath):
        if mode:
            print("mkdir: cannot create directory '{}': directory has already existed.".format(dirpath))
        return
    ### recursively make directory
    _temp = []
    while not os.path.isdir(dirpath):
        _temp.append(dirpath)
        dirpath = os.path.dirname(dirpath)
    while _temp:
        dirpath = _temp.pop()
        print("mkdir", dirpath)
        os.mkdir(dirpath)

client_pcap_flag, server_pcap_flag, tsync_flag, cimon_flag = False, False, False, False

# ***********************************************************************************************
# TODO: 設定目標日期
target_date = "2023-08-21"

# TODO: 設定電腦根目錄的資料夾路經
# computer_root_folder = "/Users/jackbedford/Desktop/MOXA/temp/"
computer_root_folder = "/home/wmnlab/D/phone_temp/"

# TODO: 設定手機和電腦的資料夾路徑
client_pcap_flag = True
tsync_flag = True
cimon_flag = True
server_pcap_flag = True

# 設定手機備份資料夾路徑
client_pcap_phone_folder = "/sdcard/pcapdir/"
client_pcap_computer_folder = os.path.join(computer_root_folder, target_date, "client_pcap")
makedir(client_pcap_computer_folder)

tsync_phone_folder = os.path.join("/sdcard/wmnl-handoff-research/experimental-tools-beta/sync/log/", target_date)
tsync_computer_folder = os.path.join(computer_root_folder, target_date)
makedir(tsync_computer_folder)

cimon_phone_folder = "/sdcard/Android/data/com.example.cellinfomonitor/files/Documents/"
cimon_computer_folder = os.path.join(computer_root_folder, target_date, "cimon")
makedir(cimon_computer_folder)

# 設定電腦的備份資料夾路徑
server_pcap_computer_folder1 = "/home/wmnlab/temp/"
server_pcap_computer_folder2 = os.path.join(computer_root_folder, target_date, "server_pcap")
makedir(server_pcap_computer_folder2)
# ***********************************************************************************************

# 執行 adb 命令取得連接的手機資訊
adb_devices_command = "adb devices"
output = os.popen(adb_devices_command).read()

# 解析 adb devices 命令的輸出
lines = output.strip().split("\n")[1:]
serial_numbers = {}
unauthorized_serial_numbers = {}

serial_to_device = {
    "R5CR20FDXHK":"sm00",
    "R5CR30P9Z8Y":"sm01",
    "R5CRA1GCHFV":"sm02",
    "R5CRA1JYYQJ":"sm03",
    "R5CRA1EV0XH":"sm04",
    "R5CRA1GBLAZ":"sm05",
    "R5CRA1ESYWM":"sm06",
    "R5CRA1ET22M":"sm07",
    "R5CRA1D23QK":"sm08",
    "R5CRA2EGJ5X":"sm09",
    "R5CRA1ET5KB":"sm10",
    "R5CRA1D2MRJ":"sm11",
    "73e11a9f":"xm00",
    "491d5141":"xm01",
    "790fc81d":"xm02",
    "e2df293a":"xm03",
    "28636990":"xm04",
    "f8fe6582":"xm05",
    "d74749ee":"xm06",
    "10599c8d":"xm07",
    "57f67f91":"xm08",
    "232145e8":"xm09",
    "70e87dd6":"xm10",
    "df7aeaf8":"xm11",
    "e8c1eff5":"xm12",
    "ec32dc1e":"xm13",
    "2aad1ac6":"xm14",
    "64545f94":"xm15",
    "613a273a":"xm16",
    "fe3df56f":"xm17",
    "76857c8" :"qc00",
    "bc4587d" :"qc01",
    "5881b62f":"qc02",
    "32b2bdb2":"qc03",
}

for line in lines:
    parts = line.split("\t")
    if len(parts) >= 2:
        serial_number, status = parts[:2]
        if status == "device":
            serial_numbers[serial_to_device[serial_number]] = serial_number
        elif status == "unauthorized":
            unauthorized_serial_numbers[serial_to_device[serial_number]] = serial_number

# 按字典順序排序
serial_numbers = dict(sorted(serial_numbers.items()))
unauthorized_serial_numbers = dict(sorted(unauthorized_serial_numbers.items()))

# 印出結果
print("Connected devices:")
print(serial_numbers)
print("Unauthorized devices:")
print(unauthorized_serial_numbers)

# 上傳 client_pcap 檔案
if client_pcap_flag:
    print('upload client pcap...')
    for dev, serial in serial_numbers.items():
        # 使用 adb 命令列出手機上的檔案清單
        adb_ls_command = f"adb -s {serial} shell ls {client_pcap_phone_folder}"
        output = os.popen(adb_ls_command).read()
        phone_files = output.split()

        pcap_files = [filename for filename in phone_files if filename.startswith("client_pcap_") and filename.endswith(".pcap")]

        for filename in pcap_files:
            if target_date in filename:
                source_path = os.path.join(client_pcap_phone_folder, filename)
                destination_path = os.path.join(client_pcap_computer_folder, filename)
                # 使用 adb 命令將檔案從手機上傳到電腦
                adb_command = f"adb -s {serial} pull -a {source_path} {destination_path}"
                os.system(adb_command)
            
# 上傳 time_sync 檔案
if tsync_flag:
    print('upload time sync...')
    for dev, serial in serial_numbers.items():
        # 使用 adb 命令列出手機上的檔案清單
        adb_ls_command = f"adb -s {serial} shell ls {tsync_phone_folder}"
        output = os.popen(adb_ls_command).read()
        phone_files = output.split()

        sync_files = [filename for filename in phone_files if filename.startswith("time_sync_") and filename.endswith(".json")]

        for filename in sync_files:
            source_path = os.path.join(tsync_phone_folder, filename)
            destination_path = os.path.join(tsync_computer_folder, filename)
            # 使用 adb 命令將檔案從手機上傳到電腦
            adb_command = f"adb -s {serial} pull -a {source_path} {destination_path}"
            os.system(adb_command)

# 備份 server_pcap 檔案
if server_pcap_flag:
    print('copy server pcap...')
    # 遍歷檔案清單，複製符合目標日期的檔案到目標資料夾
    pcap_files = [filename for filename in os.listdir(server_pcap_computer_folder1) if filename.startswith("server_pcap_") and filename.endswith(".pcap")]

    for filename in pcap_files:
        if target_date in filename:
            source_path = os.path.join(server_pcap_computer_folder1, filename)
            destination_path = os.path.join(server_pcap_computer_folder2, filename)
            shutil.copy2(source_path, destination_path)  # 複製檔案並保留 meta data

# 上傳 cimon 檔案
if cimon_flag:
    print('upload cimon...')
    for dev, serial in serial_numbers.items():
        # 使用 adb 命令列出手機上的檔案清單
        adb_ls_command = f"adb -s {serial} shell ls {cimon_phone_folder}"
        output = os.popen(adb_ls_command).read()
        phone_files = output.split()

        cimon_files = [filename for filename in phone_files if filename.startswith("cimon_") and filename.endswith(".csv")]

        for filename in cimon_files:
            if target_date in filename:
                source_path = os.path.join(cimon_phone_folder, filename)
                destination_path = os.path.join(cimon_computer_folder, "cimon_" + dev + "_" + filename[6:])
                # 使用 adb 命令將檔案從手機上傳到電腦
                adb_command = f"adb -s {serial} pull -a {source_path} {destination_path}"
                os.system(adb_command)
