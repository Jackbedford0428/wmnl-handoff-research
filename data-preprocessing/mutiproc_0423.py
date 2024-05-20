from os.path import exists
import random
import subprocess
import multiprocessing
# For Progress Bar
# from tqdm import tqdm, trange
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
from time import time,strftime,gmtime
from collections import Counter
import logging
from typing import Optional


class Run_Test_List:
    def __init__(self, file_py: str, meta_data_list:list, cpu_count:int=8, ncols:int=70, nice_use:bool=False,completed_tasks_file=None,shuffle_use:bool=False,Disable_complete_check:bool=False):
        """初始化要測試的內容

        Args:
            file_py (str): 測試使用的py檔案
            args_list (list): 參數(多個)
            ncols (int, optional): 執行的進度條長度. Defaults to 50.
            cpu_count (int, optional): 設定一次要跑幾個. Defaults to 8.
        """
        self.count = cpu_count #multiprocessing.cpu_count()
        self.ncols = ncols # 執行的進度條長度
        self.nice_use = nice_use #是否使用nice
        self.file_py = file_py
        # 建立暫時存放已完成的列表
        self.completed_tasks_file = 'success_logs.log' if completed_tasks_file is None else completed_tasks_file
        self.shuffle_use = shuffle_use

        if meta_data_list:
            # self.doList = [file_py].extend(self.expand(meta_data_list, '-i'))
            # self.doList = self.expand(meta_data_list, [file_py + '-i'])
            self.doList = [file_py + ' -i "' + '" "'.join(s) + '"' for s in meta_data_list]
        # else:
        #     # 添加每個模型以及其對應的參數在測試用檔案之後
        #     temp = []
        #     for i in args_list:
        #         temp.extend(self.expand(i,[file_py]))
        #     self.doList = temp

        # 創建格式化器，包含時間訊息
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 創建文件處理器，用於記錄錯誤的日誌
        error_handler = logging.FileHandler('error_logs.log')
        error_handler.setFormatter(formatter)
        logging.getLogger('error').addHandler(error_handler)
        logging.getLogger('error').setLevel(logging.ERROR) # 設置級別為 ERROR，紀錄 ERROR 及以上級別的日誌

        # 創建文件處理器，用於紀錄成功完成的日誌
        success_handler = logging.FileHandler(self.completed_tasks_file)
        success_handler.setFormatter(formatter)
        logging.getLogger('success').addHandler(success_handler)
        logging.getLogger('success').setLevel(logging.INFO) # 设置级别为 INFO，记录 INFO 及以上级别的日志

        # 預期所需要執行的指令數量
        print(f"{len(self.doList)} need run")
        # 讀取已完成的任務集合
        completed_tasks = set()
        # 嘗試載入已完成的任務列表，如果不存在則略過
        if exists(self.completed_tasks_file):
            # 從已完成任務的日誌文件中提取已完成的任務
            with open(self.completed_tasks_file, 'r') as f:
                completed_tasks = set( line.split('__')[1] for line in f if "- INFO - command:" in line)

        if Disable_complete_check is False:
            # 檢查任務是否已經完成
            self.doList = [i for i in self.doList if i not in completed_tasks]
        
        print(f"{len(self.doList)} will run")

        if self.shuffle_use:
            # 使用 random.shuffle() 打亂model跑的順序，讓gpu和 cpu only交錯
            random.shuffle(self.doList)

        self.loopCase()

        
    def loopCase(self):
        # 使用process_map用於將平行任務同時執行，並顯示進度條
        # 內包含一個tqdm 用於確認目前有多少任務已被放入執行(似乎不管用，好像都會被一起放入)
        r = process_map(self.runcmd, 
                        enumerate(self.doList), 
                        max_workers=self.count, 
                        total=len(self.doList), 
                        ncols=self.ncols,
                        desc=self.file_py.split('/')[-1])
        # 使用 Counter 統計回傳值
        print ("done , follow is the return conuter ('return code': count of return)")
        print (Counter(r))
        print('---------------')

    def expand(self,args:list,do_List:list) -> list:
        """將do_List作為檔案，把args依據排列組合添加在其後(有附上--以方便argsparser使用)

        Args:
            args (list): 需要添加的參數(添加會同時增加--在前面)
            do_List (str): 檔案名稱

        Returns:
            list: 添加完之後的list回傳
        """
        for i in args:
            preList, do_List = do_List, []
            for j in args[i]:
                do_List.extend([x + ' ' + i + ' ' + str(j) for x in preList])
        return do_List

    def runcmd(self,parameter:tuple) -> int:
        """額外建立子程序執行parameter

        Args:
            parameter (tuple(nice_number, command)): 將command建立子程序執行，並使用nice_number來設定nice值 

        Returns:
            int: 回傳執行結果
        """
        return_code = -1
        nice_number, command_org = parameter
        #使用nice就將執行指令補上nice(目前使用0-9)
        command = f'nice -n {nice_number % 10} {command_org}' if self.nice_use else command_org 
        start_time = time()
        try:
            r = subprocess.Popen(command ,shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 等待命令完成
            stdout, stderr = r.communicate()

            # 獲得命令的返回值
            return_code = r.returncode
            if return_code:
                # 如果任務執行失敗，記錄錯誤訊息
                logging.getLogger('error').error(f"Task:'{command}' \nstd:{str(stdout)} \nerr:{str(stderr)}")

                    
            else:
                success_str = f'command:__{command_org}__ ,use time: {strftime("%H:%M:%S", gmtime((time()- start_time)))}'
                if self.nice_use: success_str+= f',nice value: {nice_number % 10}'
                logging.getLogger('success').info(success_str)
        
        # except AssertionError as e:
        except Exception as e:
            # 如果任务执行失败，记录错误信息
            logging.error(f"Task '{command}' \nfailed: {e}")
                
        return return_code 