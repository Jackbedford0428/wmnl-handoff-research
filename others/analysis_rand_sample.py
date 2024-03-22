import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import os
import statistics
import numpy as np
import random
import time


def server_time_to_client_time(server_time, ref_time1, ref_time2, diff1, diff2):
    return server_time - (diff1*(ref_time2-server_time)+diff2*(server_time-ref_time1))/(ref_time2-ref_time1)




phone_list = ['galaxy', 'pixel', 'xiaomi']
telecom_list = ['CHT', 'TWM', 'FET']


color_map = {
    "galaxyCHT": "red",
    "galaxyTWM": "orange",
    "pixelCHT": "blue",
    "pixelTWM": "dodgerblue",
    "xiaomiCHT": "green",
    "xiaomiTWM": "lime",
}

color_map1 = {
    0: "red",
    1: "orange",
    2: "yellow",
    3: "green",
    4: "blue",
    5: "indigo",
    6: "violet",
}


result_list = {}
result_list1 = {}


num_sample = 10


for phone in phone_list:
    for telecom in telecom_list:
        sys_path = sys.argv[1] + phone + "_" + telecom + "/"

        server = []
        client = []

        try:
            with open(sys_path + "server_sync1.csv", newline='') as f:
                reader = csv.reader(f)
                server.append(list(reader))
            with open(sys_path + "server.csv", newline='') as f:
                reader = csv.reader(f)
                server.append(list(reader))
            with open(sys_path + "server_sync2.csv", newline='') as f:
                reader = csv.reader(f)
                server.append(list(reader))
            with open(sys_path + "sync1.csv", newline='') as f:
                reader = csv.reader(f)
                client.append(list(reader))
            with open(sys_path + "client.csv", newline='') as f:
                reader = csv.reader(f)
                client.append(list(reader))
            with open(sys_path + "sync2.csv", newline='') as f:
                reader = csv.reader(f)
                client.append(list(reader))
            print(phone, telecom)


            diff1 = 0.0
            diff2 = 0.0
            diff_time1 = 0.0
            diff_time2 = 0.0
            cnt1 = 0
            cnt2 = 0

            for i in range(len(client[0])):
                RTT = float(client[0][i][1]) - float(client[0][i][0])
                if(RTT > 0.005):
                    continue
                cen_client = (float(client[0][i][1]) + float(client[0][i][0]))/2
                cen_server = (float(server[0][i][1]) + float(server[0][i][0]))/2
                diff1 += cen_server - cen_client
                cnt1 = cnt1+1
            diff1 /= cnt1

            for i in range(len(client[2])):
                RTT = float(client[2][i][1]) - float(client[2][i][0])
                if(RTT > 0.01):
                    continue
                cen_client = (float(client[2][i][1]) + float(client[2][i][0]))/2
                cen_server = (float(server[2][i][1]) + float(server[2][i][0]))/2
                diff2 += cen_server - cen_client
                cnt2 = cnt2+1
            diff2 /= cnt2

            diff_time1 = float(server[0][50][0])
            diff_time2 = float(server[2][-50][0])
            diff = (diff1+diff2)/2
            # print("average time diff =", diff*1000, "ms")
            
            client = client[1]
            server = server[1]

            records = [[0.0, 0.0, -1] for i in range(int(client[-1][0])+1)]
            records1 = [[0.0, 0.0, -1] for i in range(int(client[-1][0])+1)]
            latency = [[] for i in range(300+1)]
            latency1 = [[] for i in range(300+1)]

            if sys_path.find("dlOnly") != -1:
                for record in server:
                    records[int(record[0])][0] = server_time_to_client_time(float(record[1]), diff_time1, diff_time2, diff1, diff2)
                    records[int(record[0])][2] = int(record[2])
                for record in client:
                    if records[int(record[0])][1] > 0:
                        continue
                    records[int(record[0])][1] = float(record[1])

            elif sys_path.find("ulOnly") != -1:
                for record in client:
                    records[int(record[0])][1] = float(record[1])
                    records[int(record[0])][2] = int(record[2])
                for record in server:
                    records[int(record[0])][0] = server_time_to_client_time(float(record[1]), diff_time1, diff_time2, diff1, diff2)

            elif sys_path.find("dlFirst") != -1:
                for record in server:
                    records[int(record[0])][0] = server_time_to_client_time(float(record[1]), diff_time1, diff_time2, diff1, diff2)
                    records[int(record[0])][2] = int(record[3])
                    records1[int(record[0])][0] = server_time_to_client_time(float(record[2]), diff_time1, diff_time2, diff1, diff2)
                    records1[int(record[0])][2] = int(record[3])
                for record in client:
                    if records[int(record[0])][1] > 0:
                        continue
                    records[int(record[0])][1] = float(record[1])
                    records1[int(record[0])][1] = float(record[2])
            elif sys_path.find("ulFirst") != -1:
                for record in client:
                    records[int(record[0])][1] = float(record[1])
                    records[int(record[0])][2] = int(record[3])
                    records1[int(record[0])][1] = float(record[2])
                    records1[int(record[0])][2] = int(record[3])
                for record in server:
                    records[int(record[0])][0] = server_time_to_client_time(float(record[1]), diff_time1, diff_time2, diff1, diff2)
                    records1[int(record[0])][0] = server_time_to_client_time(float(record[2]), diff_time1, diff_time2, diff1, diff2)
                

            print(diff1, diff2)


            if not os.path.exists(sys_path+"rand_sample/"):
                os.mkdir(sys_path+"rand_sample/")
            random.seed(48763)
            i = 0
            while i < num_sample:
                start = random.randint(0, int(client[-1][0])-30)
                origin = min(records[start][0], records[start][1])
                render_list = []
                flag = False

                for j in range(start, int(client[-1][0])+1):
                    if records[j][2] == -1:
                        flag = True
                        break
                    if max(records[j][0], records[j][1]) > origin+2:
                        break
                    render_list.append(j)

                if flag:    # data is wrong
                    continue


                name = " ("
                name += "LTE " if sys_path.find("lte") != -1 else "NR "
                name += "DL only" if sys_path.find("dlOnly") != -1 else ""
                name += "UL only" if sys_path.find("ulOnly") != -1 else ""
                name += "DL first" if sys_path.find("dlFirst") != -1 else ""
                name += "UL first" if sys_path.find("ulFirst") != -1 else ""
                name += ")"
                plt.figure(figsize=(20,6))
                plt.title(f'{phone}_{telecom} {name} {start}')
                plt.xlim(-0.05, 2.05)
                plt.yticks([0.75, 0.25], ["server", "client"], rotation='vertical')
                x_major_locator = MultipleLocator(0.1)
                x_minor_locator = MultipleLocator(0.05)
                plt.gca().xaxis.set_major_locator(x_major_locator)
                plt.gca().xaxis.set_minor_locator(x_minor_locator)

                def format_x_ticks(x, pos):
                    if int(x*20) % 2 == 0:
                        return round(x, 1)
                    else:
                        return ''
                
                x_formatter = FuncFormatter(format_x_ticks)
                plt.gca().xaxis.set_major_formatter(x_formatter)



                for good in render_list:
                    plt.axvline(x=records[good][0]-origin, ymin = 0.5, ymax = 1, color=color_map1[(good-start)%7])
                    plt.axvline(x=records[good][1]-origin, ymin = 0, ymax = 0.5, color=color_map1[(good-start)%7])
                    
                    if sys_path.find("First") != -1:
                        plt.axvline(x=records1[good][0]-origin, ymin = 0.5, ymax = 1, color=color_map1[(good-start)%7])
                        plt.axvline(x=records1[good][1]-origin, ymin = 0, ymax = 0.5, color=color_map1[(good-start)%7])
                        plt.text((records[good][0]+records[good][1])/2-origin, 0.49, str(int(1000*abs(records[good][0]-records[good][1]))), rotation=0, va='center', ha='center')
                        plt.text((records1[good][0]+records1[good][1])/2-origin, 0.51, str(int(1000*abs(records1[good][0]-records1[good][1]))), rotation=0, va='center', ha='center')

                    else:
                        plt.text((records[good][0]+records[good][1])/2-origin, 0.49+0.02*(good%2), str(int(1000*abs(records[good][0]-records[good][1]))), rotation=0, va='center', ha='center')
                    
                    if good == start:
                        continue
                        
                    if sys_path.find("ulOnly") != -1:
                        plt.text((records[good-1][1]+records[good][1])/2-origin, 0.25, str(records[good][2]), rotation=0, va='center', ha='center')
                    elif sys_path.find("dlOnly") != -1:
                        plt.text((records[good-1][0]+records[good][0])/2-origin, 0.75, str(records[good][2]), rotation=0, va='center', ha='center')
                    elif sys_path.find("ulFirst") != -1:
                        plt.text((records1[good-1][1]+records[good][1])/2-origin, 0.25, str(records[good][2]), rotation=0, va='center', ha='center')
                    elif sys_path.find("dlFirst") != -1:
                        plt.text((records1[good-1][0]+records[good][0])/2-origin, 0.75, str(records[good][2]), rotation=0, va='center', ha='center')
                        


                plt.savefig(f'{sys_path}rand_sample/{i+1}.jpg', dpi=600)
                # plt.show()
                plt.close()
                i += 1
                




        except:
            print(phone, telecom, "error")
            continue