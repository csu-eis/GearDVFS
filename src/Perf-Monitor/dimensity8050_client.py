import os
import os.path as path
import time
import datetime
import random
import configparser
import subprocess
from multiprocessing import Process, Pipe
import numpy as np
from utils.dimensity8050_utils import *
from utils.nn_api import *
from utils.dimensity8050_monitor import AndroidMonitor

# Device specific constants
AVAIL_GOVS_CPU = ["ondemand","schedutil","userspace","powersave","performance"]
AVAIL_GOVS_GPU = ["simple_ondemand","userspace"]
# AVAIL Benchmarks
AVAIL_BENCHS = {
    "video":"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lcd/ffmpeg_build/lib python3 ./benchmarks/render/player.py",
    "yolo":None,
    "cpu":"sysbench --test=cpu --num-threads=3 --cpu-max-prime=20000 run",
    "mem":"sysbench --test=memory --num-threads=1 --memory-block-size=2G --memory-total-size=100G run",
    "io":'sysbench --threads=16 --events=10000 fileio --file-total-size=3G --file-test-mode=rndrw run',
    "thread":'sysbench threads --time=5 --thread-yields=2000 --thread-locks=2 --num-threads=64  run',
    }
# There are also some hw cache events
AVAIL_EVENTS = ["cycles","instructions","cache-misses","cache-references"]
# AVAIL FREQS
AVAIL_FREQS = [i*100000 for i in range(2,21)]

# time in us(microseconds)
PERF_TIME = 10*1e3

TARGET_GOV = AVAIL_GOVS_CPU[1]
TARGET_GOV_GPU = AVAIL_GOVS_GPU[0]
TARGET_BENCH = AVAIL_BENCHS["io"]
TARGET_BENCH = "sleep 60"
BENCH_EPOCH, TEST_EPOCH = 2, 3
def main():
    # Load Configurations
    device = "src/Perf-Monitor/configs/camon20_5g"
    config = configparser.ConfigParser()
    config.read("{}.ini".format(device))

    # Determine Perf Events
    cpus = [0,1,2,3,4,5]
    events = [0,4,5] # ["cycles","stalled-cycles-front", "stalled-cycles-back"]

    # Switch cpufreq governor to userspace
    AVAIL_CPU_FREQS = getAvailableClockCPU(ip,idx=0)
    onlineAllCPUs(ip,8)
    switchGovernorCPU(ip,0,"userspace")
    switchGovernorCPU(ip,4,"userspace")
    switchGovernorCPU(ip,7,"userspace")
    
    # Switch gpu devfreq governor
    AVAIL_GPU_FREQS = getAvailableClockGPU(ip)
    print(AVAIL_GPU_FREQS)
    print(AVAIL_FREQS)
    # Create monitor
    monitor = AndroidMonitor(ip,config) 

    count, start_time = 0, time.time()
    # Benchmark training loop
    
    ONLY_TRAIN = True
    ONLY_TEST = False
    
    gpu_u = []
    cpu_u = []
    cpu_t = []
    gpu_t = []
    power = []
    
    for epoch in range(1, BENCH_EPOCH):
        TRAIN = False if epoch%TEST_EPOCH==0 else True
        # Start bechmark process
        # bench_proc = subprocess.Popen(TARGET_BENCH,shell=True)
        # Sampling loop
        record_count = 0
        iter =  5
        while(True):
            # Reset Monitor
            t = time.time()
            # Run perf cmd & Request monitor data
            try:
                log_data, pmus = sample(config, monitor, events, cpus, PERF_TIME)
                iter+=1
                gpu_u.append(log_data["gpu_u"])
                cpu_u.append(log_data["cpu_u"])
                cpu_t.append(log_data["cpu_t"])
                gpu_t.append(log_data["gpu_t"])
                power.append(log_data["power"])
                # print(log_data)
                if iter < 5:
                    time.sleep(0.2)
                    continue
                iter = 0
                log_data["gpu_u"] = np.mean(np.asanyarray(gpu_u),axis=0).tolist()
                log_data["cpu_u"] = np.mean(np.asanyarray(cpu_u),axis=0).tolist()
                log_data["cpu_t"] = np.mean(np.asanyarray(cpu_t),axis=0).tolist()
                log_data["gpu_t"] = np.mean(np.asanyarray(gpu_t),axis=0).tolist()
                log_data["power"] = np.mean(np.asanyarray(power),axis=0).tolist()
                if ONLY_TRAIN:
                    res = get_action(url_base,m_info,{
                    "data":log_data
                    })
                    c0,c4,c7,g = res["action"]
                    setSoCFreq(ip,c0,c4,c7,g)
                    print(log_data)
                    record_count+=1
                    # 每5个数据训练一次
                    if (record_count%5==0 and record_count!=0):
                        res = request_update(url_base,m_info)
                elif ONLY_TEST:
                    res = get_action_test(url_base,m_info,{
                        "data":log_data
                    })
                    c0,c4,c7,g = res["action"]
                    setSoCFreq(ip,c0,c4,c7,g)
                    print(c0,c4,c7,g)
            except Exception as err:
                print("Perf Failed with Err: {}".format(err))
                return
            # print(time.time()-t)

 

    # Reset to default governors
    switchGovernorCPU(ip,0,"schedutil")
    switchGovernorCPU(ip,4,"schedutil")
    switchGovernorCPU(ip,7,"schedutil")
    switchGovernorGPU(ip,"simple_ondemand")
    # set_value(config["cpu"]["gov"].replace("$$",str(0)),'schedutil')
    # set_value(config['gpu']['gov'], 'simple_ondemand')

if __name__ == '__main__':
    ip = "172.16.101.79:5555"
    url_base="http://172.16.101.75:5000/nn"
    
    # init_params = {
    #     # "m_type": "DQN"
    # }
    m_info = {
        "m_type":"DQN_PHONE",
    }
    
    res = init_model(url_base,m_info,{})
    m_info["m_id"] = res["m_id"]
    # print(res)
    # res = clear_all(url_base)
    # print(res)
    main()
