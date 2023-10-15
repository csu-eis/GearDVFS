import os
import os.path as path
import time
import datetime
import random
import configparser
import subprocess
from multiprocessing import Process, Pipe

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
    for epoch in range(1, BENCH_EPOCH):
        TRAIN = False if epoch%TEST_EPOCH==0 else True
        # Start bechmark process
        # bench_proc = subprocess.Popen(TARGET_BENCH,shell=True)
        # Sampling loop
        while(True):
            # Reset Monitor
            t = time.time()
            # Run perf cmd & Request monitor data
            # try:
            # monitor.reset()
            log_data, pmus = sample(config, monitor, events, cpus, PERF_TIME)
            print(log_data, pmus)
                # log_data = step_sample(config,start_time,perf_cmd,parent_conn)
            # except Exception as err:
            #     print("Perf Failed with Err: {}".format(err))
            #     return
            print(time.time()-t)

 

    # Reset to default governors
    switchGovernorCPU(ip,0,"schedutil")
    switchGovernorCPU(ip,4,"schedutil")
    switchGovernorCPU(ip,7,"schedutil")
    switchGovernorGPU(ip,"simple_ondemand")
    # set_value(config["cpu"]["gov"].replace("$$",str(0)),'schedutil')
    # set_value(config['gpu']['gov'], 'simple_ondemand')

if __name__ == '__main__':
    ip = "172.16.101.79:5555"
    url_base="http://172.16.101.75"
    main()
