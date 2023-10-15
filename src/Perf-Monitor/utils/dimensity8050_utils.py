import subprocess
import time
# import utils.perf_lib.PyPerf as Perf
import os
import re

def adb_shell(ip,cmd):
    cmd = f'adb -s {ip} shell "{cmd}"'
    res = os.popen(cmd).read()
    # print()
    return res

def sample(config, monitor, events, cpus, t):
    # raw = Perf.sys_perf(cpus, events, int(t))
    log_data = {}
    log_data = monitor.query()
    return log_data, None

def onlineAllCPUs(ip,num_cpu):
    for idx in range(0,num_cpu):
        res = adb_shell(ip,f"echo 1 > /sys/devices/system/cpu/cpu{idx}/online")

def switchGovernorCPU(ip,idx,governor):
    if governor not in ["schedutil","userspace"]:
        return 
    res = adb_shell(ip,f"echo {governor} > /sys/devices/system/cpu/cpufreq/policy{idx}/scaling_governor")

def switchGovernorGPU(ip,governor):
    if governor == "simple_ondemand":
        res = adb_shell(ip,f"echo 0 > /proc/gpufreq/gpufreq_opp_freq")

def getAvailableClockCPU(ip,idx=0):
    res = adb_shell(ip,f"cat /sys/devices/system/cpu/cpufreq/policy{idx}/scaling_available_frequencies")
    items = res.split(" ")
    clk_list = [int(i) for i in items if i!='' and i!='\n']
    clk_list.reverse()  # 保证低档位是低频率
    return clk_list

def getAvailableClockGPU(ip):
    res = adb_shell(ip,f"cat /proc/gpufreq/gpufreq_opp_dump")
    pattern = r'freq = (\d+)'
    matches = re.findall(pattern, res)
    clk_list = [int(s) for s in matches]
    clk_list.reverse() # 保证低档位是低频率
    return clk_list

# CPU Cores Utilization
def parse_core_util(ip,prev_cpu_time, num_cpu):
    last_idles, last_totals = prev_cpu_time   
    lines = adb_shell(ip,"cat /proc/stat").split("\n")
    utils = []
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idle, total = fields[3], sum(fields)
        idle_delta, total_delta = idle - last_idles[i], total - last_totals[i]
        last_idles[i], last_totals[i] = idle, total
        utilization = 1.0 - idle_delta / total_delta
        utils.append(utilization)
    return utils, (last_idles, last_totals)

def get_core_time(ip,num_cpu):
    lines = adb_shell(ip,"cat /proc/stat").split("\n")
    # with open('/proc/stat') as f: 
    #     lines = f.readlines()
    idles,totals = [0]*num_cpu, [0]*num_cpu
    for i, l in enumerate(lines[1:num_cpu+1]):
        fields = [float(column) for column in l.strip().split()[1:]]
        idles[i] = fields[3]
        totals[i] = sum(fields)
    return idles, totals

def check_cpus(config):
    # TODO
    return 8

# File operations for sysfs nodes
# def get_value(file):
#     with open(file, 'r') as f:
#         text = f.read().strip("\n")
#     return text

# def read_value(f):
#     f.seek(0)
#     text = f.read().strip("\n")
#     return text

# def set_value(file,v):
#     with open(file,'w') as f:
#         f.write(str(v))
#     return 0

def s2i(s):
    # convert string to int
    i = int(s.replace(',',''))
    return i


if __name__=="__main__":
    ip = "172.16.101.79:5555"
    res = get_core_time(ip,8)
    print(res)