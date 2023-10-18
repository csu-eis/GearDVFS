from utils.dimensity8050_utils import *
import time

'''
Monitoring Features
    Process Variables:
        - CPU Utilization
    State Variables:
        - Power
        - Temperature
        - Frequency(optional)
'''

class WindowAverageMeter(object):
    def __init__(self):
        pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AndroidMonitor(object):
    prev_cpu_time=([0 for i in range(8)],[0 for i in range(8)])
    def __init__(self, ip,config):
        self.ip = ip
        self.num_cpu = check_cpus(config)
        print("Monitor Process Started for {}".format(config["device"]["name"]))
        print("Adb connect for {}".format(config["device"]["ip"]))
        print("Num of online cpus: {}".format(self.num_cpu))

        # self.powers, self.thermals = [[]]*2

        # for i, k in enumerate(config['power']):
        #     node = config['power'][k]
        #     self.powers.append({
        #         "node":node,"f":open(node,'r'),"name":k,"m":AverageMeter()})

        # for i in range(2):
        #     node = config['thermal']['temp'].replace("$$",str(i))
        #     name = get_value(config['thermal']['t_type'].replace("$$",str(i)))
        #     self.thermals.append({
        #         "node":node,"f":open(node,'r'),"name":name,"m":AverageMeter()})

        # # self.prev_cpu_time = get_core_time(self.num_cpu)
        # self.gpu_util_node = open(config['gpu']['load'],'r')
        # # frequencies
        # self.gpu_freq_node = open(config['gpu']['freq'],'r')
        # self.cpu_cluster_id = open(config['cpu']['cluster_ids'],'r')
        # self.cpu_freq_node = open(config['cpu']['freq'].replace("$$",str(0)),'r')

    # def __sample(self):
    #     # Sample Powers
    #     for domain in [self.powers, self.thermals]:
    #         for item in domain:
    #             val = read_value(item['f'])
    #             item["m"].update(float(val))

    def reset(self):
        pass
        # for domain in [self.powers, self.thermals]:
        #     for item in domain: item["m"].reset()
        # # self.prev_cpu_time = get_core_time(self.num_cpu)
        # self.__sample()

    def get_gpu_loading(self):
        res = adb_shell(self.ip,"cat /proc/gpufreq/gpufreq_var_dump")
        pattern = r'gpu_loading = (\d+)'

        matches = re.findall(pattern, res)
        return [float(matches[0])]
    
    def get_cpu_loading(self):
        # if not self.prev_cpu_time:
        #     self.prev_cpu_time=(0,0)
        core_util,self.prev_cpu_time=parse_core_util(self.ip,self.prev_cpu_time,8)
        return core_util
      
    def get_gpu_freq(self):
        res = adb_shell(self.ip,"cat /proc/gpufreq/gpufreq_var_dump")
        pattern = r'freq: (\d+)'
        match = re.search(pattern, res)

        if match:
            freq_value = match.group(1)
            return [int(freq_value)]
        
    def get_cpu_freq(self,idx):
        res = adb_shell(self.ip,f"cat /sys/devices/system/cpu/cpufreq/policy{idx}/scaling_cur_freq")
        res = res.replace("\n","")
        return int(res)
    
    def get_cpu_temp(self):
        res = adb_shell(self.ip,f"cat /sys/class/thermal/thermal_zone4/temp")
        res = res.replace("\n","")
        return [float(res)/1000.0]
    
    def get_gpu_temp(self):
        res = adb_shell(self.ip,f"cat /sys/class/thermal/thermal_zone4/temp")
        res = res.replace("\n","")
        return [float(res)/1000.0] # 摄氏度

    def get_power(self):
        res = adb_shell(self.ip,f"cat /sys/class/power_supply/battery/current_now")
        current = float(res.replace("\n","")) * 1.0 / 1e3
        res = adb_shell(self.ip,f"cat /sys/class/power_supply/battery/voltage_now")
        voltage = float(res.replace("\n","")) * 1.0 / 1e6
        return [float(abs(current)*abs(voltage))] # mW
    
    


    def query(self):
        # utils, self.prev_cpu_time = parse_core_util(self.prev_cpu_time,self.num_cpu)
        # self.__sample()
        query_result = {}

        query_result["gpu_u"] = self.get_gpu_loading()
        query_result["cpu_u"] = self.get_cpu_loading()
        query_result["gpu_f"] = self.get_gpu_freq()
        query_result["cpu_f"] = [self.get_cpu_freq(0),self.get_cpu_freq(4),self.get_cpu_freq(7)]
        query_result["cpu_t"] = self.get_cpu_temp()
        query_result["gpu_t"] = self.get_gpu_temp()
        query_result["power"] = self.get_power()
        return query_result


def monitor_daemon(config,conn):
    monitor = AndroidMonitor(config)
    while True:
        pass