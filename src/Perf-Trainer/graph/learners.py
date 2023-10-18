import os
import os.path as path
import csv
import json
import time
import datetime

import math
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter

import graph.train_utils as train_utils
from graph.model import DQN_v0, ReplayMemory
from graph.agents import DQN_AGENT,DQN_AGENT_AB

# Available Freqs GHz
NX_FREQS = np.arange(2,16)/10

"""
RL Learners are responsible for 
1. provide train/inference service
2. logging
3. response to server
Transitions dtype:
    <state(np.float32),action(int64/long),next_state(np.float32),reward(float)>
"""
def dqn_nx(pipe, learn_state, init_params):
    # Agent Initialization
    NAME = "DQN_NX" 
    ROOT = os.path.join(
        "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    log_path = os.path.join(ROOT,"Log")
    model_savepath = os.path.join(ROOT,"Model")
    if not os.path.isdir(log_path): os.makedirs(log_path)
    train_logger = SummaryWriter(log_path)

    # training hyper-parameters
    EPS_START = 0.99
    EPS_END = 0.2
    EPS_DECAY = 1000

    n_update, n_batch = 20,100
    SYNC_STEP = 30
    N_S, N_A, N_BUFFER = 13, 1, 12000
    learn_state.value = 1
    AGENT = DQN_AGENT(N_S,N_A,N_BUFFER,None)
    # AGENT.load_model("./db/DQN/test/Model")
    # Reset States
    prev_state, prev_action = [None]*2
    record_count, test_count, n_round, g_step = [0]*4
    # Response ready yo server
    pipe.send("ready")

    while True:
        # wait for command
        msg = pipe.recv()
        cmd = msg['cmd']
        print("receive pipe message {} from {}".format(cmd,os.getpid()))

        if cmd == "RECORD":
            # Extract state(require np.float32), rewards(float)
            state, reward = get_ob(msg['data'])
            
            # Inference
            AGENT.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))
            print("reward is {}".format(reward))
            pipe.send({"action":int(action)})

            # Add transition (state, action, next_state, reward) into replay buffer
            if record_count!=0:
                AGENT.mem.push(prev_state,prev_action,state,reward)
            prev_state, prev_action = state, action

            # Write data of a sample slot logFile.write(...)
            if record_count==0:
                record_log_file = create_log(ROOT,"RECORD")
                record_log = open(record_log_file,'w',newline='')
                record_writer = csv.DictWriter(record_log, msg['data'].keys())
                record_writer.writeheader()
            record_writer.writerow(msg['data'])
            record_log.flush()
            record_count+=1

        elif cmd == "TRAIN":
            learn_state.value = 0 # disable inference
            # train loop
            losses = AGENT.train(n_round,n_update,n_batch)
            g_step = train_utils.log_scalar_list(train_logger, "Train/Loss", g_step, losses)
            # Reset initial states/actions to None
            prev_state,prev_action,record_count = None,None,0
            if record_log: record_log.close()
            # save model
            AGENT.save_model(n_round, model_savepath)
            n_round += 1
            if n_round % SYNC_STEP == 0: AGENT.sync_model()
            learn_state.value = 1

        elif cmd == "TEST":
            # Extract state 
            state,reward = get_ob(msg['data'])
            # Inference
            AGENT.eps = 0
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))

            pipe.send({"action":int(action)})
            if test_count==0:
                test_log_file = create_log(ROOT,"TEST")
                test_log = open(test_log_file,'w',newline='')
                test_writer = csv.DictWriter(test_log, msg['data'].keys())
                test_writer.writeheader()
            test_writer.writerow(msg['data'])
            test_count+=1
            test_log.flush()

        elif cmd == "END_TEST":
            # Reset test count
            test_count = 0
            if test_log: test_log.close()
            # Extract power and total time cost 
            power = cal_xu3power(test_log_file)
            train_logger.add_scalars("Test",power,n_round)
            pipe.send(power)


def dqn_phone(pipe, learn_state, init_params):
    # Agent Initialization
    NAME = "DQN_PHONE" 
    # ROOT = os.path.join(
    #     "./db/", NAME, str(datetime.datetime.now().strftime('%m%d-%H%M')))
    ROOT = os.path.join(
        "./db/", NAME, "test")
    log_path = os.path.join(ROOT,"Log")
    model_savepath = os.path.join(ROOT,"Model")
    if not os.path.isdir(log_path): os.makedirs(log_path)
    train_logger = SummaryWriter(log_path)

    # training hyper-parameters
    EPS_START = 0.99
    EPS_END = 0.2
    EPS_DECAY = 1000

    n_update, n_batch = 5,4
    SYNC_STEP = 30
 
    N_S,  N_BUFFER = 18, 36000
    learn_state.value = 1
    # TODO Analysis
    AGENT = DQN_AGENT_AB(N_S,8,[16,16,16,40],N_BUFFER,None)
    AGENT.load_model(model_savepath)
    # Reset States
    prev_state, prev_action = [None]*2
    record_count, test_count, n_round, g_step = [0]*4
    # Response ready yo server
    pipe.send("ready")

    while True:
        # wait for command
        msg = pipe.recv()
        cmd = msg['cmd']
        print("receive pipe message {} from {}".format(cmd,os.getpid()))

        if cmd == "RECORD":
            # Extract state(require np.float32), rewards(float)
            state, reward = get_ob_phone(msg['data'])
            
            # Inference
            AGENT.eps = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * g_step / EPS_DECAY)
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))
            print("reward is {}, action is {}".format(reward,action))
            pipe.send({"action":action})

            # Add transition (state, action, next_state, reward) into replay buffer
            if record_count!=0:
                AGENT.mem.push(prev_state,prev_action,state,reward)
            prev_state, prev_action = state, action

            # Write data of a sample slot logFile.write(...)
            if record_count==0:
                record_log_file = create_log(ROOT,"RECORD")
                record_log = open(record_log_file,'w',newline='')
                record_writer = csv.DictWriter(record_log, msg['data'].keys())
                record_writer.writeheader()
            record_writer.writerow(msg['data'])
            record_log.flush()
            record_count+=1

        elif cmd == "TRAIN":
            learn_state.value = 0 # disable inference
            # train loop
            losses = AGENT.train(n_round,n_update,n_batch)
            g_step = train_utils.log_scalar_list(train_logger, "Train/Loss", g_step, losses)
            # Reset initial states/actions to None
            prev_state,prev_action,record_count = None,None,0
            if record_log: record_log.close()
            # save model
            AGENT.save_model(n_round, model_savepath)
            n_round += 1
            if n_round % SYNC_STEP == 0: AGENT.sync_model()
            learn_state.value = 1

        elif cmd == "TEST":
            # Extract state 
            state,reward = get_ob_phone(msg['data'])
            # Inference
            AGENT.eps = 0
            action = AGENT.select_action(torch.from_numpy(state).unsqueeze(0))

            pipe.send({"action":action})
            if test_count==0:
                test_log_file = create_log(ROOT,"TEST")
                test_log = open(test_log_file,'w',newline='')
                test_writer = csv.DictWriter(test_log, msg['data'].keys())
                test_writer.writeheader()
            test_writer.writerow(msg['data'])
            test_count+=1
            test_log.flush()

        elif cmd == "END_TEST":
            # Reset test count
            test_count = 0
            if test_log: test_log.close()
            # Extract power and total time cost 
            power = cal_xu3power(test_log_file)
            train_logger.add_scalars("Test",power,n_round)
            pipe.send(power)

def create_log(root, name):
    log_dir = os.path.join(root,name)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    log_file = str(datetime.datetime.now().strftime('%m%d-%H%M'))
    log_file = os.path.join(log_dir,log_file) + ".csv"
    return log_file

def get_ob(log_data):
    s = np.random.random(13).astype(np.float32)
    r = 1.0
    return s,r

import math
def cal_cpu_reward(cpu_utils,cpu_temps,cluster_num):
    lambda_value = 0.1
    # for cpu
    cpu_u_max,cpu_u_min = 0.85,0.75
    cpu_u_g = 0.8
    u,v,w = -0.1,0.11,0.1
    temp_thre = 60
    reward_value = 0.0
    print('cpu',end=': ')
    for cpu_u,cpu_t in zip(cpu_utils,cpu_temps):
        if cpu_u < cpu_u_min and cpu_u > cpu_u_max:
            d =lambda_value
        else:
            d = u+v*math.exp(-(cpu_u-cpu_u_g)**2 / (w ** 2))
        if cpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre-cpu_t)
        else:
            w = -2
        reward_value += d
        print(f"{cpu_u}:{d}",end=',')
    
    return reward_value/cluster_num
  
def cal_gpu_reward(gpu_utils,gpu_temps,num):
    lambda_value = 0.1
    # for cpu
    gpu_u_max,gpu_u_min = 0.85,0.75
    gpu_u_g = 0.8
    u,v,w = -0.1,0.11,0.1
    temp_thre = 60
    reward_value = 0
    print('gpu',end=': ')
    for gpu_u,gpu_t in zip(gpu_utils,gpu_temps):
        if gpu_u < gpu_u_min and gpu_u > gpu_u_max:
            d =lambda_value
        else:
            d = u+v*math.exp(-(gpu_u-gpu_u_g)**2 / (w ** 2))
        if gpu_t < temp_thre:
            w = 0.2 * math.tanh(temp_thre-gpu_t)
        else:
            w = -2
        reward_value += d
        print(f"{gpu_u}:{d}",end=',')
    return reward_value/num 

def get_ob_phone(log_data):
    # State Extraction and Reward Calculation
    gpu_util = log_data["gpu_u"]
    cpu_util = log_data["cpu_u"]
    gpu_freq = log_data["gpu_f"]
    cpu_freq = log_data["cpu_f"]
    gpu_thremal= log_data["gpu_t"]
    cpu_thremal = log_data["cpu_t"]
    power = log_data["power"]
    
    # 16个数据
    states = np.concatenate([gpu_util,cpu_util,gpu_freq,cpu_freq,gpu_thremal,cpu_freq,power]).astype(np.float32)
    
    reward = cal_cpu_reward(cpu_util,cpu_thremal,8)
    reward += cal_gpu_reward(gpu_util,gpu_thremal,1)
    print()
    return states,reward

def get_ob_xu3(log_data):
    # State Extraction and Reward Calculation
    NUM_CPU, NUM_TMU, CLUSTER_IDS = 8, 5, [0,4]
    thermal = 0
    for k in range(NUM_TMU):
        thermal += log_data["THERMAL_{}".format(k)]
    # Normalized key states
    avg_thermal = [thermal / NUM_TMU / 1000 / 40]
    cluster_freqs = [log_data["CLUSTER_0_FREQ"]/1e6, log_data["CLUSTER_4_FREQ"]/1e6]
    cpu_powers = [log_data["A7_W"], log_data["A15_W"]]
    cpu_utils = [log_data["CPU{}_UTIL".format(k)] for k in range(4,8)]
    cpu_ipcs = [log_data["CPU{}_instructions".format(k)]/log_data["CPU{}_cycles".format(k)] for k in range(4,8)] 
    states = np.concatenate([avg_thermal, cluster_freqs, cpu_powers, cpu_utils, cpu_ipcs]).astype(np.float32)

    def cal_reward(utils, freqs, powers, thermal,cpu_ipcs):
        max_util = max(np.array(utils))
        high_util, low_util = 0.9, 0.8
        max_freq, min_freq = 2.0, 0.2
        max_cap, min_cap = high_util*max_freq, low_util*min_freq
        if max_util*cluster_freqs[1]>max_cap or max_util*cluster_freqs[1]<min_cap:
            r_util = 0.05
        else:
            if max_util>low_util and max_util<high_util:
                r_util = 1.2
            else:
                r_util = -0.5
        r_max = 90/40
        r_thermal = 0.1 if thermal<r_max else -1.5
        avg_ipc = np.mean(np.array(cpu_ipcs))
        # return -powers[1]*1.15 + avg_ipc
        return r_util
        return -powers[1]*1.15 + r_util
        return r_util + 2/powers[1] + r_thermal
    reward = cal_reward(cpu_utils, cluster_freqs, cpu_powers, avg_thermal[0],cpu_ipcs)

    s = np.random.random(5).astype(np.float32)
    r = 1.0
    return states,reward

def cal_xu3power(log_file):
    df = pd.read_csv(log_file)
    t = df.Time_Stamp.values # Series to numpy
    dt = t[-1] - t[0]
    w15, w7 = df.A15_W, df.A7_W
    p15, p7 = np.trapz(w15,t), np.trapz(w7,t)
    return {"t":dt,"p":p15+p7,"p15":p15,"p7":p7}


if __name__ == "__main__":
    print("HI")
