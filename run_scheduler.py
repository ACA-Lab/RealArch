import pandas as pd
import numpy as np
import pdb
import random
import copy
import glob
import csv
import os
import torch
from itertools import permutations, product, combinations
acc_pool = ['tpu', 'shidiannao', 'eyeriss']

def read_csv_numpy(fname):
    df = pd.read_csv(fname)
    return df.to_numpy()


def compute_buffer_use(esti_res, idx, bandwidth):
    DM_idx = np.array([i * 3 for i in range(int(esti_res.shape[1] / 3))])
    buffer_use = esti_res[idx, :][:, DM_idx] * bandwidth
    return buffer_use

def select_jobs(esti_res, bandwidth, job_num = 10):
    job_total_num = esti_res.shape[0]
    if job_total_num <= job_num:
        idx = np.arange(job_total_num)
    else:
        idx = sorted(np.random.choice(job_total_num, job_num, replace = False))

    buffer_use = compute_buffer_use(esti_res, idx, bandwidth)
    DM_idx = np.array([i * 3 for i in range(int(esti_res.shape[1] / 3))])
    EX_idx = DM_idx + 1
    Iter_idx = EX_idx + 1
    DM_EX_idx = sorted(np.concatenate([DM_idx, EX_idx]))
    esti_res[:, DM_idx] = esti_res[:, DM_idx] * 256 / bandwidth
    jobs = esti_res[idx, :][:, DM_EX_idx] * esti_res[idx, :][:, Iter_idx].repeat(2, axis = 1)

    return jobs, buffer_use

def dynamic_input(jobs_list, index_list, bandwidth, batch_set):
    es_list = []
    for i, [job, idx, bi] in enumerate(zip(jobs_list, index_list, batch_set)):

        if idx == job.shape[0]:
            batch_set[i] -= 1
            idx = 0
            index_list[i] = 0
        
        if bi > 0 and idx < job.shape[0]:
            job_temp = job[idx, :]
        else:
            job_temp = np.full(job[0, :].shape, float('Inf'))
        
        
        
        es_list.append(job_temp)

    esti_res = np.stack(es_list, 0)
    jobs, buffer_use = select_jobs(esti_res, bandwidth, job_num = esti_res.shape[0])

    return jobs, buffer_use
    
def pop_slot_num(net_work_list, bandwidth, slot_num):
    es_list = []
    net_work_list_temp = []
    for job in net_work_list:
        if np.all(job == []):
            continue
        else:
            net_work_list_temp.append(job)
    
    net_work_list = net_work_list_temp
            
    net_work_num = torch.tensor([i.shape[0] for i in net_work_list])
    pop_num = (net_work_num > 0).sum() if (net_work_num > 0).sum() < slot_num else slot_num

    _, indices = net_work_num.topk(pop_num, dim = 0, largest = True, sorted = False)
    indices = sorted(indices)
    for idx in indices:
        nw_list = net_work_list[idx].tolist()
        job_np = nw_list.pop(0)
        es_list.append(job_np)

        net_work_list[idx] = np.array(nw_list)
    
    esti_res = np.stack(es_list, 0)
    

    jobs, buffer_use = select_jobs(esti_res, bandwidth, job_num = esti_res.shape[0])
        
    return jobs, buffer_use, net_work_list

def pop_max(net_work_list, bandwidth, acc_num):
    
    es_list = []
    net_work_list_temp = []
    for job in net_work_list:
        if np.all(job == []):
            continue
        else:
            net_work_list_temp.append(job)
    
    net_work_list = net_work_list_temp

    if net_work_list == []:
        single_job = np.full(acc_num * 2, float('Inf'))
        single_buffer_use = np.full(acc_num, float('Inf'))
        return single_job, single_buffer_use, net_work_list

    net_work_num = np.array([i.shape[0] for i in net_work_list])
    
    idx = np.argmax(net_work_num)
    nw_list = net_work_list[idx].tolist()
    job_np = nw_list.pop(0)
    es_list.append(job_np)
    net_work_list[idx] = np.array(nw_list)

    esti_res = np.array(es_list)
    single_job, single_buffer_use = select_jobs(esti_res, bandwidth, job_num = esti_res.shape[0])

    return single_job, single_buffer_use, net_work_list


def baseline_schedule(net_work_list, batch_set, layer_num, buffer_size, bandwidth, slot_num):
    job_num = (np.array(batch_set) * np.array(layer_num)).sum()
    net_work_list_this = copy.deepcopy(net_work_list)
    jobs, buffer_use, net_work_list_this = pop_slot_num(net_work_list_this, bandwidth, slot_num)

    acc_num = len(buffer_size)
    remain_buffer = copy.deepcopy(buffer_size)
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0
    finished_job = 0
    assigned_job = 0
    core_job = [[] for i in range(acc_num)]

    job_idx = 0
    acc_idx = 0
    
    job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
    time_frame[0] = job[0]
    time_frame[acc_idx + 1] = job[0] + job[1]
    remain_buffer[acc_idx] = remain_buffer[acc_idx] - buffer_use[job_idx, acc_idx]
    assigned_job +=1
    core_job[acc_idx].append(job_idx)
    
    while(1):
        if assigned_job < job_num:

            min_frame = time_frame.min()
            min_idx = np.argmin(time_frame)
        else:
            time_frame[time_frame == float('Inf')] = 0
            total_time_frame = time_frame.max()
            bounded_acc = np.argmax(time_frame)
            break
       

        if min_idx == 0:
            single_job, single_buffer_use, net_work_list_this = pop_max(net_work_list_this, bandwidth, acc_num)
            
            if np.all(single_job == float('Inf')):
                jobs[job_idx,:] = float('Inf')
                buffer_use[job_idx,:] = float('Inf')
            else:
                jobs[job_idx,:] = single_job
                buffer_use[job_idx,:] = single_buffer_use

            job_idx = job_idx + 1 if job_idx + 1 < slot_num else 0
            
            while(1):
                if jobs[job_idx, 0] == float('Inf'):
                    job_idx = job_idx + 1 if job_idx + 1 < slot_num else 0
                else:
                    break
            acc_idx = acc_idx + 1 if acc_idx + 1 <= (acc_num - 1) else 0
            
            core_job[acc_idx].append(job_idx)

            job = jobs[job_idx, acc_idx*2:acc_idx*2+2]

            if buffer_use[job_idx, acc_idx] <= remain_buffer[acc_idx]:
                time_frame[0] += job[0]
                if time_frame[acc_idx + 1] == float('Inf'):
                    time_frame[acc_idx + 1] = time_frame[0] + job[1]
                    
                else:
                    acc_start_latency = max(time_frame[0], time_frame[acc_idx + 1]) - time_frame[acc_idx + 1]
                    time_frame[acc_idx + 1] += acc_start_latency + job[1]
                remain_buffer[acc_idx] -= buffer_use[job_idx, acc_idx]
            else:

                time_frame[0] = time_frame[acc_idx + 1] + job[0]
                time_frame[acc_idx + 1] = time_frame[0] + job[1]
                remain_buffer[acc_idx] = buffer_size[acc_idx] - buffer_use[job_idx, acc_idx]
            
            assigned_job += 1
        
        if min_idx > 0 and min_idx <= acc_num:

            finished_job += 1
            
            time_frame[min_idx] = float('Inf')
            remain_buffer[min_idx - 1] = buffer_size[min_idx - 1]
        
        if min_idx < 0 or min_idx > acc_num:
            raise IndexError("scheduler 索引溢出")
    

    return total_time_frame, bounded_acc, core_job


def greedy_scheduler(net_work_list, batch_set, layer_num, buffer_size, bandwidth, slot_num):
    
    acc_num = len(buffer_size)
    job_num = slot_num
    total_job_num = (np.array(batch_set) * np.array(layer_num)).sum()
    net_work_list_this = copy.deepcopy(net_work_list)
    jobs, buffer_use, net_work_list_this = pop_slot_num(net_work_list_this, bandwidth, slot_num)
    
    remain_buffer = copy.deepcopy(buffer_size)
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0
    finished_job = 0
    assigned_job = 0
    core_job = [[] for i in range(acc_num)]
    scheduler_list = []

    DM_list_idx = np.arange(0, acc_num) * 2
    EX_list_idx = DM_list_idx + 1

    greedy_key = jobs[:, DM_list_idx] + jobs[:, EX_list_idx]
    job_idx, acc_idx = np.unravel_index(np.argmin(greedy_key, axis=None), greedy_key.shape)
    core_job[acc_idx].append(job_idx)
    scheduler_list.append([job_idx, acc_idx])

    job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
    time_frame[0] = job[0]
    time_frame[acc_idx + 1] = job[0] + job[1]

    remain_buffer[acc_idx] = remain_buffer[acc_idx] - buffer_use[job_idx, acc_idx]
    assigned_job +=1

    single_job, single_buffer_use, net_work_list_this = pop_max(net_work_list_this, bandwidth, acc_num)
    jobs[job_idx,:] = single_job
    buffer_use[job_idx,:] = single_buffer_use    

    while(1):

        if assigned_job < total_job_num:
            min_frame = time_frame.min()
            min_idx = np.argmin(time_frame)
        else:
            time_frame[time_frame == float('Inf')] = 0
            total_time_frame = time_frame.max()
            bounded_acc = np.argmax(time_frame)
            
        if min_idx == 0:
            
            time_frame_new = np.where(time_frame == float('Inf'), 0.0, time_frame)
            acc_frame_array = time_frame_new[1:].reshape(1, time_frame_new[1:].shape[0]).repeat(job_num, axis = 0)
            remain_buffer_array = remain_buffer.reshape(1, remain_buffer.shape[0]).repeat(job_num, axis = 0)
            buffer_mask = buffer_use <= remain_buffer_array
            buffer_size_array = buffer_size.reshape(1, buffer_size.shape[0]).repeat(job_num, axis = 0)

            less_DM_frame = jobs[:, DM_list_idx] + time_frame[0]
            acc_start_latency = np.where(less_DM_frame <= acc_frame_array, 0, less_DM_frame - acc_frame_array)
            less_EX_frame = acc_frame_array + acc_start_latency + jobs[:, EX_list_idx]
            less_remain_buffer = remain_buffer_array - buffer_use

            more_DM_frame = acc_frame_array + jobs[:, DM_list_idx]
            more_EX_frame = more_DM_frame + jobs[:, EX_list_idx]
            more_remain_buffer = buffer_size_array - buffer_use

            DM_frame = np.where(buffer_mask == True, less_DM_frame, more_DM_frame)
            remain_buffer_array = np.where(buffer_mask == True, less_remain_buffer, more_remain_buffer)
            greedy_key = np.where(buffer_mask == True, less_EX_frame, more_EX_frame)
            job_idx, acc_idx = np.unravel_index(np.argmin(greedy_key, axis=None), greedy_key.shape)
            core_job[acc_idx].append(job_idx)
            scheduler_list.append([job_idx, acc_idx])

            time_frame[0] = DM_frame[job_idx, acc_idx]
            time_frame[acc_idx + 1] = greedy_key[job_idx, acc_idx]

            remain_buffer[acc_idx] = remain_buffer_array[job_idx, acc_idx]
            assigned_job +=1

            single_job, single_buffer_use, net_work_list_this = pop_max(net_work_list_this, bandwidth, acc_num)
            jobs[job_idx,:] = single_job
            buffer_use[job_idx,:] = single_buffer_use     
            
        
        if min_idx > 0 and min_idx <= acc_num:

            finished_job += 1
            
            time_frame[min_idx] = float('Inf')
            remain_buffer[min_idx - 1] = buffer_size[min_idx - 1]
        
        if min_idx < 0 or min_idx > acc_num:
            raise IndexError("scheduler Index overflow")

    return total_time_frame, bounded_acc, core_job, scheduler_list


def core_job(jobs):
    acc_num = len(buffer_size)
    job_num = jobs.shape[0]
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0
    finished_job = 0
    assigned_job = 0
    core_job = [[] for i in range(acc_num)]


    DM_list_idx = np.arange(0, acc_num) * 2
    EX_list_idx = DM_list_idx + 1

    all_latency = np.zeros([job_num, acc_num])
    latency_DM = np.zeros([job_num, acc_num])
    latency_EX = np.zeros([job_num, acc_num])

    while(all(jobs == float('Inf'))):

        greedy_key = jobs[:, DM_list_idx] + jobs[:, EX_list_idx] + all_latency

        job_idx, acc_idx = np.unravel_index(np.argmin(greedy_key, axis=None), greedy_key.shape)
        core_job[acc_idx].append(job_idx)
        job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
        latency_DM += job[0]
        latency_EX[:, acc_num] += job[1]
        all_latency = latency_DM + latency_EX
        jobs[job_idx, :] = float('Inf')    

    return core_job

def scheduler_list_run(jobs, buffer_use, buffer_size, scheduler_list):
    acc_num = len(buffer_size)
    job_num = len(scheduler_list)
    remain_buffer = copy.deepcopy(buffer_size)
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0
    finished_job = 0
    assigned_job = 0

    covered_latency = 0
    sl_idx = 0
    job_idx = scheduler_list[sl_idx][0]
    acc_idx = scheduler_list[sl_idx][1]
    sl_idx += 1

    job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
    time_frame[0] = job[0]
    time_frame[acc_idx + 1] = job[0] + job[1]
    remain_buffer[acc_idx] = remain_buffer[acc_idx] - buffer_use[job_idx, acc_idx]
    assigned_job +=1

    while(1):
        
        if assigned_job < job_num:

            min_frame = time_frame.min()
            min_idx = np.argmin(time_frame)
        else:
            time_frame[time_frame == float('Inf')] = 0
            total_time_frame = time_frame.max()
            bounded_acc = np.argmax(time_frame)
            
            break
       


        if min_idx == 0:
            job_idx = scheduler_list[sl_idx][0]
            acc_idx = scheduler_list[sl_idx][1]
            sl_idx += 1

            job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
            
            if buffer_use[job_idx, acc_idx] <= remain_buffer[acc_idx]:
                time_frame[0] += job[0]
                if time_frame[acc_idx + 1] == float('Inf'):
                    time_frame[acc_idx + 1] = time_frame[0] + job[1]

                    acc_temp = np.argmax(time_frame)
                    for i in range(sl_idx):
                        if scheduler_list[i][1] != acc_temp - 1:
                            covered_latency += jobs[scheduler_list[i][0], scheduler_list[i][1] * 2 + 1]
                        if scheduler_list[i][1] == acc_temp - 1:
                            for j in range(i+1, sl_idx):
                                covered_latency += jobs[scheduler_list[j][0], scheduler_list[j][1] * 2]
                else:
                    acc_start_latency = max(time_frame[0], time_frame[acc_idx + 1]) - time_frame[acc_idx + 1]
                    time_frame[acc_idx + 1] += acc_start_latency + job[1]

                    acc_temp = np.argmax(time_frame)
                    for i in range(sl_idx):
                        if scheduler_list[i][1] != acc_temp - 1:
                            covered_latency += jobs[scheduler_list[i][0], scheduler_list[i][1] * 2 + 1]
                        if scheduler_list[i][1] == acc_temp - 1:
                            for j in range(i+1, sl_idx):
                                covered_latency += jobs[scheduler_list[j][0], scheduler_list[j][1] * 2]
                    if acc_temp == acc_idx - 1:
                        covered_latency -= acc_start_latency
            else:

                time_frame[0] = time_frame[acc_idx + 1] + job[0]
                time_frame[acc_idx + 1] = time_frame[0] + job[1]
                remain_buffer[acc_idx] = buffer_size[acc_idx] - buffer_use[job_idx, acc_idx]
            
            assigned_job += 1
        
        if min_idx > 0 and min_idx <= acc_num:

            finished_job += 1
            
            time_frame[min_idx] = float('Inf')
            remain_buffer[min_idx - 1] = buffer_size[min_idx - 1]
        
        if min_idx < 0 or min_idx > acc_num:
            raise IndexError("scheduler Index overflow")
    

    return total_time_frame, bounded_acc, covered_latency


def reorder(jobs, core_job, buffer_use, buffer_size):
    acc_num = len(buffer_size)
    job_num = jobs.shape[0]
    remain_buffer = copy.deepcopy(buffer_size)
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0
    finished_job = 0
    assigned_job = 0
    scheduler_list = []

    DM_list_idx = np.arange(0, acc_num) * 2
    EX_list_idx = DM_list_idx + 1

    covered_latency = np.full([job_num, acc_num], -1)
    for i, cj in enumerate(core_job):
        if cj != []:
            covered_latency[cj, i] = 0

    
    job_idx, acc_idx = np.unravel_index(np.argmax(covered_latency, axis=None), covered_latency.shape)
    for i in range(len(core_job[acc_idx])):
        if core_job[acc_idx][i] == job_idx:
            core_job[acc_idx].pop(i)
            break
    scheduler_list.append([job_idx, acc_idx])

    while(any(core_job)):
        left_scheduler_list = copy.deepcopy(scheduler_list)
        right_scheduler_list = copy.deepcopy(scheduler_list)
        left_scheduler_list.insert(0, [])
        right_scheduler_list.append([])
        left_covered_latency = copy.deepcopy(covered_latency)
        right_covered_latency = copy.deepcopy(covered_latency)
        for i, cj in enumerate(core_job):
            for job in cj:
                new_pair = [job, i]
                left_scheduler_list[0] = new_pair
                left_total_time_frame, left_bounded_acc, left_cover = scheduler_list_run(jobs, buffer_use, buffer_size, left_scheduler_list)
                right_scheduler_list[-1] = new_pair
                right_total_time_frame, right_bounded_acc, right_cover = scheduler_list_run(jobs, buffer_use, buffer_size, right_scheduler_list)

                left_covered_latency[job, i] = left_cover
                right_covered_latency[job, i] = right_cover

        if left_covered_latency.max() > right_covered_latency.max():
            job_idx, acc_idx = np.unravel_index(np.argmax(left_covered_latency, axis=None), left_covered_latency.shape)
            for i in range(len(core_job[acc_idx])):
                if core_job[acc_idx][i] == job_idx:
                    core_job[acc_idx].pop(i)
                    break
            scheduler_list = left_scheduler_list
        else:
            job_idx, acc_idx = np.unravel_index(np.argmax(right_covered_latency, axis=None), right_covered_latency.shape)
            for i in range(len(core_job[acc_idx])):
                if core_job[acc_idx][i] == job_idx:
                    core_job[acc_idx].pop(i)
                    break
            scheduler_list = right_scheduler_list
    
    total_time_frame, bounded_acc, _ = scheduler_list_run(jobs, buffer_use, buffer_size, scheduler_list)
    return total_time_frame, bounded_acc


def oracle(jobs, buffer_use, buffer_size):
    res_all = []
    job_num = jobs.shape[0]
    acc_num = buffer_size.shape[0]
    job_idx = np.arange(job_num)
    acc_idx = np.arange(acc_num)
    job_indices = list(permutations(job_idx, job_num))

    acc_indices = list(product(acc_idx, repeat = job_num))

    for ji in job_indices:
        for ai in acc_indices:
            schedule = np.array([ji, ai])
            schedule = schedule.swapaxes(0,1).tolist()
            total_time_frame, bounded_acc, covered_latency = scheduler_list_run(jobs, buffer_use, buffer_size, schedule)
            res_all.append(total_time_frame)

    res_all_np = np.array(res_all)
    
    return res_all_np.min()

    

def AI_MT(net_work_list, batch_set, layer_num, buffer_size, bandwidth, threshold, slot_num):
    job_num = slot_num
    total_job_num = (np.array(batch_set) * np.array(layer_num)).sum()
    net_work_list_this = copy.deepcopy(net_work_list)
    jobs, buffer_use, net_work_list_this = pop_slot_num(net_work_list_this, bandwidth, slot_num)

    acc_num = buffer_size.shape[0]
    ACL_EX = np.zeros(acc_num)
    DM_list_idx = np.arange(0, acc_num) * 2
    EX_list_idx = DM_list_idx + 1
    remain_buffer = copy.deepcopy(buffer_size)
    time_frame = np.full(acc_num + 1, float('Inf'))
    total_time_frame = 0

    finished_job = 0
    assigned_job = 0
    DMs = jobs[:, DM_list_idx]
    EXs = jobs[:, EX_list_idx]
    min_EX_idx = np.argmin(EXs, axis = 1)
        

    scheduler_list = []
    core_job = [[] for i in range(acc_num)]
    target = None
    for job_idx in range(job_num):
        acc_idx = min_EX_idx[job_idx]
        if DMs[job_idx, acc_idx] == float('Inf'):
            continue
        if buffer_use[job_idx, acc_idx] < remain_buffer[acc_idx]:
            if ACL_EX[acc_idx] < threshold[acc_idx]:
                if DMs[job_idx, acc_idx] < EXs[job_idx, acc_idx]:
                    target = [job_idx, acc_idx]
                    break
                target = [job_idx, acc_idx]
                break
            else:
                target = [job_idx, acc_idx]
                break

    scheduler_list.append(target)
    ACL_EX[acc_idx] = max(ACL_EX[acc_idx] - DMs[job_idx, acc_idx], 0) + EXs[job_idx, acc_idx]
    job = jobs[job_idx, acc_idx*2:acc_idx*2+2]
    time_frame[0] = job[0]
    time_frame[acc_idx + 1] = job[0] + job[1]
    remain_buffer[acc_idx] = remain_buffer[acc_idx] - buffer_use[job_idx, acc_idx]
    assigned_job +=1
    core_job[acc_idx].append(job_idx)

    single_job, single_buffer_use, net_work_list_this = pop_max(net_work_list_this, bandwidth, acc_num)
    jobs[job_idx,:] = single_job
    buffer_use[job_idx,:] = single_buffer_use  

    
    while(1):
        
        if assigned_job < total_job_num:

            min_frame = time_frame.min()
            min_idx = np.argmin(time_frame)
        else:
            time_frame[time_frame == float('Inf')] = 0
            total_time_frame = time_frame.max()
            bounded_acc = np.argmax(time_frame)
            break

        if min_idx == 0:
            DMs = jobs[:, DM_list_idx]
            EXs = jobs[:, EX_list_idx]

            min_EX_idx = np.argmin(EXs, axis = 1)

            target = None
            for job_idx in range(job_num):
                acc_idx = min_EX_idx[job_idx]
                if DMs[job_idx, acc_idx] == float('Inf'):
                    continue
                if ACL_EX[acc_idx] < threshold[acc_idx]:
                    if DMs[job_idx, acc_idx] < EXs[job_idx, acc_idx]:
                        target = [job_idx, acc_idx]
                        break
                else:
                    target = [job_idx, acc_idx]
                    break
                target = [job_idx, acc_idx]
                break
                    

            scheduler_list.append(target)
            core_job[acc_idx].append(job_idx)

            job = jobs[job_idx, acc_idx*2:acc_idx*2+2]

            if buffer_use[job_idx, acc_idx] <= remain_buffer[acc_idx]:
                time_frame[0] += job[0]
                ACL_EX[acc_idx] = max(ACL_EX[acc_idx] - DMs[job_idx, acc_idx], 0) + EXs[job_idx, acc_idx]
                if time_frame[acc_idx + 1] == float('Inf'):
                    time_frame[acc_idx + 1] = time_frame[0] + job[1]
                else:
                    acc_start_latency = max(time_frame[0], time_frame[acc_idx + 1]) - time_frame[acc_idx + 1]
                    time_frame[acc_idx + 1] += acc_start_latency + job[1]
                remain_buffer[acc_idx] -= buffer_use[job_idx, acc_idx]
            else:
                ACL_EX[acc_idx] = EXs[job_idx, acc_idx]
                time_frame[0] = time_frame[acc_idx + 1] + job[0]
                time_frame[acc_idx + 1] = time_frame[0] + job[1]
                
                remain_buffer[acc_idx] = buffer_size[acc_idx] - buffer_use[job_idx, acc_idx]

            assigned_job += 1

            single_job, single_buffer_use, net_work_list_this = pop_max(net_work_list_this, bandwidth, acc_num)
            jobs[job_idx,:] = single_job
            buffer_use[job_idx,:] = single_buffer_use  
        
        if min_idx > 0 and min_idx <= acc_num:

            finished_job += 1
            time_frame[min_idx] = float('Inf')
            remain_buffer[min_idx - 1] = buffer_size[min_idx - 1]
        
        if min_idx < 0 or min_idx > acc_num:
            raise IndexError("scheduler Index overflow")
    return total_time_frame, bounded_acc, core_job, scheduler_list


def space(batch_set, layer_num, magma_iter, acc_num):
    magma_space = 0
    RealArch_space = 0

    layer_num_np = np.array(layer_num)
    layer_deep = layer_num_np.max()
    for i in range(layer_deep):
        job_per_layer = (layer_num_np > 0).sum()
        magma_space += job_per_layer * magma_iter
        layer_num_np -= 1

    layer_num_np = np.array(layer_num)
    job_num = (np.array(batch_set) * layer_num_np).sum()
    for i in range(job_num):
        job_per_layer = (layer_num_np > 0).sum()
        RealArch_space += acc_num  * job_per_layer
        idx = i % len(layer_num)
        layer_num_np[idx] -= 1
    
    return magma_space, RealArch_space

def test():
    # set output dir and file
    exp_dir = 'exp_res/'
    file_name = 'cycle_output.csv'
    exp_file = exp_dir + file_name
    if os.path.exists(exp_file):
        os.remove(exp_file)
    with open(exp_file, 'a+', newline = '') as f:
        writer = csv.writer(f)
        head = ['bandwidth','buffer_size', 'slot_num', 'network_selet', 'baseline_cycle', 'RealArch_cycle', 'AI_MT_cycle', 'RealArch_speedup', 'AI-MT_speedup']
        writer.writerow(head)
    f.close()

    # batch size of input
    batch_num = 3
    # accelerator combination
    acc_select = [np.array([0,1,2])]
    slot_num_list = [3]

    # net_work_total_name = np.array(['alexnet', 'resnet18', 'vgg11',  'resnet34', 'vgg13', 'resnet50', 'vgg16', 'resnet101', 'vgg19'])
    # net_select = []
    # ns_idx = np.arange(net_work_total_name.shape[0])
    # for i in range(net_work_total_name.shape[0]):
    #     comp = np.array(list(combinations(ns_idx, i + 1))[0:3])
    #     for j in range(comp.shape[0]):
    #         idx = comp[j,:]
    #         temp = net_work_total_name[idx] 
    #         net_select.append(temp)

    # network combination
    net_select = [np.array(['alexnet', 'resnet18', 'resnet34', 'resnet50'])]
    # buffer size configuration (DM)
    buffer_size_list = [25]
    # DRAM bandwidth configuration (B/cycle)
    bandwidth_list = [256]

    for bandwidth in bandwidth_list:
        for bs in buffer_size_list:
            for slot_num in slot_num_list:
                for ns in net_select:
                    for ac in acc_select:
                        acc_idx = sorted(np.concatenate([ac * 3, ac * 3 + 1, ac * 3 + 2]))
                        tpu_buffer_size = bs
                        shidiannao_buffer_size = bs
                        eyeriss_buffer_size = bs
                        buffer_size = np.array([tpu_buffer_size, shidiannao_buffer_size, eyeriss_buffer_size]) * 1024 * 1024

                        dir_name = 'estimation_result_csv/'
                        file_list = []
                        for i in range(ns.shape[0]):
                            fname = dir_name + ns[i] + '_estimation.csv'
                            file_list.append(fname)
                        
                        net_work_list = []
                        jobs_list = [[] for i in range(slot_num)]
                        layer_num = []
                        
                        for fname in file_list:
                            esti_res = read_csv_numpy(fname)
                            net_work_list.append(esti_res[:, acc_idx])

                        batch_per_network = np.full(len(net_work_list), batch_num)
                        
                        net_work_total = batch_per_network.sum()
                        for i in range(net_work_total):
                            jl_idx = i % slot_num
                            nw_idx = i % len(net_work_list)
                            jobs_list[jl_idx].append(net_work_list[nw_idx])

                        
                        for i in range(slot_num):
                            if jobs_list[i] != []:
                                jobs_list[i] = np.concatenate(jobs_list[i], 0)
                                layer_num.append(jobs_list[i].shape[0])
                            else:
                                for j in range(i, slot_num):
                                    jobs_list.pop(-1)
                                    layer_num.append(0)
                                break
                        
                        net_work_list = net_work_list * batch_per_network[0]

                        batch_set = np.full(slot_num, 1)
                        # run Round-Robin scheduler
                        total_time_frame_baseline, bounded_acc_baseline, core_job_baseline = baseline_schedule(net_work_list, batch_set, layer_num, buffer_size[ac], bandwidth, slot_num)

                        # run RealArch scheduler
                        batch_set = np.full(slot_num, 1)
                        total_time_frame_greedy, bounded_acc_greedy, core_job_greedy, scheduler_list_greedy = greedy_scheduler(net_work_list, batch_set, layer_num, buffer_size[ac], bandwidth, slot_num)
                        
                        # run AI-MT scheduler 
                        threshold = np.array([1902704,  1839104, 26298208]) * batch_per_network[0]
                        batch_set = np.full(slot_num, 1)
                        total_time_frame_AIMT, bounded_acc_AIMT, core_job_AIMT, scheduler_list_AIMT= AI_MT(net_work_list, batch_set, layer_num, buffer_size[ac], bandwidth, threshold, slot_num)

                        
                        with open(exp_file, 'a+', newline = '') as f:
                            writer = csv.writer(f)
                            head = [bandwidth, bs, slot_num, ns, int(total_time_frame_baseline), int(total_time_frame_greedy), int(total_time_frame_AIMT), \
                             total_time_frame_baseline/total_time_frame_greedy, total_time_frame_baseline/total_time_frame_AIMT]
                            writer.writerow(head)
                        f.close()

test()