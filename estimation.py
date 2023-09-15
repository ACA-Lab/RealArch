import torch
from math import ceil as ceil
from datetime import datetime
import csv

class estimation_model:
    def __init__(self, input, output, layer, bandwidth = 256):
        self.input = input.shape
        self.output = output.shape
        self.layer = layer
        # B/cycle
        self.bandwidth = bandwidth
        self.byte = 2

    def tpu(self, pe_dim, pe_num):
        

        if isinstance(self.layer, torch.nn.Conv2d):
            b = self.input[0]
            ic = self.input[1]
            iw = self.input[2]
            ih = self.input[3]
            k = self.layer.kernel_size[0]
            oc = self.output[1]
            ow = self.output[2]
            oh = self.output[3]

            DM = (ow * oh * pe_dim * b + pe_dim * pe_dim + ow * oh * pe_dim * b) * self.byte / self.bandwidth
            EX = ceil(ow * oh / pe_num) * b + pe_dim - 1
            Iter = ceil((ic * k * k) / pe_dim) * ceil(oc / (pe_dim))
        
        if isinstance(self.layer, torch.nn.Linear):
            b = self.input[0]
            ic = self.input[1]
            oc = self.output[1]

            DM = (pe_dim * b + pe_dim * pe_dim * pe_num + pe_dim * b) * self.byte / self.bandwidth
            EX = b + pe_dim - 1
            Iter = ceil(oc / (pe_dim * pe_num)) * ceil(ic / pe_dim)
        
        return [DM, EX, Iter]
    
    def shidiannao(self, pe_dim, pe_num):
        

        if isinstance(self.layer, torch.nn.Conv2d):
            b = self.input[0]
            ic = self.input[1]
            iw = self.input[2]
            ih = self.input[3]
            k = self.layer.kernel_size[0]
            oc = self.output[1]
            ow = self.output[2]
            oh = self.output[3]

            DM = (pe_dim * pe_dim * pe_num * ic * b + k * k * ic + pe_dim * pe_dim * pe_num * b) * self.byte / self.bandwidth
            EX = k * k * ic * b
            Iter = ceil(ceil(ow * oh * oc/ (pe_dim * pe_dim)) / pe_num)

        
        if isinstance(self.layer, torch.nn.Linear):
            b = self.input[0]
            ic = self.input[1]
            oc = self.output[1]

            DM = (b * ic + pe_dim * pe_dim * pe_num * ic + pe_dim * pe_dim * pe_num * b) * self.byte / self.bandwidth
            EX = ic * b
            Iter = ceil(oc / (pe_dim * pe_dim * pe_num))

        return [DM, EX, Iter]
    
    def eyeriss(self, pe_dim, pe_num):
        

        if isinstance(self.layer, torch.nn.Conv2d):
            b = self.input[0]
            ic = self.input[1]
            iw = self.input[2]
            ih = self.input[3]
            k = self.layer.kernel_size[0]
            oc = self.output[1]
            ow = self.output[2]
            oh = self.output[3]

            DM = ((pe_dim * k + ceil(pe_dim / k) * (pe_dim - 1))* b * pe_num + k * pe_dim + pe_dim * pe_num * b) * self.byte / self.bandwidth
            EX = (k + pe_dim) * b
            Iter = ceil(oh / pe_dim) * ceil(ow / pe_num) * oc * ceil(ic * k / pe_dim)

        
        if isinstance(self.layer, torch.nn.Linear):
            b = self.input[0]
            ic = self.input[1]
            oc = self.output[1]

            DM = (pe_dim * b + pe_dim * pe_dim * pe_num + pe_dim * b) * self.byte / self.bandwidth
            EX = b + pe_dim - 1
            Iter = ceil(oc / (pe_dim * pe_num) * ceil(ic / pe_dim))
            
        return [DM, EX, Iter]
    
    def run_all(self, pe_dim, pe_num, plot, csv_file):
        tpu_res = self.tpu(pe_dim, pe_num)
        shidiannao_res = self.shidiannao(pe_dim, pe_num)
        eyeriss_res = self.eyeriss(pe_dim, pe_num)
        if plot:
            with open(csv_file, 'a+', newline = '') as f:
                writer = csv.writer(f)
                head = [tpu_res[0], tpu_res[1], tpu_res[2], shidiannao_res[0], shidiannao_res[1], shidiannao_res[2], \
                eyeriss_res[0], eyeriss_res[1], eyeriss_res[2]]
                writer.writerow(head)
            f.close()
        return tpu_res, shidiannao_res, eyeriss_res



