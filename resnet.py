import torchvision.models as models
import torch
import copy
from estimation import estimation_model
from datetime import datetime
import csv
# from torchsummary import summary
import pdb


def Resnet(pe_dim, pe_num, model_name = 'vgg16', plot = True):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    feature = torch.rand(8, 3, 224, 224, device = device)
    instr = 'models.' + model_name + '()'
    model = eval(instr).to(device)

    model.eval()

    if plot:
        result_dir = 'estimation_result_csv/'
    
        file_name = model_name + '_estimation.csv'
        csv_file = result_dir + file_name

        with open(csv_file, 'a+', newline = '') as f:
            writer = csv.writer(f)
            head = ['tpu_MB', 'tpu_CB', 'tpu_Iter', 'shidiannao_MB', 'shidiannao_CB', 'shidiannao_Iter', \
                'eyeriss_MB', 'eyeriss_CB', 'eyeriss_Iter']
            writer.writerow(head)
        f.close()

    for m in model:
        input = feature.detach()
        feature = m(feature)
        output = feature.detach()
        esm = estimation_model(input, output, m)
        if isinstance(m, torch.nn.Conv2d):
            tpu_res, shidiannao_res, eyeriss_res = esm.run_all(pe_dim, pe_num, plot = plot, model_name = model_name, csv_file = csv_file)
       

    m = model.avgpool
    # print(m)
    feature = m(feature)
    for m in model.classifier:
        feature = torch.flatten(feature, 1)
        input = feature.detach()
        feature = m(feature)
        output = feature.detach()
        esm = estimation_model(input, output, m)
        if isinstance(m, torch.nn.Linear):
            tpu_res, shidiannao_res, eyeriss_res = esm.run_all(pe_dim, pe_num, plot = plot, model_name = model_name, csv_file = csv_file)

if __name__ == "__main__":
    pe_dim = 16
    pe_num = 2
    resnet_list = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    for model_name in resnet_list:
        Resnet(pe_dim, pe_num, model_name, plot = True)
    

