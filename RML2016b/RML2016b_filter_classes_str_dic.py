import numpy as np
import torch
import torch.nn.functional as F
# # 检查可用的 GPU 数量
# num_gpus = torch.cuda.device_count()
# print("Number of available GPUs: ", num_gpus)
# # 使用所有可用的 GPU
# devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
# print("Using devices:", devices)
label_mapping = {'8PSK': 0, 'BPSK': 1, 'CPFSK': 2, 'GFSK': 3, 'PAM4': 4, 'QAM16': 5, 'QAM64': 6, 'QPSK': 7, 'AM-DSB': 8,  'WBFM': 9}
labels = {
    0 : ("8PSK",0),
    1: ("BPSK", 1),
    2: ("CPFSK", 2),
    3: ("GFSK",3),
    4: ("PAM4",4),
    5: ("QAM16",5),
    6 : ("QAM64", 6),
    7: ("QPSK", 7),
    8: ("AM-DSB",8),
    
    9: ("WBFM",9),
}


def get_modulation_types(keys):
    return [labels[key][0] for key in keys]

def get_positions(keys):
    return [labels[key][1] for key in keys]

def convert_labels_to_dict(label_array, label_dict=labels):
    dict_labels = {i: label_dict[label][0] for i, label in enumerate(label_array)}
    return dict_labels


def tensor_to_labels(tensor, label_dict=labels):
    tensor = tensor.flatten().long().tolist()  # 将输入Tensor转换为list
    labels = [label_dict[i][1] for i in tensor]  # 获取对应的位置值
    return torch.tensor(labels)  # 将位置值转换为Tensor并返回
