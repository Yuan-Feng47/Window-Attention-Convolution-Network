import numpy as np
import torch
# # 检查可用的 GPU 数量
# num_gpus = torch.cuda.device_count()
# print("Number of available GPUs: ", num_gpus)
# # 使用所有可用的 GPU
# devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
# print("Using devices:", devices)
number=[0,1,7,8,13,14,15,16,17,18,19,24]
labels = {
    0 : ("BPSK",0),
    1: ("QPSK", 1),
    2: ("8PSK", 2),
    3: ("16PSK",3),
    4: ("32PSK",4),
    5: ("64PSK",5),
    6 : ("4QAM", 6),
    7: ("8QAM", 7),
    8: ("16QAM",8),
    9: ("32QAM",9),
    10: ("64QAM",10),
    11: ("128QAM",11),
    12: ("256QAM",12),
    13: ("2FSK",13),
    14: ("4FSK",14),
    15: ("8FSK",15),
    16: ("16FSK",16),
    17: ("4PAM",17),
    18: ("8PAM",18),
    19: ("16PAM",19),
    20: ("AM-DSB",20),
    21: ("AM-DSB-SC",21),
    22: ("AM-USB",22),
    23: ("AM-LSB",23),
    24: ("FM",24),
    25: ("PM",25)
}
# labels = {
#     0: ('OOK', 0),
#     1: ('4ASK', 1),
#     2: ('8ASK', 2),
#     3: ('BPSK', 3),
#     4: ('QPSK', 4),
#     5: ('8PSK', 5),
#     6: ('16PSK',6),
#     7: ('32PSK', 7),
#     8: ('16APSK', 8),
#     9: ('32APSK', 9),
#     10: ('64APSK', 10),
#     11: ('128APSK', 11),
#     12: ('16QAM', 12),
#     13: ('32QAM', 13),
#     14: ('64QAM', 14),
#     15: ('128QAM', 15),
#     16: ('256QAM', 16),
#     17: ('AM-SSB-WC', 17),
#     18: ('AM-SSB-SC', 18),
#     19: ('AM-DSB-WC', 19),
#     20: ('AM-DSB-SC', 20),
#     21: ('FM', 21),
#     22: ('GMSK', 22),
#     23: ('OQPSK',23)
# }


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


# HisarMod_list = [0,10,20,30,40,50,1,11,21,31,41,51,61,2,12,22,32,3,13,23,4,14,24,34,44,54]
# classes_dic = convert_labels_to_dict(HisarMod_list)
# print(classes_dic)
#
# input_tensor = torch.tensor([[0.],[2.]], device='cuda:0')
# print(tensor_to_labels(input_tensor))