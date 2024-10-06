'''
本部分用于新人熟悉RML2018.01a数据库的数据类型
'''
import h5py
import numpy as np
import torch
from modulation_dict import get_modulation_array_list, \
    filter_classes_str_dic
from tqdm import tqdm

# 读取数据集文件
select_every = 128  # 建议为1、128、256
filename = '/home/hust/Desktop/modulation/RML2018/GOLD_XYZ_OSC.0001_1024.hdf5'
snr_list = list(range(-20, 30 + 1, 2))
RML201801a_modulation_Difficult_list = list(range(0, 24))
modulation_list = get_modulation_array_list(RML201801a_modulation_Difficult_list)

with h5py.File(filename, 'r') as f:
    X = f['X']  # IQ路信号，(2555904, 1024, 2)=2555904个，1024位，2路
    lbl = f['Y']  # 调制类型，(2555904, 24)=2555904个,24位（详情见modulation_dict.py)
    snrs = f['Z']  # 信噪比，(2555904, 1) =2555904个,1位
    print("X.shape={}, dtype={}".format(X.shape, X.dtype))
    print("lbl.shape={}, dtype={}".format(lbl.shape, lbl.dtype))
    print("snrs.shape={}, dtype={}".format(snrs.shape, snrs.dtype))
    num_samples = len(f['X'])
    all_indices = list(range(0, num_samples, select_every))
    if snr_list is not None or modulation_list is not None:
        filtered_indices = []
        for idx in tqdm(all_indices, desc="Filtering SNR and Modulation"):
            if snr_list is None or f['Z'][idx][0] in snr_list:
                current_modulation = torch.tensor(f['Y'][idx])
                if modulation_list is None or any(
                        torch.equal(current_modulation, mod_type) for mod_type in modulation_list):
                    filtered_indices.append(idx)
        all_indices = filtered_indices
    print("Filtered indices: {}".format(all_indices))
    signal_idx = int(input("请输入数字1~{}：".format(X.shape[0])))
    signal = X[signal_idx]
    modulation = lbl[signal_idx]
    snr = snrs[signal_idx, 0]
    print("第{}个数据{}的signal.shape={}".format(signal_idx, signal, signal.shape))
    print("第{}个数据的modulation={}".format(signal_idx, modulation))
    print("第{}个数据的snr={}".format(signal_idx, snr))

    # print('解析SNR数据')
    # unique_elements_SNRs, counts_SNR = np.unique(snrs, return_counts=True)
    # for element_SNRs, count_SNRs in zip(unique_elements_SNRs, counts_SNR):
    #     print(f"{element_SNRs}: {count_SNRs}")

