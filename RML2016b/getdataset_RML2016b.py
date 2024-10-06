import h5py
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm

class RML2016bDataset(Dataset):
    def __init__(self, data, labels, snr=None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        if snr is not None:
            self.snr = torch.from_numpy(snr).float()
        else:
            self.snr = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.snr is not None:
            return self.data[idx], self.labels[idx], self.snr[idx]
        else:
            return self.data[idx], self.labels[idx]

# 自定义数据集类
class ModulationRecognitionDataset_RML2016b(Dataset):
    def __init__(self, h5_file, select_every=None, snr_list=None, modulation_list=None):
        self.h5_file = h5_file
        self.modulation_list=modulation_list
        with h5py.File(h5_file, 'r') as f:
            num_samples = len(f['X'])
            all_indices = list(range(0, num_samples, select_every))
            if snr_list is not None or modulation_list is not None:
                filtered_indices = []
                for idx in tqdm(all_indices, desc="Filtering SNR and Modulation"):
                    if snr_list is None or f['Z'][idx] in snr_list:
                        if modulation_list is None or f['Y'][idx] in modulation_list:
                            filtered_indices.append(idx)
                            #print(modulation_list.index(f['Y'][idx]))
                            #print(f['Z'][idx])
                all_indices = filtered_indices

            random.shuffle(all_indices)  # 打乱索引以增加随机性
            self.indices = all_indices
            self.length = len(self.indices)
            print("Length of dataset: ", self.length)

    def __getitem__(self, index):
        selected_index = self.indices[index]
        with h5py.File(self.h5_file, 'r') as f:
            X = f['X'][selected_index]
            Y = f['Y'][selected_index]
            Z = f['Z'][selected_index]
           
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor(Z, dtype=torch.float32))

    def __len__(self):
        return self.length
