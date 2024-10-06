import h5py
import torch
import random
from torch.utils.data import Dataset
from tqdm import tqdm


class ModulationRecognitionDataset_RML201801A(Dataset):
    @staticmethod
    def create_datasets(h5_file, train_ratio, select_every=None, snr_list=None, modulation_list=None):
        with h5py.File(h5_file, 'r') as f:
            num_samples = len(f['X'])
            all_indices = list(range(0, num_samples, select_every))
            groups = {}
            progress_bar = tqdm(all_indices, desc="Grouping by SNR and Modulation")
            for idx in progress_bar:
                if snr_list is None or f['Z'][idx][0] in snr_list:
                    current_modulation = torch.tensor(f['Y'][idx])
                    if modulation_list is None or any(
                            torch.equal(current_modulation, mod_type) for mod_type in modulation_list):
                        group_key = (f['Z'][idx][0], tuple(current_modulation.tolist()))  # Create a key by SNR and Modulation
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].append(idx)
                        progress_bar.set_description("Grouping SNR=%s, Modulation=%s" % group_key)

            train_indices = []
            valid_indices = []
            progress_bar = tqdm(groups.items(), desc="Splitting into train and valid sets")
            for group_key, indices in progress_bar:
                random.shuffle(indices)  # 打乱索引以增加随机性
                split_point = int(len(indices) * train_ratio)
                train_indices.extend(indices[:split_point])
                valid_indices.extend(indices[split_point:])
                progress_bar.set_description("Splitting SNR=%s, Modulation=%s" % group_key)

            train_dataset = ModulationRecognitionDataset_RML201801A(h5_file, train_indices)
            valid_dataset = ModulationRecognitionDataset_RML201801A(h5_file, valid_indices)

            return train_dataset, valid_dataset

    def __init__(self, h5_file, indices):
        self.h5_file = h5_file
        self.indices = indices
        self.length = len(self.indices)

    def __getitem__(self, index):
        selected_index = self.indices[index]
        with h5py.File(self.h5_file, 'r') as f:
            X = f['X'][selected_index].transpose(1, 0)  # 转置，使得形状变为 (2, 1024)
            Y = f['Y'][selected_index]
            Z = f['Z'][selected_index]

        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(Y, dtype=torch.float32),
                torch.tensor(Z, dtype=torch.float32))

    def __len__(self):
        return self.length

    def get_modulation_snr_counts(self):
        counts = {}
        with h5py.File(self.h5_file, 'r') as f:
            for idx in self.indices:
                snr = f['Z'][idx][0]
                modulation = tuple(f['Y'][idx].tolist())
                key = (snr, modulation)
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
        return counts