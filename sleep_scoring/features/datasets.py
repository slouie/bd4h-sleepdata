import bisect
import numpy as np
import torch
from operator import itemgetter
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.sampler import Sampler


class RecordSampler(Sampler):
    '''
    Randomly sample in one record before moving onto the next to avoid excessive loading.
    '''

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        idxs = np.concatenate([torch.randperm(y - x) + x for x, y in self.dataset.epoch_ranges])
        return iter(idxs)

    def __len__(self):
        return len(self.dataset)


class EpochRangeKeyList(object):
    '''
    Wrapper to bisect epoch ranges in feature files to find which file contains particular epoch.
    '''

    def __init__(self, l, key):
        self.l = l
        self.key = key

    def __getitem__(self, index):
        return self.key(self.l[index])

    def __len__(self):
        return len(self.l)


class EpochDataset(Dataset):
    '''
    Epoch is (6 channel x 3000 sample) tensor. Randomly sampled from each feature file.
    '''

    def __init__(self, feature_paths):
        self.files = feature_paths
        self.epoch_ranges = []
        onset = 0
        for feature_file in self.files:
            data = np.load(feature_file)['arr_0']
            num_epochs = int(data[0][-1] + 1)
            self.epoch_ranges.append((onset, onset + num_epochs))
            onset += num_epochs
        self.key_list = EpochRangeKeyList(self.epoch_ranges, key=lambda x: x[0])
        self.cache = {}

    def __getitem__(self, epoch_idx):
        # find feature file for this particular epoch
        file_idx = bisect.bisect_right(self.key_list, epoch_idx) - 1
        if file_idx not in self.cache:
            # cache one file at a time
            self.cache = {file_idx : np.load(self.files[file_idx])['arr_0']}
        data = self.cache[file_idx]
        # subtract onset to get epoch idx within this file
        rel_idx = epoch_idx - self.epoch_ranges[file_idx][0]
        data = data[1:, data[0, :] == rel_idx]
        target = int(1)
        return data.astype(np.float32), target

    def __len__(self):
        return self.epoch_ranges[-1][1]