import bisect
import numpy as np
import pyedflib
import torch
from model.models import Epoch
from operator import itemgetter
from torch.utils.data import TensorDataset, Dataset
import sys


EPOCH_LENGTH = 30


class FeatureConstruction(object):

    @staticmethod
    def construct(sc, rdd, model_type):
        # TODO: use spark later
        pass


class EpochRangeKeyList(object):

    def __init__(self, l, key):
        self.l = l
        self.key = key

    def __len__(self):
        return len(self.l)

    def __getitem__(self, index):
        return self.key(self.l[index])


class EDFEpochDataset(Dataset):

    def __init__(self, edf_paths):
        self.files = edf_paths
        self.files.sort(key=itemgetter(0))

        self.epoch_ranges = []
        onset = 0
        for psg, _ in self.files:
            reader = pyedflib.EdfReader(psg)
            # TODO: is this assumption correct?
            num_epochs = reader.file_duration // EPOCH_LENGTH
            self.epoch_ranges.append((onset, onset + num_epochs))
            onset += num_epochs
        self.key_list = EpochRangeKeyList(self.epoch_ranges, key=lambda x: x[0])

        # temporarily cache files to test for now since we don't want to load same file
        # every time for each epoch in the file. but if dataloader shuffle flag set to
        # True this is pointless.
        self.cache = {}

    def __getitem__(self, epoch_idx):
        # find file for this particular epoch
        file_idx = bisect.bisect_right(self.key_list, epoch_idx) - 1
        if file_idx not in self.cache:
            # cache one file at a time
            self.cache = {file_idx : load_edf(self.files[file_idx])}
        # subtract onset to get epoch idx within this file
        epoch = self.cache[file_idx][epoch_idx - self.epoch_ranges[file_idx][0]]

        data = np.zeros((6, 3000))
        # TODO: use expert annotations
        target = int(1)
        for channel, signal in epoch.channels.items():
            if epoch.sample_frequencies[channel] == 100:
                data[channel] = signal
            elif epoch.sample_frequencies[channel] == 1:
                data[channel] = np.repeat(signal, 100, axis=0)
            else:
                raise Exception('Unexpected sample frequency')
        return data.astype(np.float32), target

    def __len__(self):
        return self.epoch_ranges[-1][1]


def chunk(signal, sample_frequency):
    num_samples = EPOCH_LENGTH * sample_frequency
    for i in range(0, len(signal), num_samples):
        yield signal[i:i+num_samples]


def load_edf(edf_path):
    '''
    return: list of Epoch(s) in edf
    '''
    epochs = []
    psg_path, hypno_path = edf_path
    psg_reader = pyedflib.EdfReader(psg_path)
    hyp_reader = pyedflib.EdfReader(hypno_path)
    annotations = hyp_reader.readAnnotations()
    annotations_by_epoch = []
    for n in np.arange(hyp_reader.annotations_in_file - 1):
        annotations_by_epoch.extend([annotations[2][n]] * int(annotations[1][n]/EPOCH_LENGTH))
    for channel in range(psg_reader.signals_in_file - 1):
        buf = psg_reader.readSignal(channel)
        sample_frequency = psg_reader.getSampleFrequency(channel)
        for epoch_id, epoch_signal in enumerate(chunk(buf, sample_frequency)):
            if epoch_id >= len(epochs):
                epochs.append(Epoch(epoch_id))#, annotations_by_epoch[epoch_id]))
            epochs[epoch_id].add_channel(channel, epoch_signal, sample_frequency)

    # TODO: some files have diff # of annotations/epochs, missing?
    # print(psg_path, len(annotations_by_epoch), len(epochs))
    # if len(annotations_by_epoch) != len(epochs):
    #     print("NOT EQUAL")

    return epochs


