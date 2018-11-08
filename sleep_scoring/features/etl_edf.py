import numpy as np
import pyedflib
import torch
from model.models import Epoch
from torch.utils.data import TensorDataset


EPOCH_LENGTH = 30

class FeatureConstruction(object):

    @staticmethod
    def construct(sc, rdd, model_type):
        # TODO: use spark later
        if model_type == 'CNN':
            epochs = [epoch for epoch_list in rdd for epoch in epoch_list]
            data = np.zeros((len(epochs), 6, 3000))
            # TODO: use expert annotations
            target = np.ones(len(epochs)) 
            for i, epoch in enumerate(epochs):
                for channel, signal in epoch.channels.items():
                    if epoch.sample_frequencies[channel] == 100:
                        data[i][channel] = signal
                    elif epoch.sample_frequencies[channel] == 1:
                        data[i][channel] = np.repeat(signal, 100, axis=0)
                    else:
                        raise Exception('Unexpected sample frequency')
            data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
            target_tensor = torch.from_numpy(target).type(torch.LongTensor)
            dataset = TensorDataset(data_tensor, target_tensor)
        return dataset


def chunk(signal, sample_frequency):
    num_samples = EPOCH_LENGTH * sample_frequency
    for i in range(0, len(signal), num_samples):
        yield signal[i:i+num_samples]


def load_rdd_from_edf(sc, edf_paths):
    '''
    TODO: not really rdd, use spark later
    return: list of list of epochs per recording
    '''
    epochs_per_recording = []
    for psg_path, hypno_path in edf_paths:
        epochs = []
        reader = pyedflib.EdfReader(psg_path)
        for channel in range(reader.signals_in_file - 1):
            buf = reader.readSignal(channel)
            sample_frequency = reader.getSampleFrequency(channel)
            for epoch_id, epoch_signal in enumerate(chunk(buf, sample_frequency)):
                if epoch_id >= len(epochs):
                    epochs.append(Epoch(epoch_id))
                epochs[epoch_id].add_channel(channel, epoch_signal, sample_frequency)
        epochs_per_recording.append(epochs)
    return epochs_per_recording


