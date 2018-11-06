from model.models import Epoch
import pyedflib

EPOCH_LENGTH = 30

class FeatureConstruction(object):

    @staticmethod
    def construct(sc, rdd):
        return rdd


def chunk(signal, sample_frequency):
    num_samples = EPOCH_LENGTH * sample_frequency
    for i in range(0, len(signal), num_samples):
        yield signal[i:i+num_samples]


def load_rdd_from_edf(sc, edf_paths):
    epochs = []
    for psg_path, hypno_path in edf_paths:
        reader = pyedflib.EdfReader(psg_path)
        channels = []
        for channel in range(reader.signals_in_file - 1):
            buf = reader.readSignal(channel)
            sample_frequency = reader.getSampleFrequency(channel)
            for epoch_id, epoch_signal in enumerate(chunk(buf, sample_frequency)):
               channels.append(Epoch(epoch_id, channel, epoch_signal, sample_frequency))
        epochs.append(channels)
    return epochs