class Epoch(object):

    def __init__(self, epoch_id):
        self.epoch_id = epoch_id
        self.channels = {}
        self.sample_frequencies = {}
    
    def add_channel(self, channel, signals, sample_frequency):
        self.channels[channel] = signals
        self.sample_frequencies[channel] = sample_frequency
    
