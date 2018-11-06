class Epoch(object):

    # TODO: Maybe put all channels in one model?
    def __init__(self, epoch_id, channel, signals, sample_rate):
        self.epoch_id = epoch_id
        self.channel = channel
        self.signals = signals
        self.sample_rate = sample_rate
