import torch
import torch.nn as nn
import torch.optim as optim
from features.edf_loader import PhysiobankEDFLoader
from features.etl_edf import EDFEpochDataset
from helper import spark_helper
from helper.utils import train
from model.cnn import CNN
from torch.utils.data import DataLoader


MODEL_TYPE = 'CNN'
BATCH_SIZE = 32
NUM_WORKERS = 0

if __name__ == "__main__":
    #spark_session = spark_helper.start_spark()
    #sc = spark_session.sparkContext

    # Load data
    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=True)

    if MODEL_TYPE == 'CNN':
        model = CNN()
    else:
        raise AssertionError('Model type does not exist')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Create train/valid/test sets
    # TODO: no shuffling at the moment
    train_dataset = EDFEpochDataset(records)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Train
    print("Training (model={}, workers={}, batch_size={})".format(MODEL_TYPE, NUM_WORKERS, BATCH_SIZE))
    train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, 0, print_freq=10)
    
    