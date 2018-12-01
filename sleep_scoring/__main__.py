import os
import torch
import torch.nn as nn
import torch.optim as optim
from features.datasets import EpochDataset, RecordSampler, TimeOrderedDataset
from features.edf_loader import PhysiobankEDFLoader
from features.etl_edf import extract_features
from helper import spark_helper
from helper.plots import plot_learning_curves, plot_confusion_matrix, save_metrics
from helper.utils import train, evaluate
from model.cnn import TsinalisCNN
from model.cnn import SimpleCNN
from model.rcnn import RCNN
from torch.utils.data import DataLoader


# Path for saving model
PATH_OUTPUT = "./output/bestmodels/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_TRAINING_EPOCHS = 1
MODEL_TYPE = 'SimpleCNN'
BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_MAP = {
    'Sleep stage W' : 0,
    'Sleep stage 1' : 1,
    'Sleep stage 2' : 2,
    'Sleep stage 3' : 3,
    'Sleep stage 4' : 4,
    'Sleep stage R' : 5,
}

if __name__ == "__main__":
    spark_session = spark_helper.start_spark()
    sc = spark_session.sparkContext

    # Load data
    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=True)
    #loader.print_record('data/sleep-cassette/SC4362F0-PSG.edf')
    #loader.print_record('data/sleep-cassette/SC4362FC-Hypnogram.edf')

    feature_paths = extract_features(sc, records)

    if MODEL_TYPE == 'SimpleCNN':
        model = SimpleCNN()
    elif MODEL_TYPE == 'TsinalisCNN':
        model = TsinalisCNN()
    elif MODEL_TYPE == 'RCNN':
        model = RCNN()
    else:
        raise AssertionError('Model type does not exist')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.to(device)
    criterion.to(device)

    # Create train/valid/test sets
    print("Creating dataset ...")

    # TODO: split train/valid better
    if MODEL_TYPE == 'SimpleCNN' or MODEL_TYPE == 'TsinalisCNN':
        train_dataset = EpochDataset(feature_paths[0:130], CLASS_MAP)
        valid_dataset = EpochDataset(feature_paths[130:], CLASS_MAP)
        train_sampler = RecordSampler(train_dataset)
        valid_sampler = RecordSampler(valid_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    elif MODEL_TYPE == 'RCNN':
        #152 total features
        train_dataset = TimeOrderedDataset(feature_paths[0:130], CLASS_MAP)
        valid_dataset = TimeOrderedDataset(feature_paths[130:], CLASS_MAP)
        # train_sampler = RecordSampler(train_dataset)
        # valid_sampler = RecordSampler(valid_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # # Train
    print("Training (model={}, workers={}, batch_size={})".format(MODEL_TYPE, NUM_WORKERS, BATCH_SIZE))

    best_val_acc = 0.0
    train_losses, train_accuracies = [], []
    valid_losses, valid_accuracies = [], []
    for training_epoch in range(NUM_TRAINING_EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, training_epoch, print_freq=100)
        valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion, print_freq=100)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)

        is_best = valid_accuracy > best_val_acc
        if is_best:
            best_val_acc = valid_accuracy
            torch.save(model, os.path.join(PATH_OUTPUT, '{}.pth'.format(MODEL_TYPE)))

    plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, MODEL_TYPE)

    best_model = torch.load(os.path.join(PATH_OUTPUT, '{}.pth'.format(MODEL_TYPE)))
    test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion)

    class_names = ['W', '1', '2', '3', '4', 'R']
    plot_confusion_matrix(test_results, class_names, MODEL_TYPE)

    y_true, y_pred = zip(*test_results)
    save_metrics(MODEL_TYPE, y_pred, y_true)
