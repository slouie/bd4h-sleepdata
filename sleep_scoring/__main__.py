import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from features.datasets import EpochDataset, RecordSampler, WeightedRecordSampler, RNNDataset
from features.edf_loader import PhysiobankEDFLoader
from features.etl_edf import extract_features
from helper import spark_helper
from helper.plots import plot_learning_curves, plot_confusion_matrix, print_metrics
from helper.utils import train, evaluate
from model.cnn import SimpleCNN
from model.rcnn import RCNN, SimpleRNN
from optparse import OptionParser
from torch.utils.data import DataLoader

torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed(777)

# Path for saving model
PATH_OUTPUT = "./output/bestmodels/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_TRAINING_EPOCHS = 1
MODEL_TYPE = 'SimpleRNN'
BATCH_SIZE = 32
NUM_WORKERS = 0

CLASS_MAP = {
    'Sleep stage W' : 0,
    'Sleep stage 1' : 1,
    'Sleep stage 2' : 2,
    'Sleep stage 3' : 3,
    'Sleep stage 4' : 3, # Map stage 4 to 3
    'Sleep stage R' : 4,
}

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m', '--best_model', dest='best_model')
    parser.add_option('-t', '--train', dest='train_model', default=MODEL_TYPE)
    parser.add_option('-p', '--plot', action="store_true", dest='plot', default=False)
    options, args = parser.parse_args(sys.argv)

    spark_session = spark_helper.start_spark()
    sc = spark_session.sparkContext

    if options.train_model:
        MODEL_TYPE = options.train_model

    if options.best_model:
        MODEL_TYPE = options.best_model
        print("Running best model: ", options.best_model)

    # Load data
    loader = PhysiobankEDFLoader()
    records = loader.load_sc_records(save=False)
    feature_paths = extract_features(sc, records, save=False)

    lr = 0.001

    if MODEL_TYPE == 'SimpleCNN':
        model = SimpleCNN()
    elif MODEL_TYPE == 'SimpleRNN':
        model = SimpleRNN()
    elif MODEL_TYPE == 'RCNN':
        model = RCNN()
        lr = 0.0001
    else:
        raise AssertionError('Model type does not exist')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    criterion.to(device)

    # Create train/valid/test sets
    print("Creating dataset ...")

    train_valid_split = int(0.85 * len(feature_paths))
    train_paths, valid_paths = feature_paths[0:train_valid_split], feature_paths[train_valid_split:]

    if MODEL_TYPE in ['SimpleCNN', 'RCNN']:
        train_dataset = EpochDataset(train_paths, CLASS_MAP)
        valid_dataset = EpochDataset(valid_paths, CLASS_MAP)
        train_sampler = WeightedRecordSampler(train_dataset)
        valid_sampler = RecordSampler(valid_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    elif MODEL_TYPE == 'SimpleRNN':
        train_dataset = RNNDataset(train_paths, CLASS_MAP)
        valid_dataset = RNNDataset(valid_paths, CLASS_MAP)
        train_sampler = WeightedRecordSampler(train_dataset)
        valid_sampler = RecordSampler(valid_dataset)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Train
    if not options.best_model:
        print("Training (model={}, epochs={}, batch_size={})".format(MODEL_TYPE, NUM_TRAINING_EPOCHS, BATCH_SIZE))

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
        if options.plot:
            plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, MODEL_TYPE)

    # Test best model
    print("Testing model ...")
    if torch.cuda.is_available():
        best_model = torch.load(os.path.join(PATH_OUTPUT, '{}.pth'.format(MODEL_TYPE)))
    else:
        best_model = torch.load(os.path.join(PATH_OUTPUT, '{}.pth'.format(MODEL_TYPE)), map_location='cpu')
    test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion, print_freq=100)

    class_names = ['W', '1', '2', '3', 'R']
    if options.plot:
        plot_confusion_matrix(test_results, class_names, MODEL_TYPE)

    y_true, y_pred = zip(*test_results)
    print_metrics(MODEL_TYPE, y_pred, y_true, save=options.plot)
