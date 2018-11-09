import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import confusion_matrix


PLOT_PATH = './output/'

def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
    plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
    plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig(os.path.join(PLOT_PATH, 'loss_curve.png'))

    plt.clf()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='Train')
    plt.plot(np.arange(len(valid_accuracies)), valid_accuracies, label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.savefig(os.path.join(PLOT_PATH, 'accuracy_curve.png'))


def plot_confusion_matrix(results, class_names):
	results = list(zip(*results))
	cm = confusion_matrix(results[0], results[1])
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.clf()
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title('Normalized Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	fmt = '.2f'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()

	plt.savefig(os.path.join(PLOT_PATH, 'confusion_matrix.png'))