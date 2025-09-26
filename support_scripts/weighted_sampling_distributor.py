import numpy as np
import torch
from sklearn.metrics import confusion_matrix



def analyze_weaknesses_produce_weights(stats_pth):
    stats = torch.load(stats_pth, weights_only=False)


    targets, predictions = stats['train_targets'], stats['train_predictions']

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)

    diagonals = np.diag(cm)
    eps = 3e-2 # expected max weight = 4.6; 1e-3 => 6.9

    # log-inverse weighting
    raw_weights = np.log(1.0 / (diagonals + eps))

    # add +1 to avoid zero-weights for perfect classes
    raw_weights = raw_weights + 1.0

    return raw_weights
