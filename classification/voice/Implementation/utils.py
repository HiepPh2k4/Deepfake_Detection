import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def calculate_eer(fpr, tpr, thresholds):
    """Calculate Equal Error Rate"""
    diff = np.abs(fpr - (1 - tpr))
    min_diff_idx = np.argmin(diff)
    eer = fpr[min_diff_idx]
    return eer, thresholds[min_diff_idx]
