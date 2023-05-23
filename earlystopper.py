import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mIoU = -np.inf

    def early_stop(self, mIoU):
        if mIoU > self.best_mIoU:
            self.best_mIoU = mIoU
            self.counter = 0
        elif mIoU < (self.best_mIoU - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
