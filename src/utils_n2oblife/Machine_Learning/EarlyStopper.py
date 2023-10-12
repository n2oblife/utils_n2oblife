import numpy as np
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.bool_continuous = False
        self.continuous = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.bool_continuous = False
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.bool_continuous = True
            if self.counter >= self.patience:
                return True
        else :
            self.bool_continuous = True
        return False
    
    def earlybreak(self):
        if self.bool_continuous :
            self.continuous += 1
        else :
            if self.continuous > 0 : 
                self.continuous -= 1

        if self.continuous >= 10:
            return True
        else :
            return False
