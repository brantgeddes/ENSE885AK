
from tensorflow.keras.callbacks import Callback
from time import time

# Callback class for model training
# Tracks training/testing metrics over epochs for analysis later
class AggregateStats(Callback):

    # Initialize variables
    def __init__(self):
        self.accuracy = [0]
        self.loss = [1]
        self.val_accuracy = [0]
        self.val_loss = [1]
        self.epochs = [0]
        self.m_epoch = 0
        self.m_accuracy = 0
        self.m_loss = 0

    # Store epoch begin time
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time()

    # Push new values at end of epoch and calculate running time
    def on_epoch_end(self, epoch, logs=None):
        self.accuracy.append(logs['accuracy'])
        self.loss.append(logs['loss'])
        self.val_accuracy.append(logs['val_accuracy'])
        self.val_loss.append(logs['val_loss'] if logs['val_loss'] < 1 else 1)
        self.epochs.append(epoch + 1)
        self.epoch_time = time() - self.start_time
        if (logs['val_accuracy'] > self.m_accuracy):
            self.m_accuracy = logs['val_accuracy']
            self.m_loss = logs['val_loss']
            self.m_epoch = epoch + 1
