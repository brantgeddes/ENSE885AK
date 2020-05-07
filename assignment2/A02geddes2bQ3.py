###############################################################################################
# -
# - Title:          A02geddes2bQ3.py
# - Author:         Brant Geddes
# - Date:           Febuary 2020
# -
# - Description:    Implements Softmax Regression on the MNIST dataset to classify images into 
# -                 digits ranging from 0-9
# - Methods:        Trained using batch gradient descent
# - 
###############################################################################################

###### Imports ######
# Used when loading data
import gzip
# Used for matrix math and array methods
import numpy as np
# Used for randomly sampling and initializing weights
from random import randint, uniform
# Used in softmax function
from math import exp

# Raise numpy warnings to exceptions 
np.seterr(all='raise')

class Model:
    # Size of square image
    image_size = 28
    # Number of datapoints
    datapoints = 60000

    # Number of features
    features = (image_size**2)
    # Number of output classes
    classes = 10 # 0 - 9

    # Training Epochs
    epochs = 100
    # Training batch size
    batch = 512
    # Learning rate
    learning = 0.001

    #Number of test datapoints to split out
    test_train_split = 6000

    # data array
    data = None
    # labels array
    labels = None

    # test data array
    test_data = None
    # test labels array
    test_labels = None

    # bias vector
    b = None
    # weight matrix
    W = None

    # Placeholder for matplotlib plot variable
    plt = None

    # Constructor
    # data_path is file path to data file (in .gz form)
    # label_path is file path to label file (in .gz form)
    # Loads data and labels, formats them, normalizes data, and splits test and train datasets
    def __init__(self, data_path, label_path):
        # Open GZIP files
        data_raw = gzip.open(data_path, 'r')
        label_raw = gzip.open(label_path, 'r')
        # Remove leading data
        data_raw.read(16)
        label_raw.read(8)
        # Load MNIST data into numpy array
        buf = data_raw.read(self.image_size**2 * self.datapoints)
        self.data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        self.data = self.normalize(self.data)
        self.data = self.data.reshape(self.datapoints, self.image_size, self.image_size, 1)
        self.data = self.data.reshape(self.datapoints, 1, self.image_size**2)
        # Load MNIST labels into array
        buf = label_raw.read()
        self.labels = np.array([ x for x in buf ])
        # Close file handles
        data_raw.close()
        label_raw.close()
        # One-Hot Encode Labels
        self.labels = self.one_hot(self.labels)
        # Test/Train Split
        self.test_data = self.data[: self.test_train_split]
        self.test_labels = self.labels[: self.test_train_split]
        self.data = self.data[self.test_train_split :]
        self.labels = self.labels[self.test_train_split :]
        
    # Initialize
    # Initializes weight and bias
    # Fills Bias vector with zeros, fills Weight matrix with random values
    def initialize(self):
        self.W = np.random.normal(0, 1, (self.features, self.classes))
        self.b = np.zeros(self.classes)

    # Softmax 
    # Takes a matrix of weighted vectors
    # Returns a matrix of normalized probability vectors
    def softmax(self, X):
        Z = np.exp(X)
        return Z.T / np.sum(Z, axis=1)
     
    # Prediction -> SOFTMAX(X*W + b)
    # X is a matrix of input vectors
    # Returns a matrix of prediction vectors
    def predict(self, X):
        return self.softmax(X.reshape(-1, self.features).dot(self.W) + self.b)
    
    # Fit
    # _Y is prediction vector, Y is label vector, and X is feature vector
    # Update weights based on gradient
    def fit(self, _Y, Y, X):
        _Y = self.one_hot(_Y.T.argmax(axis=1))
        _W = self.W.T
        for index in range(len(_Y)):
            for i in range(self.classes):
                # Bias update
                self.b = self.b - self.learning * (_Y[index][i] - Y[index][i]);
                # Weight update
                _W[i] = self.normalize(_W[i] - self.learning * ((_Y[index][i] - Y[index][i]) * X[index]))
        self.W = _W.T

    # Log loss (cross entropy)
    # _Y is a matrix of prediction vectors
    # Y is a matrix of one-hot labels
    def loss(self, _Y, Y):
        return -1 * np.sum(np.log(_Y.T) * Y, axis=1)

    # Total Log loss (average cross entropy)
    # loss is the returned value from the loss function
    def total_loss(self, loss):
        return np.sum(loss) / len(loss)

    # Accuracy (positive hits / total)
    # _Y is a matrix of prediction vectors
    # Y is a matrix of one-hot labels
    def accuracy(self, _Y, Y):
        return ((_Y.T.argmax(axis=1) == Y.argmax(axis=1)).sum()) / len(_Y.T)

    # Normalize
    # Returns normalized vector
    def normalize(self, vector):
        max = np.amax(vector)
        normalize = lambda a: a / max
        return normalize(vector)

    # Sample
    # Returns random integer index inside the training dataset
    def sample(self):
        return randint(0, self.datapoints - self.test_train_split) - 1

    # Shuffle
    # Shuffles the data and labels to re-order all datapoints randomly
    def shuffle(self):
        new_data = []
        new_labels = []
        for i in range(len(self.data)):
            index = randint(0, (self.datapoints - self.test_train_split - i - 1))
            new_data.append(self.data[index])
            new_labels.append(self.labels[index])
        self.data = np.array(new_data)
        self.labels = np.array(new_labels)

    # Report
    # Reports dataset information
    def report(self):
        print()
        print('Report:')
        print()
        print('Datapoints: ' + str(self.datapoints))
        print('Image Size: ' + str(self.image_size))
        print('Features:   ' + str(self.features))
        print('Classes:    ' + str(self.classes))
        print()

    # One Hot
    # Converts passed vector to an numpy array of one-hot vectors
    def one_hot(self, vector):
        y = []
        for x in vector:
            _y = [0] * 10
            _y[x] = 1
            y.append(_y)
        return np.array(y)

# Load model, initialize weights, report dataset shape, and train model
model = Model('../datasets/MNIST/train-images-idx3-ubyte.gz', '../datasets/MNIST/train-labels-idx1-ubyte.gz')
# Initialize weights and bias
model.initialize()
# Print model parameters
model.report()

# Loop counter
counter = 0
##
# Accuracy and Loss tracking
loss= []
test_loss = []
accuracy = []
test_accuracy = []
##
while counter < model.epochs:
    # Shuffle training data
    model.shuffle()
    # Make prediction for fit function
    prediction = model.predict(model.data[:model.batch])
    model.fit(prediction, model.labels[:model.batch], model.data[:model.batch])
    # Make prediction for training metrics and calculate training accuracy and loss
    prediction = model.predict(model.data[:model.batch])
    loss.append(model.total_loss(model.loss(prediction, model.labels[:model.batch])))
    accuracy.append(model.accuracy(prediction, model.labels[:model.batch]) * 100)
    # Make prediction for testing metrics and calculate testing accuracy and loss
    test_prediction = model.predict(model.test_data[:])
    test_loss.append(model.total_loss(model.loss(test_prediction, model.test_labels[:])))
    test_accuracy.append(model.accuracy(test_prediction, model.test_labels[:]) * 100)
    # Print Epoch statistics
    print("Epoch:    " + str(counter))
    print("Training: Loss " + "%.2f" % loss[-1] + " | Acc. " + "%.2f" % accuracy[-1] + "%")
    print("Testing:  Loss " + "%.2f" % test_loss[-1] + " | Acc. " + "%.2f" % test_accuracy[-1] + "%")
    counter += 1

# Plot
try:
    import matplotlib.pyplot as plt
    epochs = range(0, model.epochs)
    loss = [ l * 20 for l in loss ]
    test_loss = [ l * 20 for l in test_loss ]
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, test_loss, label='Testing Loss')
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Testing Accuracy')
    plt.ylabel('%Acc / 20xLoss')
    plt.xlabel('Epochs')
    plt.title('MNIST Logistic Regression (Q3)')
    plt.legend(loc='upper left')
    plt.show()
except:
    print("Unable to display plot")
    
# Confusion Matrix
print()
print("Confusion Matrix:")
print()
prediction = model.predict(model.test_data[:]).T.argmax(axis=1)
confusion = np.zeros((10, 10))
for i, p in enumerate(prediction):
    y = model.test_labels[i].argmax()
    confusion[p][y] += 1
print("~\tzero\tone\ttwo\tthree\tfour\tfive\tsix\tseven\teight\tnine")
for i in range(model.classes):
    print(str(i) + ": ", end="\t")
    for j in range(model.classes):
        print(int(confusion[i][j]), end="\t")
    print()
