###############################################################################################
# -
# - Title:          A04geddes2bQ2.py
# - Author:         Brant Geddes
# - Date:           March 2020
# -
# - Description:    Uses Tensorflow v2 to train a classification model with convolutional layers 
# -                 on the MNIST dataset and plots the testing and training accuracy and loss
# - 
###############################################################################################

## Imports
# Load tensorflow library
import tensorflow as tf
# Load MNIST dataset through tensorflow
mnist = tf.keras.datasets.mnist
# Import to_categorical for one-hot encoding
from keras.utils import to_categorical
# Import l2 regularizer
from keras.regularizers import l2 as reg_l2
##

## Hyperparameters
# Regularization Learning Rate
reg_learning = 0.01
# Dropout rate
dropout_rate = 0.01
##

## Metric tracking
accuracy = [0]
val_accuracy = [0]
loss = [1]
val_loss = [1]
epochs = [0]
## 

# Callback class for model training
# Tracks training/testing metrics over epochs for analysis later
class AggregateStats(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        accuracy.append(logs['accuracy'])
        loss.append(logs['loss'])
        val_accuracy.append(logs['val_accuracy'])
        val_loss.append(logs['val_loss'])
        epochs.append(epoch + 1)
# Load testing and training data and labels
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert features to float and normalize (max value in features is 255)
x_train, x_test = x_train / 255.0, x_test / 255.0
# Convert train and test labels to one-hot encoding
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# Format data for convolutional layer
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Build sequential model
model = tf.keras.models.Sequential([
    # Flatten array from [?, 28, 28] to [?, 784]
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Convolutional Layer
    tf.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
    # Pooling layer
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # Flatten pooling output
    tf.keras.layers.Flatten(),
    # First hidden layer
    tf.keras.layers.Dense(500, activity_regularizer=reg_l2(reg_learning), activation='relu'),
    # Dropout
    tf.keras.layers.Dropout(dropout_rate),
    # Second hidden layer
    tf.keras.layers.Dense(500, activity_regularizer=reg_l2(reg_learning), activation='relu'),
    # Softmax output
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile, using cross entropy as loss function and stochastic gradient descent as optimizer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# Fit model with 250 Epochs
model.fit(x_train, y_train, epochs=250, validation_data=(x_test, y_test), callbacks=[AggregateStats()])

# Plot accuracy/epochs and loss/epochs for testing and training
try:
    import matplotlib.pyplot as plt
    plt.plot(epochs, accuracy, label='Accuracy')
    plt.plot(epochs, loss, label='Loss')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(epochs, val_accuracy, label='Accuracy')
    plt.plot(epochs, val_loss, label='Loss')
    plt.title('Testing Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy / Loss')
    plt.legend(loc='upper left')
    plt.show()

except:
    print("No matplotlib available")
