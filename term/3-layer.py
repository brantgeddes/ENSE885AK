###############################################################################################
# -
# - Title:          3-layer.py
# - Author:         Brant Geddes
# - Date:           March 2020
# -
# - Description:    Trains on NSL-KDD dataset using a deep encoder followed by three 
# -                 hidden layers and a softmax classifier
# - 
###############################################################################################

## Library Imports
import pandas as pd
import sklearn.preprocessing as skp
import tensorflow as tf
from keras.utils import to_categorical
from keras.regularizers import l2 as reg_l2

## Custom Imports
from constants import datapoints, oversample_rate,                 \
                      train_test_split, total_epochs,              \
                      hidden_layer1, hidden_layer2, hidden_layer3, \
                      reg_learning
from mappings import protocol, service, flag, attacks
import record
from callbacks import AggregateStats


# File name
filename = 'presentation-oversampled-encoded-3-layer-' + str(hidden_layer1) + '-' + str(hidden_layer2) + '-' + str(hidden_layer3)
# Save trial
save_this = True


# Load and shuffle data
traindata = pd.read_csv('../datasets/NSL_KDD/KDD+.txt', header=None)
traindata = traindata.sample(frac=1).reset_index(drop=True)

# Enumerate string values
traindata[1] = traindata[1].apply(lambda x: protocol.index(x))
traindata[2] = traindata[2].apply(lambda x: service.index(x))
traindata[3] = traindata[3].apply(lambda x: flag.index(x))
traindata[41] = traindata[41].apply(lambda y: attacks[y])

# Split train and test data
testdata = traindata.iloc[int(datapoints * (1 - train_test_split) + 1):, :]
traindata = traindata.iloc[:int(datapoints * (1 - train_test_split)), :]

# Oversample U2R records
U2R_flag = traindata[41] == 3
oversampled = traindata[U2R_flag]
traindata = traindata.append([oversampled] * oversample_rate, ignore_index=True)

# Split features and labels
(x_train, y_train) = (traindata.iloc[:, :40], traindata.iloc[:, 41])
(x_test, y_test) = (testdata.iloc[:, :40], testdata.iloc[:, 41])

# Normalize features
scaler = skp.Normalizer().fit(x_train)
x_train = scaler.transform(x_train)
scaler = skp.Normalizer().fit(x_test)
x_test = scaler.transform(x_test)

# One-Hot encode labels
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Create callback object
stats = AggregateStats()

# Build sequential model
model = tf.keras.models.Sequential([

    # Encoder
    tf.keras.layers.Dense(39, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    # Classifier
    tf.keras.layers.Dense(hidden_layer1, activity_regularizer=reg_l2(reg_learning), activation='relu'),
    tf.keras.layers.Dense(hidden_layer2, activity_regularizer=reg_l2(reg_learning), activation='relu'),
    tf.keras.layers.Dense(hidden_layer3, activity_regularizer=reg_l2(reg_learning), activation='relu'),
    # Softmax output
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile, using cross entropy as loss function and stochastic gradient descent as optimizer
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
# Fit model 
model.fit(x_train, y_train, epochs=total_epochs, validation_data=(x_test, y_test), callbacks=[stats])

# Record metrics
if save_this:
    record.plot(filename, stats)
    record.stats(filename, stats)
    record.summary(filename, stats, total_epochs, model, y_test, x_test)
