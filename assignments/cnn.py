# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, optimizers
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, GlobalMaxPooling1D, Input
from sklearn import preprocessing

# Load data
colnames = pd.array(range(1,112321)) # The number of time intervals in the database
train_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_data.csv', names = colnames, header = None)
train_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_data.csv', names = colnames, header = None)

# Creating labels
zeros = pd.Series(np.zeros(694))
ones = pd.Series(np.ones(694))
y_train = zeros.append(ones)

# Combine cases and controls
x_train = pd.concat([train_controls, train_cases])

# Downsampling
x_train_reduced = x_train.groupby(np.arange(len(x_train.columns))//64, axis=1).mean()

# Remove NAs
x_train_reduced[np.isnan(x_train_reduced)] = 0

# Scale
x_train_processed = preprocessing.scale(x_train_reduced,axis=0)

# Re-shaping data
x_train_reshaped = x_train_processed.reshape(len(x_train_processed), 1755, 1)

# Specify CNN
input = Input(shape=(1755,1))
x = Conv1D(filters=1,kernel_size=50,strides=4)(input)
x = MaxPool1D(2)(x)
x = Conv1D(filters=10,kernel_size=10,strides=2)(x)
x = MaxPool1D(2)(x)
x = Dense(256, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=pred)

opt = optimizers.Adam(lr=0.01)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train_reshaped, y_train, epochs =100, verbose = 1)