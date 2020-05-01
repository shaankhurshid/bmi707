# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, optimizers
from tensorflow.keras.layers import Dense, LSTM
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

# Save copy of downsampled
np.savetxt('/mnt/ml4cvd/projects/skhurshid/bmi707/x_train_downsampled_3510.tsv',x_train_reshaped,fmt='%.1f')
np.savetxt('/mnt/ml4cvd/projects/skhurshid/bmi707/y_train.tsv',y_train,fmt='%.1f')

timesteps = 1755
data_dim = 1

# Model
model = Sequential()
model.add(LSTM(16, return_sequences=False,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(Dense(1, activation='sigmoid'))

# Compile
opt = optimizers.SGD(lr=0.01)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Fit
model.fit(x_train_reshaped, y_train, epochs =100, verbose = 1)


