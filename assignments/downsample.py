# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense

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
x_train_reduced = x_train.groupby(np.arange(len(x_train.columns))//32, axis=1).mean()

# Re-shaping data
x_train_reshaped = x_train.values.reshape(-1, 1, 3510)

# Save copy of downsampled
np.savetxt('/mnt/ml4cvd/projects/skhurshid/bmi707/x_train_downsampled_3510.tsv',x_train_reshaped,fmt='%.1f')
np.savetxt('/mnt/ml4cvd/projects/skhurshid/bmi707/y_train.tsv',y_train,fmt='%.1f')

# Model
model = Sequential()
model.add(layers.LSTM(128, input_shape=(1,3510)))
model.add(Dense(1),activation='sigmoid')

# Compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit
model.fit(x_train_reshaped, y_train, epochs = 10, verbose = 1)
