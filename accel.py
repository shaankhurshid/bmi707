from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import pandas as pd

# input
X_train = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/accel_flat.tsv',sep='\t',header=None)
X_train_values = X_train.values
y_train = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/outcome.tsv',sep='\t',header=None)
y_train = y_train.iloc[:,0]

# model
mod = Sequential()
mod.add(Dense(256, activation='relu', input_dim=11230))
mod.add(Dense(256, activation='relu'))
mod.add(Dense(256, activation='relu'))
mod.add(Dense(256, activation='relu'))
mod.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.05)
mod.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

mod.fit(X_train.values,y_train,
        epochs=50,batch_size=16)