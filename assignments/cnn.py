# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, optimizers
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, GlobalMaxPooling1D, Input, Dropout
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.utils import shuffle

# Load data
colnames = pd.array(range(1,112321)) # The number of time intervals in the database
train_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_data.csv', names = colnames, header = None)
train_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_data.csv', names = colnames, header = None)
val_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_controls_data.csv', names = colnames, header = None)
val_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_cases_data.csv', names = colnames, header = None)

# Creating labels
## Train
zeros = pd.Series(np.zeros(694))
ones = pd.Series(np.ones(694))
y_train = zeros.append(ones)

## Test
zeros = pd.Series(np.zeros(len(val_controls)))
ones = pd.Series(np.ones(len(val_cases)))
y_val = zeros.append(ones)

# Combine cases and controls
x_train = pd.concat([train_controls, train_cases])
x_val = pd.concat([val_controls,val_cases])

# Downsampling
x_train_reduced = x_train.groupby(np.arange(len(x_train.columns))//64, axis=1).mean()
x_val_reduced = x_val.groupby(np.arange(len(x_val.columns))//64, axis=1).mean()

# Remove NAs
x_train_reduced[np.isnan(x_train_reduced)] = 0
x_val_reduced[np.isnan(x_val_reduced)] = 0

# Scale
x_train_processed = preprocessing.scale(x_train_reduced,axis=0)
x_val_processed = preprocessing.scale(x_val_reduced,axis=0)

# Re-shaping data
x_train_reshaped = x_train_processed.reshape(len(x_train_processed), 1755, 1)
x_val_reshaped = x_val_processed.reshape(len(x_val_processed), 1755, 1)

# Shuffle
x_train_final, y_train_final = shuffle(x_train_reshaped, y_train)

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

opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train_final, y_train_final, epochs =100, verbose = 1,shuffle=True,
          validation_data=(x_val_reshaped,y_val))

# Evaluate
model.evaluate(x_val_reshaped,y_val)

# Validate performance
## Predictions
#val_preds = model.predict(x_val_reshaped)
## Accuracy
#acc_val = len(np.where(np.round(val_preds[:,0]) == y_val)[0]) / float(len(y_val))
## ROC
# Compute ROC curve and ROC area in validation set
#fpr_val = dict()
#tpr_val = dict()
#roc_auc_val = dict()
#fpr_val, tpr_val, _ = roc_curve(y_val,np.round(val_preds[:,0],decimals=0))
#roc_auc_val = auc(fpr_val, tpr_val)
