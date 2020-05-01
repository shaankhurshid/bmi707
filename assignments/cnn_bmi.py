# Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, optimizers
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPool1D, GlobalMaxPooling1D, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.utils import shuffle
import tensorflow as tf

# Load data
colnames = pd.array(range(1,112321)) # The number of time intervals in the database
train_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_data.csv', names = colnames, header = None)
train_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_data.csv', names = colnames, header = None)
val_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_controls_data.csv', names = colnames, header = None)
val_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_cases_data.csv', names = colnames, header = None)

# Load labels
## Train
y_train_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_cases_bmi.csv', header = None)
y_train_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_controls_bmi.csv', header = None)
y_val_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_cases_bmi.csv', header = None)
y_val_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_controls_bmi.csv', header = None)

# Combine cases and controls
x_train = pd.concat([train_controls, train_cases])
x_val = pd.concat([val_controls,val_cases])

# Combine cases and controls
y_train = pd.concat([y_train_controls, y_train_cases])
y_val = pd.concat([y_val_controls,y_val_cases])

# Remove NAs
x_train[np.isnan(x_train)] = 0
x_val[np.isnan(x_val)] = 0

# Scale
x_train_processed = preprocessing.scale(x_train,axis=0)
x_val_processed = preprocessing.scale(x_val,axis=0)

# Re-shaping data
x_train_reshaped = x_train_processed.reshape(len(x_train_processed), 112320, 1)
x_val_reshaped = x_val_processed.reshape(len(x_val_processed), 112320, 1)

# Shuffle
x_train_final, y_train_final = shuffle(x_train_reshaped, y_train)

# Specify CNN
input = Input(shape=(112320,1))
x = Conv1D(filters=1,kernel_size=200,strides=3)(input)
x = MaxPool1D(2)(x)
x = Conv1D(filters=200,kernel_size=10,strides=5)(x)
x = MaxPool1D(2)(x)
x = Conv1D(filters=25,kernel_size=2,strides=1)(x)
x = Dropout(0.1)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
x = GlobalMaxPooling1D()(x)
pred = Dense(1, activation='relu')(x)

model = Model(inputs=input, outputs=pred)

# Compile
opt = optimizers.Adam(lr=0.001)
model.compile(optimizer=opt,loss='mse')

# Early stopping
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

model.fit(x_train_final, y_train_final, epochs =30, verbose = 1,shuffle=True,
          validation_data=(x_val_reshaped,y_val),callbacks=[es])

# Evaluate
model.evaluate(x_val_reshaped,y_val)

## Predictions
val_preds = model.predict(x_val_reshaped)

# Plot correlation
from matplotlib import pyplot as plt

plt.scatter(val_preds,y_val)
plt.savefig('/mnt/ml4cvd/projects/skhurshid/bmi707/corr_bmi.pdf')
plt.clf()

# Plot training & validation accuracy values
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/mnt/ml4cvd/projects/skhurshid/bmi707/bmi_accuracy.pdf')
plt.clf()

# Plot training & validation loss values
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/mnt/ml4cvd/projects/skhurshid/bmi707/bmi_loss.pdf')
plt.clf()

# Output validation data and predictions
## Inputs
y_val_df = pd.DataFrame(y_val)
y_val_df.rename(columns={0:"actual_bmi"})
y_val_df.to_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/val_actual_bmi.csv')

## Predictions
val_preds_df = pd.DataFrame(val_preds)
val_preds_df.rename(columns={0:"predicted_bmi"})
val_preds_df.to_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/val_preds_bmi.csv')

# Save model and architecture to single file
model.save("/mnt/ml4cvd/projects/skhurshid/bmi707/cnn_bmi.h5")
print("Saved model to disk")