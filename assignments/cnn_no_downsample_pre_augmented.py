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

# Load regular data
colnames = pd.array(range(1,112321)) # The number of time intervals in the database
val_controls = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_controls_data.csv', names = colnames, header = None)
val_cases = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/holdout_cases_data.csv', names = colnames, header = None)

# Load augmented data
x_train_processed = pd.read_csv('/mnt/ml4cvd/projects/skhurshid/bmi707/train_data_augmented_processed.csv', names = colnames, header = None)

# Creating labels
## Train
zeros = pd.Series(np.zeros(1388))
ones = pd.Series(np.ones(1388))
y_train = zeros.append(ones)

## Test
zeros = pd.Series(np.zeros(len(val_controls)))
ones = pd.Series(np.ones(len(val_cases)))
y_val = zeros.append(ones)

# Combine cases and controls
x_val = pd.concat([val_controls,val_cases])

# Remove NAs
x_val[np.isnan(x_val)] = 0

# Scale
x_val_processed = preprocessing.scale(x_val,axis=0)

# Re-shaping data
x_train_array = np.array(x_train_processed)
x_train_reshaped = x_train_array.reshape(len(x_train_array), 112320, 1)
x_val_reshaped = x_val_processed.reshape(len(x_val_processed), 112320, 1)

# Shuffle
x_train_final, y_train_final = shuffle(x_train_reshaped, y_train)

# Specify CNN
input = Input(shape=(112320,1))
x = Conv1D(filters=1,kernel_size=20000,strides=50)(input)
x = Conv1D(filters=100,kernel_size=100,strides=5)(x)
x = Dropout(0.1)(x)
x = MaxPool1D(2)(x)
x = Conv1D(filters=25,kernel_size=5,strides=1)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.1)(x)
x = GlobalMaxPooling1D()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=pred)

# Compile
opt = optimizers.Adam(lr=0.000005)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

# Early stopping
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

# Fit
model.fit(x_train_reshaped, y_train, epochs =25, verbose = 1,shuffle=True,
          validation_data=(x_val_reshaped,y_val),callbacks=[es])

# Evaluate
model.evaluate(x_val_reshaped,y_val)

# Plot training & validation accuracy values
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/mnt/ml4cvd/projects/skhurshid/bmi707/af_cnn_accuracy.pdf')
plt.clf()

# Plot training & validation loss values
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/mnt/ml4cvd/projects/skhurshid/bmi707/af_cnn_loss.pdf')
plt.clf()

# AUC
## Predictions
val_preds = model.predict(x_val_reshaped)
## Accuracy
acc_val = len(np.where(np.round(val_preds[:,0]) == y_val)[0]) / float(len(y_val))
## ROC
# Compute ROC curve and ROC area in validation set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(y_val,np.round(val_preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

train_data_processed.to_csv(    '/mnt/ml4cvd/projects/skhurshid/bmi707/train_data_augmented_processed.csv')

