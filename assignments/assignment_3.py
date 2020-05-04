# Dependencies
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Input, Dropout, LSTM, Conv2D, MaxPool2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import roc_curve, roc_auc_score, auc
import os
import shutil,glob

## Read in labels
train = pd.read_csv('train.csv',header=None)
val = pd.read_csv('val.csv',header=None)

# Indices of files
train_cases = train[train.iloc[:,1] == 'Cardiomegaly']
train_controls = train[train.iloc[:,1] == 'No Finding']
val_cases = val[val.iloc[:,1] == 'Cardiomegaly']
val_controls = val[val.iloc[:,1] == 'No Finding']

# Function to move files to the right folders
def movefiles(in_path,out_path,key):
    for files in os.listdir(in_path):
        os.chdir(in_path)
        if files in list(key.iloc[:,0]):
            print(files)
            shutil.move(files,out_path)

# Run the move
movefiles(in_path='/home/Rebecca/bmi707_assignments/train', out_path='/home/Rebecca/bmi707_assignments/train_cases', key=train_cases)
movefiles(in_path='/home/Rebecca/bmi707_assignments/train', out_path='/home/Rebecca/bmi707_assignments/train_controls', key=train_controls)
movefiles(in_path='/home/Rebecca/bmi707_assignments/val', out_path='/home/Rebecca/bmi707_assignments/val_cases', key=val_cases)
movefiles(in_path='/home/Rebecca/bmi707_assignments/val', out_path='/home/Rebecca/bmi707_assignments/val_controls', key=val_controls)

# Preprocessing
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# define the batch size, if there is sufficient GPU memory, you can increase the batch size
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/train_data',  # the directory for the training data
        target_size=(250,250),  # resize the input images to accommodate the model
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary')

# Specify model
input = Input(shape=(250,250,3))
x = Conv2D(filters=3,kernel_size=50,strides=4)(input)
x = MaxPool2D(2)(x)
x = Conv2D(filters=10,kernel_size=10,strides=2)(x)
x = MaxPool2D(2)(x)
x = Dense(256, activation='relu')(x)
x = GlobalMaxPooling2D()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=pred)

# compile model
opt = Adam(lr=0.01)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

# fit model
nTraining = 1750
nValidation = 437
model.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)

# evaluate model
model.evaluate(validation_generator, steps = nValidation // batch_size)

# Get AUC
## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.654

# Define base models VGG16
base_model_unweighted = VGG16(weights=None, include_top=False, input_shape=[250,250,3])
base_model_weighted = VGG16(weights="imagenet", include_top=False, input_shape=[250,250,3])

# Define unweighted model

# get the output of the base model
x = base_model_unweighted.output
# get the output of the base model
x = Dense(256,activation='relu')(x)
# pool
x = GlobalAveragePooling2D()(x)
# add a layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# define the model to be trained
model_vgg_unweighted = Model(inputs=base_model_unweighted.input, outputs=predictions)

# Fix the base model weights
for layer in base_model_unweighted.layers:
    layer.trainable = False

# compile model
opt = Adam(lr=0.001)
model_vgg_unweighted.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# redefine generators
train_generator = train_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/train_data',  # the directory for the training data
        target_size=(250,250),  # resize the input images to accommodate the model
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary')# Dependencies
import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Input, Dropout, LSTM, Conv2D, MaxPool2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import roc_curve, roc_auc_score, auc
import os
import shutil,glob

## Read in labels
train = pd.read_csv('train.csv',header=None)
val = pd.read_csv('val.csv',header=None)

# Indices of files
train_cases = train[train.iloc[:,1] == 'Cardiomegaly']
train_controls = train[train.iloc[:,1] == 'No Finding']
val_cases = val[val.iloc[:,1] == 'Cardiomegaly']
val_controls = val[val.iloc[:,1] == 'No Finding']

# Function to move files to the right folders
def movefiles(in_path,out_path,key):
    for files in os.listdir(in_path):
        os.chdir(in_path)
        if files in list(key.iloc[:,0]):
            print(files)
            shutil.move(files,out_path)

# Run the move
movefiles(in_path='/home/Rebecca/bmi707_assignments/train', out_path='/home/Rebecca/bmi707_assignments/train_cases', key=train_cases)
movefiles(in_path='/home/Rebecca/bmi707_assignments/train', out_path='/home/Rebecca/bmi707_assignments/train_controls', key=train_controls)
movefiles(in_path='/home/Rebecca/bmi707_assignments/val', out_path='/home/Rebecca/bmi707_assignments/val_cases', key=val_cases)
movefiles(in_path='/home/Rebecca/bmi707_assignments/val', out_path='/home/Rebecca/bmi707_assignments/val_controls', key=val_controls)

# Preprocessing
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# define the batch size, if there is sufficient GPU memory, you can increase the batch size
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/train_data',  # the directory for the training data
        target_size=(250,250),  # resize the input images to accommodate the model
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary')

# Specify model
input = Input(shape=(250,250,3))
x = Conv2D(filters=3,kernel_size=50,strides=4)(input)
x = MaxPool2D(2)(x)
x = Conv2D(filters=10,kernel_size=10,strides=2)(x)
x = MaxPool2D(2)(x)
x = Dense(256, activation='relu')(x)
x = GlobalMaxPooling2D()(x)
pred = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input, outputs=pred)

# compile model
opt = Adam(lr=0.01)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

# fit model
nTraining = 1750
nValidation = 437
model.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)

# evaluate model
model.evaluate(validation_generator, steps = nValidation // batch_size)

# Get AUC
## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.654

# Define base models VGG16
base_model_unweighted = VGG16(weights=None, include_top=False, input_shape=[250,250,3])
base_model_weighted = VGG16(weights="imagenet", include_top=False, input_shape=[250,250,3])

# Define unweighted model

# get the output of the base model
x = base_model_unweighted.output
# get the output of the base model
x = Dense(256,activation='relu')(x)
# pool
x = GlobalAveragePooling2D()(x)
# add a layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# define the model to be trained
model_vgg_unweighted = Model(inputs=base_model_unweighted.input, outputs=predictions)

# Fix the base model weights
for layer in base_model_unweighted.layers:
    layer.trainable = False

# compile model
opt = Adam(lr=0.001)
model_vgg_unweighted.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# redefine generators
train_generator = train_datagen.flow_from_directory(
        '/home/Rebecca/bmi707_assignments/train_data',  # the directory for the training data
        target_size=(250,250),  # resize the input images to accommodate the model
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.# fit unweighted model
nTraining = 1750
nValidation = 437
model_vgg_unweighted.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)
flow_from_directory(
        '/home/Rebecca/bmi707_assignments/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=batch_size,
        class_mode='binary')


# Get AUC
## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model_vgg_unweighted.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.627

# Define weighted model

# get the output of the base model
x = base_model_weighted.output
# get the output of the base model
x = Dense(256,activation='relu')(x)
# pool
x = GlobalAveragePooling2D()(x)
# add a layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# define the model to be trained
model_vgg_weighted = Model(inputs=base_model_weighted.input, outputs=predictions)

# Fix the base model weights
for layer in base_model_weighted.layers:
    layer.trainable = False

# fit weighted model
nTraining = 1750
nValidation = 437
model_vgg.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)

# compile model
opt = Adam(lr=0.001)
model_vgg_weighted.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.627

# fit unweighted model
nTraining = 1750
nValidation = 437
model_vgg_unweighted.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)

# Get AUC
## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model_vgg_unweighted.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.627

# Define weighted model

# get the output of the base model
x = base_model_weighted.output
# get the output of the base model
x = Dense(256,activation='relu')(x)
# pool
x = GlobalAveragePooling2D()(x)
# add a layer for binary classification
predictions = Dense(1, activation='sigmoid')(x)

# define the model to be trained
model_vgg_weighted = Model(inputs=base_model_weighted.input, outputs=predictions)

# Fix the base model weights
for layer in base_model_weighted.layers:
    layer.trainable = False

# fit weighted model
nTraining = 1750
nValidation = 437
model_vgg.fit(train_generator,
        steps_per_epoch=nTraining // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=nValidation // batch_size)

# compile model
opt = Adam(lr=0.001)
model_vgg_weighted.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

## Get predictions
validation_generator = test_datagen.flow_from_directory(
        '/content/val_data', # the directory for the validation data
        target_size=(250,250),
        batch_size=437,
        class_mode='binary',
        shuffle=False)

preds = model.predict(validation_generator,steps=1)
truth = np.concatenate([np.repeat(0,218),np.repeat(1,219)])

## AUC
# Compute ROC curve and ROC area in training set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(truth,np.round(preds[:,0],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

print(roc_auc_val)

# AUC = 0.627