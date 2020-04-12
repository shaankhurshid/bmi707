# import python packages
import numpy as np              # linear algebra
import pandas as pd             # to process the input files
import statistics as stat
from sklearn.linear_model import LogisticRegression
from numpy.random import randint
from numpy.random import seed
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import matplotlib.pyplot as plt # for plotting

%matplotlib inline

# read in the data files; please use these files, instead of the raw files from the Pima Indian Diabetes Database
train_data = pd.read_csv('https://www.dropbox.com/s/n0jjos6faoyqncp/diabetesTrain.csv?dl=1')
test_data = pd.read_csv('https://www.dropbox.com/s/6uu2754iqv4cxei/diabetesTest.csv?dl=1')

# check the data files
train_data.head()

# construct the design matrix for the input features and the outcome vector
features = list(train_data.columns.values)
features.remove('Outcome')
X_train = train_data[features]
y_train = train_data['Outcome']
X_val = test_data[features]
y_val = test_data['Outcome']

X_train_values = X_train.values
X_scaled = preprocessing.scale(X_train_values)
X_train_scaled = pd.DataFrame(X_scaled)

X_val_values = X_val.values
X_scaled = preprocessing.scale(X_val.values)
X_val_scaled = pd.DataFrame(X_scaled)

X_train_scaled = X_train_scaled.assign(Bias = 1)
X_val_scaled = X_val_scaled.assign(Bias = 1)

train_data.groupby('Outcome').size() # 219 with diabetes, 395 without diabetes

print(stat.median(train_data['BMI'])) # Median 32.35 in training set
print(stat.median(train_data.loc[train_data["Outcome"]==1,"BMI"])) # Median 34.3 in training set with DM
print(stat.median(train_data.loc[train_data["Outcome"]==0,"BMI"])) # Median 30.1 in training set without DM
print(stat.median(test_data.loc[test_data["Outcome"]==1,"BMI"])) # Median 33.9 in training set with DM
print(stat.median(test_data.loc[test_data["Outcome"]==0,"BMI"])) # Median 31.6 in training set without DM

regressor = LogisticRegression()
bmi = X_train['BMI'].values.reshape(-1,1)
regressor.fit(bmi,y_train)

print(regressor.coef_)
print(regressor.intercept_)

# Using logistic regression of the form outcome ~ BMI, there is a positive association between BMI and diabetes.
# Specifically, for every 1 point increase in BMI, there is a 1.12 increase in the odds of diabetes (p < 0.01).
# This assumes that the effect of BMI on the log-odds of diabetes is linear.
# This also does not adjust for any potential confounders on the BMI -> diabetes relationship.

# Since the data are cross-sectional (individuals either have diabetes or not), it is difficult to extrapolate these findings
# to the question of incident diabetes (i.e., "risk of developing diabetes"). Furthermore, it is possible
# that the observed association between BMI and diabetes is confounded (i.e., a third factor leads to both
# increased diabetes risk and increased BMI). However, if one assumes the association is not confounded, there is some reason
# to suspect that weight loss could lead to decreased diabetes risk.

# Please complete this function, and make it return the desired result
# X: the design matrix of the input features
# w: weight vector

def forward_pass(X,w):
  pred = 1 / (1 + np.exp(-(np.dot(X,w))))
  return pred

weights = np.array([1.,-1.,1.,-1.,1.,-1.,1.,-1.,0.])
p = forward_pass(X=X_train_scaled,w=weights)

# Please complete this function, and make it return the desired result
# X: the design matrix of the input features
# y: the ground truth of outcome labels (classes)
# p: the class probability outputted by the model
def backward_pass(y,X,p,weights,lr=0.01):
  w_new = weights
  grad_w = pd.DataFrame()

  grad_w[0] = 2*(p - y)*p*(1 - p)*X[:,0]
  grad_w[1] = 2*(p - y)*p*(1 - p)*X[:,1]
  grad_w[2] = 2*(p - y)*p*(1 - p)*X[:,2]
  grad_w[3] = 2*(p - y)*p*(1 - p)*X[:,3]
  grad_w[4] = 2*(p - y)*p*(1 - p)*X[:,4]
  grad_w[5] = 2*(p - y)*p*(1 - p)*X[:,5]
  grad_w[6] = 2*(p - y)*p*(1 - p)*X[:,6]
  grad_w[7] = 2*(p - y)*p*(1 - p)*X[:,7]
  grad_w[8] = 2*(p - y)*p*(1 - p)*X[:,8]

  w_new[0] = w_new[0] - lr*np.mean(grad_w[0])
  w_new[1] = w_new[1] - lr*np.mean(grad_w[1])
  w_new[2] = w_new[2] - lr*np.mean(grad_w[2])
  w_new[3] = w_new[3] - lr*np.mean(grad_w[3])
  w_new[4] = w_new[4] - lr*np.mean(grad_w[4])
  w_new[5] = w_new[5] - lr*np.mean(grad_w[5])
  w_new[6] = w_new[6] - lr*np.mean(grad_w[6])
  w_new[7] = w_new[7] - lr*np.mean(grad_w[7])
  w_new[8] = w_new[8] - lr*np.mean(grad_w[8])

  return(w_new)

# Training loop
def train(X_train,X_val,y_train,y_val,weights,iters,lr):
  w_new = np.copy(weights)
  train_error_tracker = np.empty(1000)
  val_error_tracker = np.empty(1000)

  for i in range(iters):
    pred_train = forward_pass(X=X_train,w=w_new)
    pred_val = forward_pass(X=X_val,w=w_new)
    w_new = backward_pass(y=y_train,X=X_train,p=pred_train,weights=w_new,lr=lr)
    train_error = (y_train - pred_train)**2
    val_error = (y_val - pred_val)**2
    train_error_tracker[i] = np.mean(train_error)
    val_error_tracker[i] = np.mean(val_error)
    acc_train = len(np.where(np.round(pred_train) == y_train)[0])/float(len(y_train))
    acc_val = len(np.where(np.round(pred_val) == y_val)[0])/float(len(y_val))
    print("Training error at iteration " + str(i) + ": " + str(np.mean(train_error))
    + '\t' + "Train Accuracy: " + str(acc_train)
    + '\t' + "Test error: " + str(np.mean(val_error))
    + '\t' + "Test accuracy: " + str(np.mean(acc_val)))
    output = {"train_error_final":np.mean(train_error),"train_acc":acc_train,"val_acc":acc_val,
              "weights":w_new,"val_error_final":np.mean(val_error),"pred_train":pred_train,"pred_val":pred_val,
              "train_error_log":train_error_tracker,"val_error_log":val_error_tracker}
  return(output)

# Initialize weights
weights = np.array([1.,-1.,1.,-1.,1.,-1.,1.,-1.,0.])

# Run training loop
result = train(X_train=X_train_scaled.values,X_val=X_val_scaled.values,y_train=y_train,y_val=y_val,
               iters=1000,lr=0.08,weights=weights)

# Loss plots
plt.figure()
plt.plot(result['train_error_log'],label="training")
plt.plot(result['val_error_log'],label="validation")
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Perceptron loss per epoch')
plt.legend(loc="upper right")
plt.show()

# Compute ROC curve and ROC area in training set
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_train, tpr_train, _ = roc_curve(y_train,np.round(result['pred_train'],decimals=0))
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and ROC area in validation set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(y_val,np.round(result['pred_val'],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

# Performance measures
## Train
print(result['train_error_final'],result['train_acc'],roc_auc_train) # Loss = 0.154, Accuracy = 0.787, AUC = 0.756
## Test
print(result['val_error_final'],result['val_acc'],roc_auc_val) # Loss = 0.184, Accuracy = 0.760, AUC = 0.729

# Plot the ROC
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange',
         lw=lw, label='Train ROC (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_val, tpr_val, color='navy',
         lw=lw, label='Validate ROC (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Modify the backward pass
def backward_pass_penalized(y, X, p, weights, l=0.1, lr=0.01):
    w_new = weights
    grad_w = pd.DataFrame()

    grad_w[0] = 2 * (p - y) * p * (1 - p) * X[:, 0] + l * 2 * w_new[0]
    grad_w[1] = 2 * (p - y) * p * (1 - p) * X[:, 1] + l * 2 * w_new[1]
    grad_w[2] = 2 * (p - y) * p * (1 - p) * X[:, 2] + l * 2 * w_new[2]
    grad_w[3] = 2 * (p - y) * p * (1 - p) * X[:, 3] + l * 2 * w_new[3]
    grad_w[4] = 2 * (p - y) * p * (1 - p) * X[:, 4] + l * 2 * w_new[4]
    grad_w[5] = 2 * (p - y) * p * (1 - p) * X[:, 5] + l * 2 * w_new[5]
    grad_w[6] = 2 * (p - y) * p * (1 - p) * X[:, 6] + l * 2 * w_new[6]
    grad_w[7] = 2 * (p - y) * p * (1 - p) * X[:, 7] + l * 2 * w_new[7]
    grad_w[8] = 2 * (p - y) * p * (1 - p) * X[:, 8] + l * 2 * w_new[8]

    w_new[0] = w_new[0] - lr * np.mean(grad_w[0])
    w_new[1] = w_new[1] - lr * np.mean(grad_w[1])
    w_new[2] = w_new[2] - lr * np.mean(grad_w[2])
    w_new[3] = w_new[3] - lr * np.mean(grad_w[3])
    w_new[4] = w_new[4] - lr * np.mean(grad_w[4])
    w_new[5] = w_new[5] - lr * np.mean(grad_w[5])
    w_new[6] = w_new[6] - lr * np.mean(grad_w[6])
    w_new[7] = w_new[7] - lr * np.mean(grad_w[7])
    w_new[8] = w_new[8] - lr * np.mean(grad_w[8])

    return (w_new)


# Training loop
def train_penalized(X_train, X_val, y_train, y_val, weights, iters, lr, l):
    w_new = np.copy(weights)
    train_error_tracker = np.empty(1000)
    val_error_tracker = np.empty(1000)

    for i in range(iters):
        pred_train = forward_pass(X=X_train, w=w_new)
        pred_val = forward_pass(X=X_val, w=w_new)
        w_new = backward_pass_penalized(y=y_train, X=X_train, p=pred_train, weights=w_new, lr=lr, l=l)
        train_error = (y_train - pred_train) ** 2
        val_error = (y_val - pred_val) ** 2
        train_error_tracker[i] = np.mean(train_error)
        val_error_tracker[i] = np.mean(val_error)
        acc_train = len(np.where(np.round(pred_train) == y_train)[0]) / float(len(y_train))
        acc_val = len(np.where(np.round(pred_val) == y_val)[0]) / float(len(y_val))
        print("Training error at iteration " + str(i) + ": " + str(np.mean(train_error))
              + '\t' + "Train Accuracy: " + str(acc_train)
              + '\t' + "Test error: " + str(np.mean(val_error))
              + '\t' + "Test accuracy: " + str(np.mean(acc_val)))
        output = {"train_error_final": np.mean(train_error), "train_acc": acc_train, "val_acc": acc_val,
                  "weights": w_new, "val_error_final": np.mean(val_error), "pred_train": pred_train,
                  "pred_val": pred_val,
                  "train_error_log": train_error_tracker, "val_error_log": val_error_tracker}
    return (output)


# Initialize weights
weights = np.array([1., -1., 1., -1., 1., -1., 1., -1., 0.])

# Run the training loop
result_penalized = train_penalized(X_train=X_train_scaled.values, X_val=X_val_scaled.values, y_train=y_train,
                                   y_val=y_val,
                                   iters=1000, lr=0.08, l=0.02, weights=weights)

# Loss plots
plt.figure()
plt.plot(result_penalized['train_error_log'],label="training")
plt.plot(result_penalized['val_error_log'],label="validation")
plt.xlabel('Epoch')
plt.ylabel('Loss value')
plt.title('Perceptron loss per epoch')
plt.legend(loc="upper right")
plt.show()

# Compute ROC curve and ROC area in training set
fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()
fpr_train, tpr_train, _ = roc_curve(y_train,np.round(result_penalized['pred_train'],decimals=0))
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and ROC area in validation set
fpr_val = dict()
tpr_val = dict()
roc_auc_val = dict()
fpr_val, tpr_val, _ = roc_curve(y_val,np.round(result_penalized['pred_val'],decimals=0))
roc_auc_val = auc(fpr_val, tpr_val)

# Performance measures
## Train
print(result_penalized['train_error_final'],result_penalized['train_acc'],roc_auc_train) # Loss = 0.158, Accuracy = 0.777, AUC = 0.741
## Test
print(result_penalized['val_error_final'],result_penalized['val_acc'],roc_auc_val) # Loss = 0.174, Accuracy = 0.760, AUC = 0.720

# Slightly lower accuracy, slightly greater loss, and smaller delta between train and validate is consistent with expected behavior of L2 penalization

# Plot the ROC
plt.figure()
lw = 2
plt.plot(fpr_train, tpr_train, color='darkorange',
         lw=lw, label='Train ROC (area = %0.2f)' % roc_auc_train)
plt.plot(fpr_val, tpr_val, color='navy',
         lw=lw, label='Validate ROC (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# compare weights with and without L2
## without
print(max(abs(result['weights']))) # max weight magnitude 1.15
## with
print(max(abs(result_penalized['weights']))) # max weight magnitude 0.61

# Lower maximum weight is consistent with L2 penalty effect

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard

mod = Sequential()
mod.add(Dense(256, activation='relu', input_dim=9))
mod.add(Dropout(rate=0.5))
mod.add(Dense(256, activation='relu'))
mod.add(Dropout(rate=0.5))
mod.add(Dense(256, activation='relu'))
mod.add(Dropout(rate=0.5))
mod.add(Dense(256, activation='relu'))
mod.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.05)
mod.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=('accuracy'))

!rm -Rf logs/
import os
logdir = os.path.join("logs","no_regularization")
tensorboard_callback = TensorBoard(logdir,histogram_freq=1)

mod.fit(X_train_scaled.values,y_train,
        validation_data=(X_val_scaled.values,y_val),
        epochs=50,batch_size=16,
        callbacks=[tensorboard_callback])

%load_ext tensorboard
%tensorboard --logdir logs

# Final metrics: train accuracy 0.7980, train loss 0.4026, val accuracy 0.7727, val loss 0.5328, AUC 0.76

# Calculate AUC in validation set
dl_y_prob = mod.predict(X_val_scaled.values)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_val,np.round(dl_y_prob,decimals=0))
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Your codes and answers here
regressor = LogisticRegression()
X, y = X_train_scaled.iloc[:,0:8], y_train
lr = regressor.fit(X,y)

print(regressor.coef_)
print(regressor.intercept_)

# Obtain predicted probabilities
y_prob = lr.predict_proba(X_val_scaled.iloc[:,0:8])

# Calculate accuracy
lr_acc = len(np.where(np.round(y_prob[:,1]) == y_val)[0])/float(len(y_val))
print(lr_acc)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_val,np.round(y_prob[:,1],decimals=0))
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# LR model accuracy = 0.7532, AUC = 0.71

# When evaluated in the validation set, the neural network (AUC 0.72, accuracy 0.7727) performs very slightly better
# than the logistic regression model (AUC 0.71, accuracy 0.7532). Reasons why performance is similar (as opposed to much better with the neural network),
# could include the following:
# 1) Limited information contained within predictors with regard to outcome. In other words, the predictors may not contain rich enough information
# to effectively predict the outcome beyond the achieved accuracy level of both models.
# 2) Small sample size. A larger sample size (esp. more cases) would likely provide the DL model with more information to learn the classification
# task better.
# 3) No high-level interactions between inputs and the outcome. If each of the input features' relationship with the outcome can be captured effectively as
# a linear relationship on the log-odds scale, then a logistic regression model would already be expected to provide optimal accuracy.