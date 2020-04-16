# BMI 707 exercise 1 script

# import the required packages
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# generate simulated data
n_train = 2000
n_test = 500
n_classes = 2
X, y = make_classification(n_samples=n_train+n_test, n_features=10,
                           n_classes=n_classes,
                           n_informative=2, n_redundant=0,
                           random_state=1, shuffle=True)
X_train = X[0:n_train,:]
y_train = y[0:n_train]
X_test = X[n_train:(n_train+n_test),:]
y_test = y[n_train:(n_train+n_test)]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

########## MODEL 1: Logistic Regression
# note that only the training data X_train and y_train should be used in the training process
clf_lr = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

# get prediction
y_predicted = clf_lr.predict(X_test)

# get predicted probabilities
y_prob = clf_lr.predict_proba(X_test)

# display the confusion matrix
print(confusion_matrix(y_test,y_predicted))
tn, fp, fn, tp = confusion_matrix(y_test,y_predicted).ravel()
print(tn,fp,fn,tp)

# get LR coefficients
print(clf_lr.coef_)
print(clf_lr.intercept_)

########## MODEL 2: Random Forest
clf_rf = RandomForestClassifier().fit(X_train,y_train)

# get prediction
y_predicted = clf_rf.predict(X_test)

# get predicted probability
y_score = clf_rf.predict_proba(X_test)

# generate the confusion matrix
print(confusion_matrix(y_test,y_predicted))

# feature importance
print(clf_rf.feature_importances_)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test,y_score[:,1])
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