import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import Normalizer
import joblib

# Step1: Load Data
NN = 'DS'
MM = 'WS'
np.random.seed(seed=19970312)
trainDataset1 = np.array(pd.read_csv(r'../dataset/trainDataset/{}_samples.csv'.format(NN), header=None))
trainDataset2 = np.array(pd.read_csv(r'../dataset/trainDataset/{}_samples.csv'.format(MM), header=None))
valDataset1 = np.array(pd.read_csv(r'../dataset/testDataset/{}_samples.csv'.format(NN), header=None))
valDataset2 = np.array(pd.read_csv(r'../dataset/testDataset/{}_samples.csv'.format(MM), header=None))
trainDataset = np.concatenate((trainDataset1, trainDataset2), axis=0)
testDataset = np.concatenate((valDataset1, valDataset2), axis=0)
np.random.shuffle(trainDataset)
np.random.shuffle(testDataset)
X_train = trainDataset[:, 0:4]
X_val = testDataset[:, 0:4]
y_train = trainDataset[:, -1].astype(int)
y_val = testDataset[:, -1].astype(int)

# Step2: Data Preprocessing
Normalizer().fit_transform(X_train)
Normalizer().fit_transform(X_val)

# Step3: Training
# grid research
parameter_candidate = [
                       {'C': [40, 50],
                        'gamma':[0.8, 0.7, 0.6],
                        'kernel': ['rbf']}
                       ]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidate, cv=5, n_jobs=6)
clf.fit(X_train, y_train)
print('the score of best model：', clf.best_score_)
print('best C：', clf.best_estimator_.C)
print('best Kernel：', clf.best_estimator_.kernel)
print('best Gamma：', clf.best_estimator_.gamma)
best_model = clf.best_estimator_
train_score = best_model.score(X_train, y_train)
print("\nACC of trainset=", train_score)
val_score = best_model.score(X_val, y_val)
print("\nACC of valset=", val_score)

# Step4: Save Best Model
joblib.dump(best_model, "./single_sample_best_models/svm_{}_{}.m".format(NN, MM))


