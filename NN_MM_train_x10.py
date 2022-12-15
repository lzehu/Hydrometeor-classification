import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import Normalizer
import joblib


# Step1: Load Data
JOINT_NUM = 10
np.random.seed(seed=19970312)
trainDataset1 = np.array(pd.read_csv(r'./upDimension_dataset/train_dataset/upDimension_DS_trainDataset10.csv', header=None))
trainDataset2 = np.array(pd.read_csv(r'./upDimension_dataset/train_dataset/upDimension_RA_trainDataset10.csv', header=None))
valDataset1 = np.array(pd.read_csv(r'./upDimension_dataset/test_dataset/upDimension_DS_testDataset10.csv', header=None))
valDataset2 = np.array(pd.read_csv(r'./upDimension_dataset/test_dataset/upDimension_RA_testDataset10.csv', header=None))
trainDataset = np.concatenate((trainDataset1, trainDataset2), axis=0)
np.random.shuffle(trainDataset)
testDataset = np.concatenate((valDataset1, valDataset2), axis=0)
np.random.shuffle(testDataset)
X_train = trainDataset[:, 0:4*JOINT_NUM]
y_train = trainDataset[:, -1].astype(int)
X_test = testDataset[:, 0:4*JOINT_NUM]
y_test = testDataset[:, -1].astype(int)

# Step2: Data Preprocessing
Normalizer().fit_transform(X_train)
Normalizer().fit_transform(X_test)

# Step3: Training
# grid research
parameter_candidate = [
                       {'C': [300, 400, 500],
                        'degree': [1, 2, 3],
                        'kernel': ['poly']}
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
joblib.dump(best_model, './multi_sample_best_models/DS_RA_upDimension_x10.m')