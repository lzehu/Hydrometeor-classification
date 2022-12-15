import os
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score


# load data
np.random.seed(seed=19970312)
dataset1 = np.array(pd.read_csv(r'./testset/DS.csv', header=None))
dataset2 = np.array(pd.read_csv(r'./testset/RA.csv', header=None))
dataset3 = np.array(pd.read_csv(r'./testset/BD.csv', header=None))
dataset4 = np.array(pd.read_csv(r'./testset/HA.csv', header=None))
dataset = np.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)
np.random.shuffle(dataset)
Normalizer().fit_transform(dataset)
x = dataset[:, 0:4]
print(x.shape)
y = dataset[:, -1].astype(int)

# load the binary svm model
been_trained_models_topPath = r'./single_sample_best_models/'
clf_name = os.listdir(been_trained_models_topPath)
clf_set = ['svm_DS_RA.m', 'svm_DS_BD.m', 'svm_DS_HA.m', 'svm_RA_BD.m', 'svm_RA_HA.m', 'svm_BD_HA.m']

# creat the multi classifier based on OVO
labels_numpy = np.zeros((dataset.shape[0], 6))
for model_idx in range(6):
    NN_MM_clf = joblib.load(os.path.join(been_trained_models_topPath, clf_set[model_idx]))
    y_hat = NN_MM_clf.predict(x)
    labels_numpy[:, model_idx] = y_hat

samples_votes_map = np.zeros((dataset.shape[0], 4))
for sample_idx in range(dataset.shape[0]):
    votes_onCls_list = []
    for cls_i in range(4):
        cls_cnt = labels_numpy[sample_idx, :].tolist().count(cls_i)
        votes_onCls_list.append(cls_cnt)
    samples_votes_map[sample_idx, :] = votes_onCls_list

cls_Txt = ['DS', 'RA', 'BD', 'HA']
y_list = np.array(dataset[:, 4]).tolist()
y_predict_list = np.zeros((dataset.shape[0], 1))
# Take the highest vote as the output classification result:
for i in range(dataset_new.shape[0]):
    y = y_list[i]
    sample_votes_onSet_list = samples_votes_map[i, :].tolist()
    y_predict = sample_votes_onSet_list.index(max(sample_votes_onSet_list))
    y_predict_list.append(y_predict)
test_accuracy = accuracy_score(np.array(y_list), np.array(y_predict_list))
print('ACC of OVO_SVM', test_accuracy)



