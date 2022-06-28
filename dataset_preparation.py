#!/usr/bin/env python
# coding: utf-8

# In[86]:


import simdjson
import itertools
from itertools import repeat
import numpy as np
from math import ceil
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import string
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
import multiprocessing
import time
import warnings

warnings.filterwarnings("ignore")


# In[2]:


with open("data/dataset_100.json") as f:
    dataset = simdjson.load(f)


# In[3]:


X, Y = [], []

for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
    X.append(key)
    Y.append(next(group)["labels"])

mlb = MultiLabelBinarizer().fit(Y)
X = np.array(X)
Y = mlb.transform(Y)


# In[4]:


msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, test_index in msss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    break

# useless = set(itertools.chain(*mlb.inverse_transform(Y_test))) ^ set(itertools.chain(*mlb.inverse_transform(Y_train)))

indices = np.unique(np.concatenate((np.where(~Y_test.any(axis=0))[0], np.where(~Y_train.any(axis=0))[0])))

print(len(indices))

Y_train = np.delete(Y_train, indices, axis=1)
Y_test = np.delete(Y_test, indices, axis=1)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)
