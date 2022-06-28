#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import repeat
import simdjson
import itertools
import numpy as np
from math import ceil
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import string
import fasttext as ft
import fasttext.util as ft_util
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import multiprocessing
import time


# In[ ]:


with open("data/dataset_100.json") as f:
    dataset = simdjson.load(f)


# In[ ]:


X, Y = [], []

for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
    X.append(key)
    Y.append(next(group)["labels"])

mlb = MultiLabelBinarizer().fit(Y)
X = np.array(X)
Y = mlb.transform(Y)

classes = mlb.classes_


# In[ ]:


msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, test_index in msss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    break
    
useless = set(itertools.chain(*mlb.inverse_transform(Y_test))) ^ set(itertools.chain(*mlb.inverse_transform(Y_train)))

print(len(useless))


# In[ ]:


dims = 300
ft_model = ft.load_model('cc.hr.300.bin')
#ft_model = ft_util.reduce_model(ft_model, dims)


# In[ ]:


def generate_embeddings(k):
    embeddings = {}

    for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
        group = list(group)
        
        sentences = []
        for sentence in group:
            if len(sentences) == k:
                break
                
            filtered_indices = [i for i, lemma in enumerate(sentence["lemmas"]) if not all(x.isdigit() or x in string.punctuation for x in lemma)]
            filtered_tokens = [sentence["tokens"][i] for i in filtered_indices]
            
            if len(filtered_tokens) < 10:
                continue

            sentences.append(filtered_tokens)

        embeddings[key] = [ft_model.get_sentence_vector(" ".join(sentence)) for sentence in sentences]

        for _ in range(k - len(embeddings[key])):
            embeddings[key].append(np.zeros((dims,)))
            
        embeddings[key] = np.stack(embeddings[key]).flatten()

    return embeddings


# In[ ]:


def train_classifier(i, X_train_features, X_test_features, Y_train):
    try:
        clf = SVC(class_weight='balanced').fit(X_train_features, Y_train[:, i])
        return clf.predict(X_test_features)
    except ValueError:
        return np.zeros((Y_test.shape[0],))


# In[ ]:


pool = multiprocessing.Pool(60)

scores = []
for i in range(1, 16):
    embeddings = generate_embeddings(i)
    X_train_features = np.stack([embeddings[x] for x in X_train])
    X_test_features = np.stack([embeddings[x] for x in X_test])

    scaler = StandardScaler().fit(X_train_features)

    X_train_features = scaler.transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    start = time.perf_counter()
    Y_test_pred = np.stack(pool.starmap(train_classifier, zip(range(Y_test.shape[1]), repeat(X_train_features), repeat(X_test_features), repeat(Y_train))), axis=-1)
    
    f1_macro = f1_score(Y_test, Y_test_pred, average='macro', zero_division=0)
    f1_micro = f1_score(Y_test, Y_test_pred, average='micro', zero_division=0)

    precision_macro = precision_score(Y_test, Y_test_pred, average='macro', zero_division=0)
    precision_micro = precision_score(Y_test, Y_test_pred, average='micro', zero_division=0)

    recall_macro = recall_score(Y_test, Y_test_pred, average='macro', zero_division=0)
    recall_micro = recall_score(Y_test, Y_test_pred, average='micro', zero_division=0)

    scores.append((f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro, time.perf_counter() - start))
    
    print(scores[-1])


# In[ ]:




