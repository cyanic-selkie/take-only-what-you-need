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

with open("data/dataset_100.json") as f:
    dataset = simdjson.load(f)

with open("X_train.npy", "rb") as f:
    X_train = np.load(f)
with open("Y_train.npy", "rb") as f:
    Y_train = np.load(f)
with open("X_test.npy", "rb") as f:
    X_test = np.load(f)
with open("Y_test.npy", "rb") as f:
    Y_test = np.load(f)

print("Loaded dataset.")


# In[73]:


def generate_features(X_train, X_test):
    stopwords = set(["a","ako","ali","bi","bih","bila","bili","bilo","bio","bismo","biste","biti","bumo","da","do","duž","ga","hoće","hoćemo","hoćete","hoćeš","hoću","i","iako","ih","ili","iz","ja","je","jedna","jedne","jedno","jer","jesam","jesi","jesmo","jest","jeste","jesu","jim","joj","još","ju","kada","kako","kao","koja","koje","koji","kojima","koju","kroz","li","me","mene","meni","mi","mimo","moj","moja","moje","mu","na","nad","nakon","nam","nama","nas","naš","naša","naše","našeg","ne","nego","neka","neki","nekog","neku","nema","netko","neće","nećemo","nećete","nećeš","neću","nešto","ni","nije","nikoga","nikoje","nikoju","nisam","nisi","nismo","niste","nisu","njega","njegov","njegova","njegovo","njemu","njezin","njezina","njezino","njih","njihov","njihova","njihovo","njim","njima","njoj","nju","no","o","od","odmah","on","ona","oni","ono","ova","pa","pak","po","pod","pored","prije","s","sa","sam","samo","se","sebe","sebi","si","smo","ste","su","sve","svi","svog","svoj","svoja","svoje","svom","ta","tada","taj","tako","te","tebe","tebi","ti","to","toj","tome","tu","tvoj","tvoja","tvoje","u","uz","vam","vama","vas","vaš","vaša","vaše","već","vi","vrlo","za","zar","će","ćemo","ćete","ćeš","ću","što"])

    documents = {}

    for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
        group = list(group)
        
        sentences = []
        for sentence in group:
            #if len(sentences) == k:
            #    break
                
            filtered_indices = [i for i, lemma in enumerate(sentence["lemmas"]) if not all(x.isdigit() or x in string.punctuation for x in lemma) and not lemma in stopwords]
            filtered_tokens = [sentence["lemmas"][i] for i in filtered_indices]
            
            if len(filtered_tokens) < 5:
                continue

            sentences.append(" ".join(filtered_tokens))

        if len(sentences) == 0:
            sentences.append("UNK")

        documents[key] = " ".join(sentences)

            
    X_train_documents = []
    X_test_documents = []
    
    for x in X_train:
        X_train_documents.append(documents[x])
    
    for x in X_test:
        X_test_documents.append(documents[x])
        
    vectorizer = TfidfVectorizer()
    X_train_features = vectorizer.fit_transform(X_train_documents)
    X_test_features = vectorizer.transform(X_test_documents)
    
    return X_train_features, X_test_features


# In[99]:


def train_classifier(i, X_train_features, Y_train, X_test_features, Y_test):
    params = {
        "C" : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
    }
    
    best_C = None
    best_f1 = None
    best_predictions = None
    try:
        for C in params["C"]:
            clf = LinearSVC(class_weight='balanced', max_iter=20000, C=C).fit(X_train_features, Y_train[:, i])
            
            Y_test_pred = clf.predict(X_test_features)
            
            f1 = f1_score(Y_test[:, i], Y_test_pred, zero_division=0)
            
            if best_C is None or best_f1 < f1:
                best_f1 = f1
                best_C = C
                best_predictions = Y_test_pred
    except ValueError:
        best_predictions = np.zeros((Y_test.shape[0],))
    
    return best_predictions, best_C


# In[ ]:


pool = multiprocessing.Pool(60)

scores = []
X_train_features, X_test_features = generate_features(X_train, X_test)

start = time.perf_counter()

predictions, C = list(zip(*pool.starmap(train_classifier, zip(range(Y_test.shape[1]), repeat(X_train_features), repeat(Y_train), repeat(X_test_features), repeat(Y_test)))))
Y_test_pred = np.stack(predictions, axis=-1)
f1_macro = f1_score(Y_test, Y_test_pred, average='macro', zero_division=0)
f1_micro = f1_score(Y_test, Y_test_pred, average='micro', zero_division=0)

precision_macro = precision_score(Y_test, Y_test_pred, average='macro', zero_division=0)
precision_micro = precision_score(Y_test, Y_test_pred, average='micro', zero_division=0)

recall_macro = recall_score(Y_test, Y_test_pred, average='macro', zero_division=0)
recall_micro = recall_score(Y_test, Y_test_pred, average='micro', zero_division=0)

scores.append((f1_macro, f1_micro, precision_macro, precision_micro, recall_macro, recall_micro, time.perf_counter() - start))

print(scores[-1])
