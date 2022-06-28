#!/usr/bin/env python
# coding: utf-8

# In[1]:


import fasttext.util
import simdjson
import itertools
from sklearn.utils import shuffle
import numpy as np
from math import ceil
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from scipy import sparse
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import string
import fasttext as ft
import fasttext.util
from torch import nn
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger


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


classes = mlb.classes_


# In[5]:


msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, test_index in msss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    break
    
useless = set(itertools.chain(*mlb.inverse_transform(Y_test))) ^ set(itertools.chain(*mlb.inverse_transform(Y_train)))

print(len(useless))

#

positive = (np.sum(Y_train, axis=0) + 1)
negative = Y_train.shape[0] - positive

weights = (negative / positive).astype(np.float32)

# In[6]:


def sentence_stats(p):
    sentence_lengths = []
    sentence_counts = []
    sentence_ratios = []
    sentence_counts_original = []

    for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
        group = list(group)
        
        sentence_counts_original.append(len(group))
        group = group[:max(ceil(p * len(group)), 1)]
        sentence_counts.append(len(group))
        for sentence in group:
            sentence_lengths.append(len(sentence["lemmas"]))

            junk_count = [i for i in sentence["lemmas"] if all(j.isdigit() or j in string.punctuation for j in i)]
            sentence_ratio = len(junk_count) / len(sentence["lemmas"])
            sentence_ratios.append(sentence_ratio)

    sentence_lengths = np.array(sentence_lengths)
    sentence_counts = np.array(sentence_counts)
    sentence_ratios = np.array(sentence_ratios)
    sentence_counts_original = np.array(sentence_counts_original)

    return sentence_lengths, sentence_counts, sentence_ratios, sentence_counts_original


# In[ ]:


def plot_sentence_stats(sentence_lengths, sentence_counts, sentence_ratios, sentence_counts_original):
    sentence_removed_ratio = 1 - sentence_counts / sentence_counts_original
    
    quantile = np.quantile(sentence_lengths, 0.9)
    sentence_lengths = sentence_lengths[sentence_lengths < quantile]

    quantile = np.quantile(sentence_counts, 0.9)
    sentence_counts = sentence_counts[sentence_counts < quantile]

    fig, ax = plt.subplots(nrows=2, ncols=2)

    fig.set_size_inches(20, 15)

    ax[0, 0].bar(*np.unique(sentence_lengths, return_counts=True))
    ax[0, 1].bar(*np.unique(sentence_counts[sentence_counts < quantile], return_counts=True))
    ax[1, 0].hist(sentence_ratios, bins=100)
    ax[1, 1].hist(sentence_removed_ratio, bins=100)

    ax[0, 0].set_ylabel('Count')
    ax[0, 0].set_xlabel('Sentence length');

    ax[0, 1].set_ylabel('Count')
    ax[0, 1].set_xlabel('Sentence count');

    ax[1, 0].set_ylabel('Count')
    ax[1, 0].set_xlabel('Waste ratio');
    
    ax[1, 1].set_ylabel('Count')
    ax[1, 1].set_xlabel('Ratio of removed sentences');


# In[8]:

k = 5
dims = 50

ft_model = ft.load_model('cc.hr.300.bin')

ft_model = fasttext.util.reduce_model(ft_model, dims)


# In[9]:


def generate_embeddings(p):
    embeddings = {}

    for key, group in itertools.groupby(dataset, lambda x: x["document_id"]):
        group = list(group)
        group = group[:p]

        sentences = []
        for sentence in group:
            filtered_indices = [i for i, lemma in enumerate(sentence["lemmas"]) if not all(x.isdigit() or x in string.punctuation for x in lemma)]
            filtered_tokens = [sentence["tokens"][i] for i in filtered_indices]

            if len(filtered_tokens) == 0:
                continue

            sentences.append(filtered_tokens)

        embeddings[key] = [ft_model.get_sentence_vector(" ".join(sentence)) for sentence in sentences]

        if len(embeddings[key]) == 0:
            embeddings[key].append(np.zeros((dims,)))
        
    return embeddings


# In[10]:


class EuroVocDataset(Dataset):
    def __init__(self, embeddings, X, Y):
        self.X = X
        self.Y = Y
        self.embeddings = embeddings
        
    def __getitem__(self, idx):
        ids = self.X[idx]
        labels = self.Y[idx]
        
        return {"input": self.embeddings[ids], "output": labels.astype(np.float32)}
    
    def __len__(self):
        return self.X.shape[0]
    
def collate(datapoints):
    X = []
    Y = []
    for datapoint in datapoints:
        missing = k - len(datapoint["input"])
        
        if missing > 0:
            X.append(np.append(datapoint["input"], np.zeros((missing, dims)), axis=0).astype(np.float32))
        else:
            X.append(np.array(datapoint["input"]).astype(np.float32))

        Y.append(datapoint["output"])

    return {"input": torch.from_numpy(np.stack(X)), "output": torch.from_numpy(np.stack(Y))}


# In[11]:


embeddings = generate_embeddings(k)


# In[12]:


dataset_train = EuroVocDataset(embeddings, X_train, Y_train)
dataset_val = EuroVocDataset(embeddings, X_test, Y_test)

train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True, collate_fn=collate, num_workers=60)
val_loader = DataLoader(dataset_val, batch_size=16, collate_fn=collate, num_workers=60)


# In[31]:


class Classifier(pl.LightningModule):
    def __init__(self, weights):
        super().__init__()
        
        #encoder_layer = nn.TransformerEncoderLayer(d_model=dims, nhead=1)
        #norm_layer = nn.LayerNorm([k, dims])
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=norm_layer)
        #self.hidden = nn.Linear(k * dims, k * dims)
        self.output = nn.Linear(k * dims, len(classes))

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights))

        self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.GELU()
        
    def forward(self, x):
        #x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        #x = self.hidden(x)
        #x = self.activation(x)
        #x = self.dropout(x)

        x = self.output(x)
        #x = self.dropout(x)
        
        return x.float()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch["input"], train_batch["output"]
        
        y_pred = self.forward(x)
        
        loss = self.criterion(y_pred, y)
        
        self.log("train_loss", loss)
        
        return {
            "loss": loss,
            "expected": y,
            "predicted": y_pred
        }
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch["input"], val_batch["output"]
        
        y_pred = self.forward(x)
        
        loss = self.criterion(y_pred, y)
        
        self.log("val_loss", loss)
        
        return {
            "loss": loss,
            "expected": y,
            "predicted": y_pred

        }
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([x["loss"] for x in outputs]).mean()
        
        y_true = np.rint(torch.cat([x["expected"] for x in outputs]).cpu())
        y_pred = np.rint(torch.sigmoid(torch.cat([x["predicted"] for x in outputs])).detach().cpu())
        
        r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        r_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        p_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        self.logger.experiment.add_scalar("Loss/Train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Macro/Train", r_macro, self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Micro/Train", r_micro, self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Macro/Train", p_macro, self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Micro/Train", p_macro, self.current_epoch)
        self.logger.experiment.add_scalar("F1/Macro/Train", f1_macro, self.current_epoch)
        self.logger.experiment.add_scalar("F1/Micro/Train", f1_micro, self.current_epoch)
        

    def validation_epoch_end(self, outputs):
        avg_loss = torch.FloatTensor([x["loss"] for x in outputs]).mean()
        
        y_true = np.rint(torch.cat([x["expected"] for x in outputs]).cpu())
        y_pred = np.rint(torch.sigmoid(torch.cat([x["predicted"] for x in outputs])).detach().cpu())
        
        r_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        r_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        p_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        p_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

        self.logger.experiment.add_scalar("Loss/Valid", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Macro/Valid", r_macro, self.current_epoch)
        self.logger.experiment.add_scalar("Recall/Micro/Valid", r_micro, self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Macro/Valid", p_macro, self.current_epoch)
        self.logger.experiment.add_scalar("Precision/Micro/Valid", p_micro, self.current_epoch)
        self.logger.experiment.add_scalar("F1/Macro/Valid", f1_macro, self.current_epoch)
        self.logger.experiment.add_scalar("F1/Micro/Valid", f1_micro, self.current_epoch)
        
        self.log("avg_val_loss", avg_loss)


# In[32]:


clf = Classifier(weights)


# In[33]:


logger = TensorBoardLogger("tb_logs", name=f"model_top_{k}")
checkpointer = pl.callbacks.ModelCheckpoint(mode="min", monitor="avg_val_loss", dirpath="checkpoints", filename="{epoch}-{avg_val_loss:.2f}", save_on_train_epoch_end = True, every_n_epochs=1)

trainer = pl.Trainer(max_epochs=1000, devices=[0], strategy="dp", logger=logger, accelerator="gpu", auto_select_gpus=False, callbacks=[checkpointer])
trainer.fit(clf, train_loader, val_loader)


# In[ ]:




