#!/usr/bin/env python
# coding: utf-8

# In[50]:


import os
import numpy as np
from math import log
import nltk.sentiment.vader as vd
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression


# In[63]:


data_x = []
data_y = []
for filename in os.listdir('./train_file/'):
    data_y.append(int(filename.split("_")[1].split(".")[0]))
    f = open('./train_file/' + filename, "r", encoding="utf-8")
    text = f.read()
    data_x.append(list(analyzer.polarity_scores(text).values()))
print("Read data done")


# In[64]:


x_train, x_test, y_train, y_test = ms.train_test_split(np.array(data_x),np.array(data_y),test_size=0.2)
x_train


# In[65]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[66]:


model.score(x_test,y_test)


# In[71]:


fold = ms.KFold(n_splits=1000)

best_x_train = None
best_y_train = None
model_k = LogisticRegression()
best_acc = 0
for train_index, val_index in fold.split(x_train):
    x_tr, x_val = x_train[train_index], x_train[val_index]
    print(x_tr, x_val)
    y_tr, y_val = y_train[train_index], y_train[val_index]
    print(y_tr, y_val)
    model_k.fit(x_tr, y_tr)
    acc = model_k.score(x_val,y_val)
    if acc > best_acc:
        best_x_train = x_tr
        best_y_train = y_tr

model_k.fit(best_x_train, best_y_train)
model_k.score(x_test,y_test)     


# In[ ]:




