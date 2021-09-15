#!/usr/bin/env python
# coding: utf-8

# ### Steam Game  Engine Recommender System Part 1
# 
# The first model of this project analyzes a dataset to predicts whether a randomly chosen user would play a randomly chosen game.

# 1) Import libraries and complete other pre-processing steps that will set up our data to input into a model
# 
# 2) Create neccessary dictionaries used for recommender system algorithm

# In[1]:


# 1

import gzip
from collections import defaultdict
import numpy as np
import pandas as pd
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


import scipy.spatial as sp
import scipy.sparse

li = []
def readJSON(path):
    for l in gzip.open(path, 'rt'):
        d = eval(l)
        u = d['userID']
        try:
            g = d['gameID']
        except Exception as e:
            g = None
        yield u,g,d
        
gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in readJSON("train.json.gz"):
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()


# In[3]:


# 2

c = 0
X_train = []
X_val = []
userGames = defaultdict(set)
gameUsers = defaultdict(set)
userHours = defaultdict(list)
gameHours = defaultdict(list)
received_free = defaultdict(list)
users = set()
games = set()
for user, game, d in readJSON('train.json.gz'):
    if d['hours_transformed'] > 8.5:
        continue
    else:
        userGames[user].add(game)
        gameUsers[game].add(user)
        users.add(user)
        games.add(game)
        userHours[user].append(d['hours_transformed'])
        gameHours[game].append(d['hours_transformed'])
        if 'compensation' in d.keys():
            received_free[game].append(1)
        else:
            received_free[game].append(0)
        c = np.random.choice([1, 0], p=[0.8, 0.2])
        if c == 1:
            X_train.append(d)
        else:
            X_val.append(d)
    
X_train = pd.DataFrame(X_train)
X_train['y'] = np.array([1]*X_train.shape[0])
X_val = pd.DataFrame(X_val)
X_val['y'] = np.array([1]*X_val.shape[0])

def user_didnt_play(uID):
    g = list(userGames[uID])
    g_prime = games.copy()
    for i in g:
        g_prime.remove(i)
    return np.random.choice(list(g_prime))

negative = []
for i in list(X_train['userID']):
    negative.append({'userID': i, 'gameID': user_didnt_play(i), 'y': 0})
neg = pd.DataFrame(negative)
X_train = X_train.append(neg).reset_index(drop=True)

negative = []
for i in list(X_val['userID']):
    negative.append({'userID': i, 'gameID': user_didnt_play(i), 'y': 0})
    
neg = pd.DataFrame(negative)
X_val = X_val.append(neg).reset_index(drop=True)

X_train = X_train.sample(frac=1).reset_index(drop=True)
X_val = X_val.sample(frac=1).reset_index(drop=True)

X_train['hours_transformed'] = X_train['hours_transformed'].fillna(0)
X_val['hours_transformed'] = X_val['hours_transformed'].fillna(0)
X_train['compensation'] = X_train['compensation'].fillna(0)
X_val['compensation'] = X_val['compensation'].fillna(0)


# 3) Define a Similarity function that will be used to asses the similarity between either two users or two games.
# 
# 4) Define a function that can take the data of a specific format and create a feature matrix (2D) that can be used in our model.

# In[4]:


# 3

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/1.5:
        break


# In[5]:


# 4

def make_features(data):
    features = []
    for i in range(data.shape[0]):
        if i % 1000 == 0:
            print('Working on ' + str(i) + ' now...')
        jac_vals = []
        user, game = data['userID'][i], data['gameID'][i]
        gUsers = gameUsers[game]
        uGames = userGames[user]
        
        for x in uGames:
            if x == game:
                continue
            jac_vals.append(Jaccard(gUsers, gameUsers[x]))
                   
        if len(jac_vals) == 0:
            jac_max = 0
        else:
            jac_max = max(jac_vals)
            
            
        pop_orNot = float(game in return1)
        
        if len(userHours[user]) == 0:
            avg_userHours = 0
        else:
            avg_userHours = np.median(userHours[user])
            
        if len(gameHours[game]) == 0:
            avg_gameHours = 0
        else:
            avg_gameHours = np.median(gameHours[game])
        
        if len(received_free[game]) == 0:
            reFree = 0
        else:
            reFree = np.sum(received_free[user])

        features.append([jac_max, pop_orNot, avg_userHours, reFree])
            
    return features


# 5) Make the feature matrix based on our training data
# 
# 6) Use the LogisticRegression model from sklearn library to classify (0 or 1) whether a user would play a game
# 
# 7) Use the model to make predictions on our validation feature matrix.
# 
# 8) Assess the accuracy of our model

# In[6]:


# 5

feats = make_features(X_train)


# In[7]:


# 6

model1 = LogisticRegression(C=10, fit_intercept=True)
model1.fit(feats, X_train['y'])


# In[8]:


# 7

preds = model1.predict(make_features(X_val))


# In[9]:


# 8

np.mean(np.array(preds) == X_val['y'])


# ## Steam Game Engine NLP Part 2
# 
# The second model of this project analyzes the dataset specifically the review of a game and predicts the genre of the game that was reviewed.

# 1) Import libraries and complete other textual pre-processing steps that will set up our data to input into a model
# 
# 2) Obtain the counts of each word to use in our bag-of-words model.

# In[11]:


# 1

import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
from sklearn import linear_model
from scipy.sparse import lil_matrix

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
        
def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        yield u,d
        
data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)

Ntrain = (9*len(data))//10
dataTrain = data[:Ntrain]


# In[ ]:


# 2

sp = set(string.punctuation)
wordCount = defaultdict(int)
for d in data:
    r = ''.join([c for c in d['text'].lower() if not c in sp])
    tokens = r.split()
    for w in tokens:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
counts[:10]


# 3) Define a function that makes a one-hot encoded feature matrix out of our top NW most occuring words
# 
# *Note:* Sparse matrix library used to optimize runtime

# In[ ]:


# 3

NW = 3500
words = [x[1] for x in counts[:NW]]

wordId = dict(zip(words, range(len(words))))
wordSet = set(words)

def feature(datum):
    feat = [0]*len(words)
    r = ''.join([c for c in datum['text'].lower() if not c in sp])
    tokens = r.split()
    for w in tokens:
        if w in wordSet:
            feat[wordId[w]] += 1
                
    feat.append(1)
    feat.append(len(tokens))
    feat.append(float(datum['early_access']))
    return feat

nf = len(feature(data[0]))

#Sparse matrix
X = lil_matrix((len(data), nf))

for i in range(len(data)):
    if not (i % 1000):
        print(i)
    x = feature(data[i])
    for j in range(nf):
        if x[j]:
            X[i,j] = x[j]

y = [d['genreID'] for d in data]


# 4) Create training and validation datasets using a specific Ntrain value
# 
# 5) Use the LogisticRegression model from sklearn library to classify the genre of the game based on our derived feature matrix
# 
# 6) Use the model to make predictions on our validation feature matrix
# 
# 7) Assess the accuracy of the multi-class classifier

# In[ ]:


# 4

Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xvalid = X[Ntrain:]
yvalid = y[Ntrain:]


# In[ ]:


# 5

mod = linear_model.LogisticRegression(C=10, max_iter=10000)
mod.fit(Xtrain, ytrain)


# In[ ]:


# 6
pred = mod.predict(Xvalid)
correct = pred == yvalid
sum(correct) / len(correct)


# Dataset Citations:
# 
# Self-attentive sequential recommendation
# Wang-Cheng Kang, Julian McAuley
# ICDM, 2018
# 
# Item recommendation on monotonic behavior chains
# Mengting Wan, Julian McAuley
# RecSys, 2018
# 
# Generating and personalizing bundle recommendations on Steam
# Apurva Pathak, Kshitiz Gupta, Julian McAuley
# SIGIR, 2017
