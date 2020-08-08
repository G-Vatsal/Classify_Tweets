# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:42:28 2020

@author: LENOVO
"""
import nltk
import re #regular expression
import pandas as pd
#from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
#stop_words= set(stopwords.words('english'))
f=pd.read_csv('train.csv', encoding='ISO-8859-1')
#X=pd.read_csv('test.csv',encoding='ISO-8859-1')['text']
#samples=pd.read_csv('sample_submission.csv')
X=f['text']
y=f['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
for i in range(len(X_train)):
    X_train[i] = X_train[i].lower()
    X_train[i] = re.sub(r'\W'," ", X_train[i]) 
    X_train[i] = re.sub(r'\s+'," ", X_train[i])
    words=nltk.word_tokenize(X_train[i])    
    '''words = [w for w in words if not w in stop_words]
    X_train[i] = ' '.join([str(elem) for elem in words]) '''
    
for i in range(len(X_test)): 
    X_test[i] = X_test[i].lower() 
    X_test[i] = re.sub(r'\W'," ", X_test[i]) 
    X_test[i] = re.sub(r'\s+'," ", X_test[i])
    words=nltk.word_tokenize(X_test[i])    
    '''words = [w for w in words if not w in stop_words]
    X_test[i] = ' '.join([str(elem) for elem in words]) '''
'''uniqueword={}
for sent in X_test:
    word=nltk.word_tokenize(sent)
    for var in word:
        if var not in uniqueword:
            uniqueword[var]=1
        else:
            uniqueword[var]+=1

import numpy as np
import operator
listofdict=sorted(uniqueword.items(),key=operator.itemgetter(1),reverse=True)
listofdict=listofdict[:1000]
listofdicti=[i[0] for i in listofdict]
X_t=[]
for sent in X_test:
    word=nltk.word_tokenize(sent)
    vector=np.zeros(len(listofdict))
    for var in word:
        if var in listofdicti:
            vector[listofdict.index(var)]+=1
    X_t.append(vector)
X_t=np.asarray(X_t)

X_train=[]
for sent in dataset:
    word=nltk.word_tokenize(sent)
    vector=np.zeros(len(listofdict))
    for var in word:
        if var in listofdicti:
            vector[listofdict.index(var)]+=1
    X_train.append(vector)
X_train=np.asarray(X_train)

meanarr=np.mean(X,axis=0)
stdarr=np.std(X,axis=0)
confditvmin=meanarr-(1.96*stdarr/np.sqrt(len(meanarr)))
confditvmax=meanarr+(1.96*stdarr/np.sqrt(len(meanarr)))
maxword=listofdict[np.argmax(confditvmax)]
minword=listofdict[np.argmin(confditvmin)]
confditvmax=list(np.around(confditvmax,decimals=0))
confditvmin=list(np.around(confditvmin,decimals=0))
confd=[]
for i in range(len(confditvmin)):
    confd.append((listofdict[i],int(confditvmin[i]),int(confditvmax[i])))
confd=np.asarray(confd)
confd=confd[confd[:,2].argsort()[::-1]]'''
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X_train = matrix.fit_transform(X_train).toarray()
X_test=matrix.transform(X_test).toarray()


#y_train=f['target']
#from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
#samples['target']=Y_pred
#samples.to_csv('final.csv',index=False)



