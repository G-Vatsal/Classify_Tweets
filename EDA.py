# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 08:45:09 2020

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:42:28 2020

@author: LENOVO
"""
import nltk
import re #regular expression
import pandas as pd
from nltk.corpus import stopwords
from spellchecker import SpellChecker
spell=SpellChecker()
from sklearn.model_selection import train_test_split
stop_words= set(stopwords.words('english'))
f=pd.read_csv('train.csv', encoding='ISO-8859-1')
X=f['text']
y=f['target']
for i in range(len(X)):
    X[i] = X[i].lower()
    X[i] = re.sub(r'\W'," ", X[i]) 
    X[i] = re.sub(r'\s+'," ", X[i])
    words=nltk.word_tokenize(X[i])
    words = [w for w in words if not w in stop_words]
    X[i] = ' '.join([str(elem) for elem in words])
uniqueword={}
for sent in X:
    word=nltk.word_tokenize(sent)
    for var in word:
        if var not in uniqueword:     
            uniqueword[var]=1
        else:
            uniqueword[var]+=1

import numpy as np
import operator
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

listofdict=sorted(uniqueword.items(),key=operator.itemgetter(1),reverse=True)
listofdict=listofdict[:2000]
listofdicti=[i[0] for i in listofdict]
X_tr=[]
for sent in X_train:
    word=nltk.word_tokenize(sent)
    vector=np.zeros(len(listofdict))
    for var in word:
        if var in listofdicti:
            vector[listofdicti.index(var)]+=1
    X_tr.append(vector)
X_tr=np.asarray(X_tr)
'''from sklearn.feature_extraction.text import TfidfVectorizer
#matrix = CountVectorizer(max_features=2000)
matrix = TfidfVectorizer()
X_tr = matrix.fit_transform(X_train).toarray()
X_te=matrix.transform(X_test).toarray()'''
X_te=[]
for sent in X_test:
    word=nltk.word_tokenize(sent)
    vector=np.zeros(len(listofdict))
    for var in word:
        if var in listofdicti:
            vector[listofdicti.index(var)]+=1
    X_te.append(vector)
X_te=np.asarray(X_te)
#y_train=f['target']
#from sklearn import linear_model
import sklearn.naive_bayes as nb
import sklearn.linear_model as lm
classifier = nb.GaussianNB()
classifier2 = nb.MultinomialNB()
classifier3 = nb.BernoulliNB()
#classifierl = lm.LinearRegression()

classifier.fit(X_tr, y_train)
classifier2.fit(X_tr,y_train)
classifier3.fit(X_tr,y_train)
#classifierl.fit(X_tr,y_train)

y_pred1=classifier.predict(X_te)
y_pred2=classifier2.predict(X_te)
y_pred3=classifier3.predict(X_te)
y_predavg=y_pred1+y_pred2+y_pred3
y_predw=(0.25*y_pred1)+(0.35*y_pred2)+(0.40*y_pred3)
#y_predl=classifierl.predict(X_te)
'''for i in range(len(y_predl)):
    if y_predl[i]>=0.6:
        y_predl[i]=1
    else:
        y_predl[i]=0'''
for i in range(len(y_predavg)):
    if y_predavg[i]>=2:
        y_predavg[i]=1
    else:
        y_predavg[i]=0
for i in range(len(y_predw)):
    if y_predw[i]>0.6:
        y_predw[i]=1
    else:
        y_predw[i]=0
from sklearn.metrics import accuracy_score
accuracy0= accuracy_score(y_test, y_predavg)*100
accuracyw=accuracy_score(y_test, y_predw)*100
accuracy = accuracy_score(y_test, y_pred1)*100
accuracy2 = accuracy_score(y_test, y_pred2)*100
accuracy3 = accuracy_score(y_test, y_pred3)*100
#accuracyl=accuracy_score(y_test,y_predl)*100
#samples['target']=Y_pred
#samples.to_csv('final.csv',index=False)



