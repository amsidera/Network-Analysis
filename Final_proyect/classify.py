# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:27:57 2017

@author: AnaMaria
"""
from collections import Counter, defaultdict
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filename = "train.csv"
filename1 = "test.csv"

def get_contexts(tweet, i, window):
    features = []
    for j in range(np.amax([0, i-window]), i):
        features.append(tweet[j] + "@" + str(j-i))
    for j in range(i+1, min(i + window + 1, len(tweet))):
        features.append(tweet[j] + "@" + str(j-i))
    return features

def tokenize(lista):
    lista = re.sub('[\[|\]\'|,]', ' ', lista)
    lista = lista.split()
    return lista 


def tokenize_time(time):
    time = re.sub('[\[|\]\'|,]', ' ', time)
    time = time.split()
    time = int(time[0])*3600 + int(time[1])*60
    return time

def context_val(words):
    contexts = defaultdict(lambda: Counter())
    window = 2
    for tweet in words:
        for i, token in enumerate(tweet):
            features = get_contexts(tweet, i, window)
            contexts[token].update(features)
    tweet_freq = Counter()
    for context in contexts.values():
        tweet_freq.update(context)
    for term, context in contexts.items():
        for term2, frequency in context.items():
            context[term2] = frequency / (1. + math.log(tweet_freq[term2]))
        length = math.sqrt(sum([v*v for v in context.values()]))
        for term2, frequency in context.items():
            context[term2] = 1. * frequency / length
    return contexts

def cluster_tweet(words):
    contexts = defaultdict(lambda: Counter())      
    contexts = context_val(words)
    vec = DictVectorizer()
    X = vec.fit_transform(contexts.values()) 
    num_cluster_options = [5,10,20,50,100]
    scores = []
    for num_clusters in num_cluster_options:
        kmeans = KMeans(num_clusters, n_init=10, max_iter=10)
        kmeans.fit(X)
        score = -1 * kmeans.score(X)
        scores.append(score)
    num_clusters = 7
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    return kmeans

def context_new_tweet(tweet):
    contexts = defaultdict(lambda: Counter())
    window = 2
    for i, token in enumerate(tweet):
        features = get_contexts(tweet, i, window)
        contexts[token].update(features)
    tweet_freq = Counter()
    for context in contexts.values():
        tweet_freq.update(context)
    for term, context in contexts.items():
        for term2, frequency in context.items():
            context[term2] = frequency / (1. + math.log(tweet_freq[term2]))
        length = math.sqrt(sum([v*v for v in context.values()]))
        for term2, frequency in context.items():
            context[term2] = 1. * frequency / length
    vec = DictVectorizer()
    X = vec.fit_transform(contexts.values())
    return X

def featurize(lista, kmeans):
    c = defaultdict(lambda: Counter())
    lista4 = []
    lista2 = np.array([])
    for i,j,k in lista:
        if len(i) > 8:
            X = context_new_tweet(i)
            result = kmeans.fit_predict(X)
            c = dict(Counter(result))
            c[len(c)] = k
            lista4.append(c)
            if j > 0: 
                lista2 = np.append(lista2, 1)
            else: 
                lista2 = np.append(lista2, 0)
    
    vec = DictVectorizer()
    X = vec.fit_transform(lista4)
    return X, lista2


def print_results(results):
    print('test accuracy=%.4f train accuracy=%.4f' %results)
    
    
def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)
    
def do_cross_validation(X, y, n_folds=5, c=1, penalty='l2', verbose=False):
    cv = KFold(n_splits=n_folds, shuffle=False)
    accuracies = []
    train_accuracies = []
    for foldi, (train, test) in enumerate(cv.split(X)):
        clf = LogisticRegression(random_state=42, C=c, penalty=penalty)
        clf.fit(X[train], y[train])
        train_accuracies.append(accuracy_score(clf.predict(X[train]), y[train]))
        pred = clf.predict(X[test])
        acc = accuracy_score(pred, y[test])
        accuracies.append(acc)
    return (np.mean(accuracies),
            
            np.mean(train_accuracies)) , clf   
    
def main():
    train = pd.read_csv(filename)
    train_words = []
    train_lista = []
    positive = 0 
    negative= 0
    for index, row in train.iterrows():
        if float(row['Stock_actual'])-float(row['Stock_value']) > 0 and positive<200:
            positive+=1
            train_words.append(tokenize(row['Tweet']))
            train_lista.append((tokenize(row['Tweet']), float(row['Stock_actual'])-float(row['Stock_value']),tokenize_time(row['Time_tweet'])))
            ejemplo1 = (tokenize(row['Tweet']), float(row['Stock_actual'])-float(row['Stock_value']),tokenize_time(row['Time_tweet']))
        if float(row['Stock_actual'])-float(row['Stock_value']) <= 0: 
            negative+=1
            train_words.append(tokenize(row['Tweet']))
            train_lista.append((tokenize(row['Tweet']), float(row['Stock_actual'])-float(row['Stock_value']),tokenize_time(row['Time_tweet'])))
            ejemplo2 = (tokenize(row['Tweet']), float(row['Stock_actual'])-float(row['Stock_value']),tokenize_time(row['Time_tweet']))
    kmeans = cluster_tweet(train_words)
    X1, label = featurize(train_lista,kmeans)
    results, model = do_cross_validation(X1, label, verbose=True)
    
    
    test = pd.read_csv(filename1)
    test_lista = []
    lista2 = []
    for index, row in test.iterrows():
        test_lista.append((tokenize(row['Tweet']),tokenize_time(row['Time_tweet'])))
    for i,j in test_lista: 
        if len(i)> 13:
            X = context_new_tweet(i)
            if X.shape[0]>9:
                result = kmeans.fit_predict(X)
                c = dict(Counter(result))
                c[len(c)]=j
                vec = DictVectorizer()
                X2 = vec.fit_transform(c)
                lista2.append(X2)

    for i in range(0,len(lista2)):
        result = model.predict(lista2[i])
    return positive, negative, ejemplo1, ejemplo2
    
    

    
if __name__ == '__main__':
    main()