# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

import time

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    token = []
    for index, row in movies.iterrows():
        if row['genres'] != '(no genres listed)':
            token.append(tokenize_string(row['genres']))
        else:
            token.append(0)
    movies = movies.assign(tokens=pd.Series(token))
    return movies

def featurize(movies):

    c = Counter()
    vocabulary = defaultdict(lambda: 0)
    
    n = len(movies)
    dictio = defaultdict(lambda: 0)
    
    indptr = [0]
    indices = []
    data = []
    
    lista1 = []
    lista2 = []
    lista3 = []
    lista4 = []
    

    for index, row in movies.iterrows():
        lista = row['tokens']
        if lista != 0:
            for term in lista:
                dictio[term] +=1
                if term not in lista1: 
                    lista1.append(term)
                    c[term] +=  1
            lista1.clear()
        else: 
            dictio['nogenre']= 0
        lista2.append((sorted(dictio.items(), key=lambda x: x[1], reverse = True)[0][1], dict(sorted(dictio.items(), key=lambda x: x[0]))))
        dictio.clear()
        
    lista3 = sorted(list(c))
    for m in lista3:
        vocabulary.setdefault(m, len(vocabulary))
    for x,y in lista2:
        if x != 0:
            for i in y: 
                tfidf = y[i] / (x*np.log10(n/c[i]))
                index = vocabulary[i]
                indices.append(index)
                data.append(tfidf)
            indptr.append(len(indices))      
            csr = csr_matrix((data, indices, indptr),shape=(1,len(vocabulary))) 
            data.clear()
            indices.clear()
            indptr.clear()
            indptr = [0]
        elif x ==0: 
            csr = np.array([[0],[0]])            
        lista4.append(csr)
    s1 = pd.Series(lista4, name='features')
    movies = pd.concat([movies, s1], axis=1)
    return movies , vocabulary

def train_test_split(ratings):
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    if a.shape[1] != 1 and b.shape[1] != 1:
        a1 = a.toarray()
        b1 = b.toarray()
        numerator = np.multiply(a1,b1).sum()
        norm_a = np.linalg.norm(a1)
        norm_b = np.linalg.norm(b1)
        denominator = np.multiply(norm_a,norm_b)
        if numerator != 0 or denominator !=0:
            result = numerator / denominator 
        else: 
            result = -1
    else: 
        result = -1 
    return result 



def make_predictions(movies, ratings_train, ratings_test):
    x = movies['features'].tolist()
    result = np.array([])
    suma = np.array([])
    for index, row in ratings_test.iterrows():
        csr1_value = movies.loc[movies['movieId'] == row['movieId']]['movieId'].index
#        print(csr1_value)
        csr1 = x[csr1_value[0]-1]
#        print(csr1.shape)
        rating_user = ratings_train.loc[ratings_train['userId'] == row['userId']]
        sum_cosine = 0 
        suma_weigh = 0
        for index1, row1 in rating_user.iterrows():
            csr2_value = movies.loc[movies['movieId'] == row1['movieId']]['movieId'].index
#            print(csr1_value)
            csr2 = x[csr2_value[0]-1]
#            print(csr2.shape)
            cosine = cosine_sim(csr1, csr2)
            if cosine > 0:
                suma_weigh += row1['rating']*cosine
                sum_cosine += cosine
        if sum_cosine == 0:
            for index2, row2 in rating_user.iterrows():
                suma = np.append(suma, row2['rating'])
            res = np.mean(suma)
        else: 
            res = suma_weigh/sum_cosine    
        result = np.append(result, res)
        print(result)
    return result

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
