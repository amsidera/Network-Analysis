# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def download_data():
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    doc = doc.lower()
    if keep_internal_punct == False: 
        doc = re.sub('[\W]+', ' ', doc).lower().split()
    elif  keep_internal_punct == True:
        doc = ' ' + doc + ' '
        doc =  re.sub('[\t|\n| ]*\W*[\t|\n| ]*\W*[\t|\n| ]', ' ', doc)
        doc =  re.sub(' [\t|\n |<|>|-|?|!|{|}|+|*|@|¢|$|%|/|(|)|[|\]|:|*|;]*\W+|\W+[\t|\n |<|>|-|?|!|{|}|!|?|+|*|@|¢|$|%|/|(|)|[|\]|:|;]* ', ' ', doc).lower().split()
    return np.array(doc)

def token_features(tokens, feats):
    for d in tokens: 
        feats['token='+d] = feats['token='+d] + 1

def token_pair_features(tokens, feats, k=3):
    if ((len(tokens)+1)%k) == 0:
        a = ((len(tokens)+1)//k) + 1
    else: 
        a = ((len(tokens)+1)//k) + 2
    for i in range(0,a):
        flag = 0 
        for x in combinations(tokens, k):
            flag = 1 
            break
        if flag == 1: 
            for t in combinations(x, 2):
                feats['token_pair='+t[0]+'_'+t[1]] = feats['token_pair='+t[0]+'_'+t[1]] + 1
        tokens = np.delete(tokens,0,None)

def lexicon_features(tokens, feats):
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for d in tokens:
        d = d.lower()
        if d in neg_words:           
            feats['neg_words'] += 1
        if d in pos_words: 
            feats['pos_words'] += 1
    

def featurize(tokens, feature_fns):
    lista = []
    feats = defaultdict(lambda: 0)
    feature = feature_fns
    for i in feature: 
        if i == lexicon_features:
            lista.append(('lexicon_features',i))
        if i == token_features:
            lista.append(('token_features',i))
        if i == token_pair_features:
            lista.append(('token_pair_features',i))
    lista = sorted(lista, key=lambda x: x[0], reverse=True)
    lis = list(lista)
    while lis: 
        x , y = lis.pop()
        if x != 'token_pair_features':
            y(tokens, feats)
        else: 
            y(tokens, feats, 3)
    return feats

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    indptr = [0]
    indices = []
    data = []
    dictio = defaultdict(lambda: 0)
    lista3 = []
    vocabulary = {} 
    for d in tokens_list:
        a = featurize(d, feature_fns)
        lista3.append(a)
        for term,key in a.items():
            if a[term]!=0:            
                dictio[term] = dictio[term] + 1
    dictio = dict(sorted(dictio.items(), key=lambda x: x[0]))
    if vocab == None:
        for key, value in dictio.items():
            if value >= min_freq:
                vocabulary.setdefault(key, len(vocabulary))
    else: 
        vocabulary = dict(vocab)
    while lista3:
        terms = lista3.pop() 
        for term in terms:
            if term in vocabulary: 
                index = vocabulary[term]
                indices.append(index)
                data.append(terms[term])
        indptr.append(len(indices))
    if vocab != None:
        notinvocab = set(indices)^set(vocab.values())
        for i in notinvocab: 
            index = i
            indices.append(index)
            data.append(0)
        indptr[-1] = indptr[-1] + len(notinvocab)
    x = csr_matrix((data, indices, indptr), dtype=np.int64) 
    return x, vocabulary

def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), n_folds=k, shuffle=False, random_state=None)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    dictio = {}
    result = []
    model = LogisticRegression()
    for z in chain(punct_vals):
        j = [tokenize(d,z) for d in docs]
        for i in range(1,len(feature_fns)+1):
            for x in combinations(feature_fns, i):
                for y in chain(min_freqs):
                    matr, vocabulary = vectorize(j, x, y, vocab=None)
                    accuracies = cross_validation_accuracy(model, matr, labels, 5)
                    dictio.setdefault('features', x)
                    dictio.setdefault('punct', z)
                    dictio.setdefault('accuracy', accuracies)
                    dictio.setdefault('min_freq', y)
                    result.append(dictio.copy())
                    dictio.clear()
    result = sorted(result, key=lambda x: x['accuracy'], reverse = True)
    return result 

def plot_sorted_accuracies(results): 
    results = sorted(results, key=lambda x: x['accuracy'])
    m =  [i['accuracy'] for i in results]
    t = range(0,len(m))
    fig = plt.figure()
    plt.plot(t, m)
    plt.ylabel('accuracy')
    plt.xlabel('setting')
    fig.savefig('accuracies.png')

def mean_accuracy_per_setting(results):
    lista = defaultdict(lambda: 0)
    c = Counter()
    resu = []
    lista2 = []
    for d1 in results:
        for d,t in d1.items():
            if d == 'punct':
                if t == False: 
                    lista['punct=False']= lista['punct=False'] + d1['accuracy']
                    c['punct=False'] += 1
                else: 
                    lista['punct=True']= lista['punct=True'] + d1['accuracy']
                    c['punct=True'] += 1
            if d == 'features':
                m = list(t)
                for x in m: 
                    lista2.append(x.__name__)
                lis = " ".join(lista2)
                lista['features='+str(lis)] = lista['features='+str(lis)] + d1['accuracy']
                c['features='+str(lis)] += 1
                lista2.clear()
            if d == 'min_freq':
                lista['min_freq='+str(t)]= lista['min_freq='+str(t)] + d1['accuracy']
                c['min_freq='+str(t)] += 1
    for k,v in lista.items():
        resu.append((v/c[k],k))
    resu = sorted(resu, key=lambda x: x[0], reverse = True)
    return resu                


def fit_best_classifier(docs, labels, best_result):
    j = [tokenize(d,best_result['punct']) for d in docs]
    matr, vocabulary = vectorize(j, best_result['features'], best_result['min_freq'] , vocab=None)
    model = LogisticRegression()
    model.fit(matr,labels)
    return model, vocabulary

def top_coefs(clf, label, n, vocab):
    lista = []
    coef = clf.coef_[0]
    if label == 0: 
        top_coef_ind = np.argsort(coef)[:n]
    else:
        top_coef_ind = np.argsort(coef)[::-1][:n]
    for i in top_coef_ind:
        for k,v in vocab.items():
            if v == i: 
                lista.append((k, np.absolute(coef[v])))
    lista = sorted(lista, key=lambda x: x[1], reverse = True)
    return lista

def parse_test_data(best_result, vocab):
    docs, labels = read_data(os.path.join('data', 'test'))
    test_doc = []
    test_labels = []
    for d in docs: 
        test_doc.append(d)
    for l in labels:
        test_labels.append(l)
    j = [tokenize(d,best_result['punct']) for d in docs]
    matr, vocabulary = vectorize(j, best_result['features'], best_result['min_freq'] , vocab)
    return test_doc, test_labels, matr 

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    result1 = []
    pred_pro = clf.predict_proba(X_test)
    for i in range(0,len(test_labels)): 
        if test_labels[i]==0 and np.subtract(pred_pro[i][0], pred_pro[i][1])<0:
            result1.append((test_docs[i], test_labels[i],1, pred_pro[i][1], pred_pro[i] ))#1
        elif test_labels[i]==1 and np.subtract(pred_pro[i][0], pred_pro[i][1])>0:
            result1.append((test_docs[i], test_labels[i],0,pred_pro[i][0], pred_pro[i] ))#0
    result1 = sorted(result1, key=lambda x: x[3], reverse = True)
    for i in result1[:n]:
        print("truth=%s predicted=%s proba=%.6f\n%s\n" % (str(i[1]),str(i[2]),i[3],i[0]))


def main():
    print(tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False))
    doc = "Isn't love?"
    print(tokenize(doc))
#    feature_fns = [token_features, token_pair_features, lexicon_features]
#    # Download and read data.
#    download_data()
#    docs, labels = read_data(os.path.join('data', 'train'))
#    # Evaluate accuracy of many combinations
#    # of tokenization/featurization.
#    results = eval_all_combinations(docs, labels,
#                                    [True, False],
#                                    feature_fns,
#                                    [2,5,10])
#    # Print information about these results.
#    best_result = results[0]
#    worst_result = results[-1]
#    print('best cross-validation result:\n%s' % str(best_result))
#    print('worst cross-validation result:\n%s' % str(worst_result))
#    plot_sorted_accuracies(results)
#    print('\nMean Accuracies per Setting:')
#    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
#
#    # Fit best classifier.
#    clf, vocab = fit_best_classifier(docs, labels, results[0])
#
#    # Print top coefficients per class.
#    print('\nTOP COEFFICIENTS PER CLASS:')
#    print('negative words:')
#    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
#    print('\npositive words:')
#    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
#
#    # Parse test data
#    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
#
#    # Evaluate on test set.
#    predictions = clf.predict(X_test)
#    print('testing accuracy=%f' %
#          accuracy_score(test_labels, predictions))
#
#    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
#    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    
if __name__ == '__main__':
    main()
