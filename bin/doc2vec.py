# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 09:46:21 2020

@author: sgmjin2
"""

import sys
import re
import os
import logging, logging.handlers
import warnings
warnings.filterwarnings('ignore') 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')

import pandas as pd
import numpy as np
import gensim
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE

from splunk import setupSplunkLogger
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_coordinate(vectors):
    "Creates and TSNE model and plots it"
    tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    z = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        z.append(value[2])
        
    return x,y,z

@Configuration(local=True)
class Doc2vec(StreamingCommand):
    textfield = Option(
        require=True,
        doc='''
        **Syntax:** **textfield=***<fieldname>*
        **Description:** Name of the field that will contain the text to search against''',
        )



    def stream(self, records):
        #load the word embedding
        word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True,limit=500000)

        tokenizer = RegexpTokenizer(r'\w+')

        vectors=[]
        for record in records:
            words=tokenizer.tokenize(record[self.textfield])
            average_vec=get_average_word2vec(words,word2vec)
            vectors.append(average_vec)
            #record['vec']=average_vec
            yield record
        
        xs, ys, zs = get_coordinate(vectors)
        dict={'xs':xs, 'ys':ys, 'zs':zs}
        df = pd.DataFrame(dict)
        df.to_csv('coordinates.csv', encoding='utf-8', index=False)

dispatch(Doc2vec, sys.argv, sys.stdin, sys.stdout, __name__)
